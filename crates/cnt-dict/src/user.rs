//! 用户个性化数据：动态调频 + 用户词库（新词学习） + 时间衰减。
//!
//! 持久化为明文 tsv（Rime 风格，可读可备份可手改）：
//! ```text
//! zhan<TAB>栈<TAB>3<TAB>1780000000
//! ```
//! 第 4 列为上次更新的 Unix 秒（jiff 时间戳）；旧格式 3 列兼容（视为刚更新）。
//!
//! **时间衰减**：计数按半衰期衰减，`effective = count × 0.5^(天数/半衰期)`，
//! 让用户模型跟随当前习惯漂移，旧习惯逐渐淡出。
//! **词库上限**：`MAX_USER_ENTRIES` 封顶，超出时淘汰有效计数最低的词条。
//!
//! 原子写盘：先写临时文件再 rename，避免中途崩溃写坏数据。

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// 单条计数上限，防止个别词被刷到永远第一。
const MAX_COUNT: u32 = 100;
/// 时间衰减半衰期（天）：30 天。
const HALF_LIFE_DAYS: f32 = 30.0;
/// 用户词库总条数上限。
const MAX_USER_ENTRIES: usize = 20_000;
/// 一天的秒数。
const DAY_SECS: i64 = 86_400;

#[derive(Debug, Clone, Copy)]
struct Entry {
    count: u32,
    updated_at: i64,
}

pub struct UserDb {
    path: PathBuf,
    counts: HashMap<(String, String), Entry>,
    dirty: bool,
}

impl UserDb {
    /// 打开（若文件不存在则从空开始）。
    ///
    /// # Errors
    /// 文件读取失败时返回 IO 错误；格式非法的行会被忽略。
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut counts = HashMap::new();
        let now = now_secs();
        if let Ok(content) = fs::read_to_string(&path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let mut it = line.split('\t');
                let (Some(pinyin), Some(word), Some(count)) = (it.next(), it.next(), it.next())
                else {
                    continue;
                };
                let Ok(count) = count.trim().parse::<u32>() else {
                    continue;
                };
                // 第 4 列时间戳可选；缺失时视为刚更新（不衰减）
                let updated_at = it
                    .next()
                    .and_then(|t| t.trim().parse::<i64>().ok())
                    .unwrap_or(now);
                counts.insert(
                    (pinyin.to_string(), word.to_string()),
                    Entry {
                        count: count.min(MAX_COUNT),
                        updated_at,
                    },
                );
            }
        }
        Ok(Self {
            path,
            counts,
            dirty: false,
        })
    }

    /// 用户对 (拼音, 词) 的累计选择次数（经时间衰减后的有效值）。
    #[must_use]
    pub fn count(&self, pinyin: &str, word: &str) -> u32 {
        self.counts
            .get(&(pinyin.to_string(), word.to_string()))
            .map_or(0, effective_u32)
    }

    /// 用户词库中拼音以 `prefix` 开头的所有 `(完整拼音, 词)`，按有效计数降序。
    /// 用于前缀补全：学过的复合词（如 chijiuhua/持久化）在输入到一半时也能出现。
    #[must_use]
    pub fn words_for_pinyin_prefix(&self, prefix: &str) -> Vec<(String, String)> {
        let mut out: Vec<(String, String, u32)> = self
            .counts
            .iter()
            .filter(|((p, _), _)| p.starts_with(prefix))
            .map(|((p, w), e)| (p.clone(), w.clone(), effective_u32(e)))
            .collect();
        out.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        out.into_iter().map(|(p, w, _)| (p, w)).collect()
    }

    /// 用户词库中某个拼音下的所有词（按有效计数降序）。
    /// 含用户调频的词典词与用户自造的新词。
    #[must_use]
    pub fn words_for_pinyin(&self, pinyin: &str) -> Vec<String> {
        let mut out: Vec<(String, u32)> = self
            .counts
            .iter()
            .filter(|((p, _), _)| p == pinyin)
            .map(|((_, w), e)| (w.clone(), effective_u32(e)))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.into_iter().map(|(w, _)| w).collect()
    }

    /// 用户选择了候选 (拼音, 词)：计数 +1（先按时间衰减，再封顶 `MAX_COUNT`）。
    pub fn bump(&mut self, pinyin: &str, word: &str) {
        let key = (pinyin.to_string(), word.to_string());
        let now = now_secs();
        if let Some(e) = self.counts.get_mut(&key) {
            // 旧计数先衰减到当前有效值，再 +1
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let effective = effective(e).round() as u32; // 衰减值 ≥ 0
            e.count = effective.saturating_add(1).min(MAX_COUNT);
            e.updated_at = now;
        } else {
            // 新词：受总条数上限约束，超出时淘汰最低有效计数词条
            if self.counts.len() >= MAX_USER_ENTRIES {
                self.evict_lowest();
                if self.counts.len() >= MAX_USER_ENTRIES {
                    return; // 淘汰后仍满（新词本身是最低的），放弃
                }
            }
            self.counts.insert(key, Entry { count: 1, updated_at: now });
        }
        self.dirty = true;
    }

    /// 淘汰有效计数最低的一条（保持上限）。
    fn evict_lowest(&mut self) {
        let Some((key, _)) = self
            .counts
            .iter()
            .min_by(|(_, a), (_, b)| {
                effective(a)
                    .total_cmp(&effective(b))
                    .then_with(|| a.updated_at.cmp(&b.updated_at))
            })
            .map(|(k, v)| (k.clone(), *v))
        else {
            return;
        };
        self.counts.remove(&key);
    }

    /// 持久化（仅在有改动时写盘；临时文件 + rename 原子替换）。
    ///
    /// # Errors
    /// 目录创建/写文件/rename 失败时返回 IO 错误。
    pub fn flush(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("tmp");
        {
            let f = BufWriter::new(File::create(&tmp)?);
            let mut w = f;
            for ((pinyin, word), e) in &self.counts {
                writeln!(w, "{pinyin}\t{word}\t{}\t{}", e.count, e.updated_at)?;
            }
            w.flush()?;
        }
        fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    /// 用户词条总数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// 用户词条是否为空。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// 当前 Unix 秒（jiff）。
fn now_secs() -> i64 {
    jiff::Timestamp::now().as_second()
}

/// 有效计数：按时间衰减。
///
/// 用户计数 ≤ 100，u32→f32 无损（clippy 允许）。
#[allow(clippy::cast_precision_loss)]
fn effective(e: &Entry) -> f32 {
    let days = (now_secs().saturating_sub(e.updated_at)) as f32 / DAY_SECS as f32;
    if days <= 0.0 {
        e.count as f32
    } else {
        e.count as f32 * 0.5f32.powf(days / HALF_LIFE_DAYS)
    }
}

/// 有效计数的整数形式（衰减值 ≥ 0，截断/符号丢失在此是预期语义）。
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn effective_u32(e: &Entry) -> u32 {
    effective(e) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_and_count() {
        let mut db = UserDb::open(Path::new("/nonexistent/cnt-user.dict")).unwrap();
        assert_eq!(db.count("zhan", "栈"), 0);
        db.bump("zhan", "栈");
        db.bump("zhan", "栈");
        assert_eq!(db.count("zhan", "栈"), 2);
        assert!(db.dirty);
    }

    #[test]
    fn flush_roundtrip_with_timestamp() {
        let dir = std::env::temp_dir().join(format!("cnt-user-{}", std::process::id()));
        let path = dir.join("user.dict");
        let _ = fs::remove_dir_all(&dir);
        let mut db = UserDb::open(&path).unwrap();
        db.bump("zhan", "栈");
        db.bump("zhan", "栈");
        db.bump("ni", "你");
        db.flush().unwrap();
        // 重新打开
        let db2 = UserDb::open(&path).unwrap();
        assert_eq!(db2.count("zhan", "栈"), 2);
        assert_eq!(db2.count("ni", "你"), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn old_format_without_timestamp_is_compatible() {
        let dir = std::env::temp_dir().join(format!("cnt-user-old-{}", std::process::id()));
        let path = dir.join("user.dict");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, "zhan\t栈\t5\n").unwrap(); // 3 列旧格式
        let db = UserDb::open(&path).unwrap();
        assert_eq!(db.count("zhan", "栈"), 5); // 视为刚更新，不衰减
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn words_for_pinyin_lists_user_words() {
        let mut db = UserDb::open(Path::new("/nonexistent/cnt-user.dict")).unwrap();
        db.bump("zhan", "栈");
        db.bump("zhan", "栈");
        db.bump("zhan", "站");
        db.bump("ni", "你");
        let ws = db.words_for_pinyin("zhan");
        assert_eq!(ws, vec!["栈".to_string(), "站".to_string()]); // 栈计数更高
        assert!(db.words_for_pinyin("ni").contains(&"你".to_string()));
    }

    #[test]
    fn cap_evicts_lowest() {
        let mut db = UserDb::open(Path::new("/nonexistent/cnt-user.dict")).unwrap();
        // 填满上限（每条 bump 多次保证有效计数不同）
        for i in 0..MAX_USER_ENTRIES {
            db.bump(&format!("p{i}"), &format!("w{i}"));
        }
        assert!(db.len() <= MAX_USER_ENTRIES);
        // 淘汰后应能继续插入新词
        db.bump("new", "词");
        assert!(db.len() <= MAX_USER_ENTRIES);
        assert_eq!(db.count("new", "词"), 1);
    }
}
