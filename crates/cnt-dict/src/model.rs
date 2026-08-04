//! 查询模型：把静态词库 + 用户调频混合排序。
//!
//! 打分公式（一般输入法的通行做法）：
//! ```text
//! score(词 | 拼音) = base_freq(词) + USER_BOOST × user_count(拼音, 词)
//! ```
//! 即：基础词频为主，用户每多选一次该候选，权重 +`USER_BOOST`，让它逐渐往前挤。

use std::io;
use std::path::Path;
use std::sync::Mutex;

use cnt_store::StoreError;
use crate::mmap_dict::MmapDict;
use crate::user::UserDb;

/// 单次查询返回的候选上限。
pub const QUERY_LIMIT: usize = 50;

/// 用户计数换算成基础词频的权重。
pub const USER_BOOST: u32 = 1000;

/// 供输入逻辑使用的查询接口（与具体实现解耦，便于测试/替换）。
pub trait DictQuery {
    fn query(&self, pinyin: &str) -> Vec<String>;
}

/// 默认词库文件名（在用户数据目录下）。
pub const DEFAULT_DICT_FILE: &str = "dict.cntd";
/// 默认用户数据文件名。
pub const DEFAULT_USER_FILE: &str = "user.dict";

/// 共享查询模型：不可变词库 + 可变用户库（Mutex 保护）。
///
/// 多个 `IBus` Engine 实例（= 多个输入框）共享同一个 `Arc<PinyinModel>`。
pub struct PinyinModel {
    dict: MmapDict,
    user: Mutex<UserDb>,
}

impl PinyinModel {
    /// 打开词库与用户库。
    ///
    /// # Errors
    /// 词库文件无法打开/格式非法，或用户库读取出错时返回 [`StoreError`]。
    pub fn open(
        dict_path: impl AsRef<Path>,
        user_path: impl AsRef<Path>,
    ) -> Result<Self, StoreError> {
        let dict = MmapDict::open(dict_path)?;
        let user = UserDb::open(user_path)?;
        Ok(Self {
            dict,
            user: Mutex::new(user),
        })
    }

    /// 查询候选（已按分数降序、去重、截断）。
    ///
    /// 包含用户词库：学过的复合词/新词在「精确」与「前缀」输入阶段都参与排序，
    /// 这样输入到一半（如 chijiuh）时学过的词（持久化）也能出现。
    pub fn query(&self, pinyin: &str) -> Vec<String> {
        let user = self.user.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // word -> (score, 是否精确匹配；精确匹配在平分时优先)
        let mut scored: std::collections::HashMap<String, (u64, bool)> = std::collections::HashMap::new();

        for c in self.dict.exact(pinyin) {
            let cnt = user.count(pinyin, c.word);
            let score = u64::from(c.freq) + u64::from(cnt) * u64::from(USER_BOOST);
            push(&mut scored, c.word, score, true);
        }
        for h in self.dict.prefix(pinyin) {
            let cnt = user.count(h.key, h.word);
            let score = u64::from(h.freq) + u64::from(cnt) * u64::from(USER_BOOST);
            push(&mut scored, h.word, score, false);
        }
        // 用户词：精确 + 前缀（学过的复合词/新词；词典里没有的也参与）
        for w in user.words_for_pinyin(pinyin) {
            let cnt = user.count(pinyin, &w);
            let score = u64::from(cnt) * u64::from(USER_BOOST);
            push(&mut scored, &w, score, true);
        }
        for (p, w) in user.words_for_pinyin_prefix(pinyin) {
            let cnt = user.count(&p, &w);
            let score = u64::from(cnt) * u64::from(USER_BOOST);
            push(&mut scored, &w, score, false);
        }
        drop(user); // 锁只覆盖计数读取阶段，排序前释放

        let mut out: Vec<(String, u64, bool)> = scored
            .into_iter()
            .map(|(w, (s, e))| (w, s, e))
            .collect();
        out.sort_by(|a, b| {
            b.1.cmp(&a.1) // 分数降序
                .then_with(|| b.2.cmp(&a.2)) // 平分时精确优先
                .then_with(|| a.0.cmp(&b.0)) // 再按词字典序（确定性）
        });
        out.truncate(QUERY_LIMIT);
        out.into_iter().map(|(w, _, _)| w).collect()
    }

    /// 只读词库（供解码器按音节精确取词）。
    #[must_use]
    pub const fn dict(&self) -> &MmapDict {
        &self.dict
    }

    /// 用户对 (拼音, 词) 的累计选择次数（供解码器做调频加成）。
    #[must_use]
    pub fn user_count(&self, pinyin: &str, word: &str) -> u32 {
        self.user
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .count(pinyin, word)
    }

    /// 用户词库中某个拼音下的所有词（含调频的词典词与自造新词，按有效计数降序）。
    #[must_use]
    pub fn user_words_for(&self, pinyin: &str) -> Vec<String> {
        self.user
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .words_for_pinyin(pinyin)
    }

    /// 某个拼音/词键的高频候选词：词典精确词 + 用户词（含新词），
    /// 按「词频 + 用户调频」排序取前 N（领域规则单点定义，解码器复用）。
    /// 返回 `(词, 词库频率)`：频率 0 表示用户新词（不在词库），
    /// 频率 1 表示次读音（多音字低频读音）。
    #[must_use]
    pub fn ranked_words(&self, key: &str, limit: usize) -> Vec<(String, u32)> {
        let mut scored: Vec<(String, u32)> = self
            .dict
            .exact(key)
            .into_iter()
            .map(|c| (c.word.to_string(), c.freq))
            .collect();
        for w in self.user_words_for(key) {
            if !scored.iter().any(|(x, _)| x == &w) {
                scored.push((w, 0));
            }
        }
        scored.sort_by(|a, b| {
            let sa = u64::from(a.1) + u64::from(self.user_count(key, &a.0)) * u64::from(USER_BOOST);
            let sb = u64::from(b.1) + u64::from(self.user_count(key, &b.0)) * u64::from(USER_BOOST);
            sb.cmp(&sa).then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);
        scored
    }

    /// 用户选中了候选 (pinyin, word)：记录调频。
    pub fn bump(&self, pinyin: &str, word: &str) {
        self.user.lock().unwrap_or_else(std::sync::PoisonError::into_inner).bump(pinyin, word);
    }

    /// 持久化用户数据（由定时任务/退出时调用）。
    ///
    /// # Errors
    /// 写盘失败（IO）时返回错误。
    pub fn flush_user(&self) -> io::Result<()> {
        self.user.lock().unwrap_or_else(std::sync::PoisonError::into_inner).flush()
    }
}

impl DictQuery for PinyinModel {
    fn query(&self, pinyin: &str) -> Vec<String> {
        Self::query(self, pinyin)
    }
}

fn push(
    scored: &mut std::collections::HashMap<String, (u64, bool)>,
    word: &str,
    score: u64,
    exact: bool,
) {
    match scored.get_mut(word) {
        Some(e) => {
            if score > e.0 || (score == e.0 && exact && !e.1) {
                *e = (score, exact);
            }
        }
        None => {
            scored.insert(word.to_string(), (score, exact));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::build;

    fn model_with(name: &str, pairs: &[(&str, &str, u32)]) -> PinyinModel {
        let pairs: Vec<(String, String, u32)> = pairs
            .iter()
            .map(|(p, w, f)| (p.to_string(), w.to_string(), *f))
            .collect();
        let bytes = build(&pairs).unwrap();
        let dir = std::env::temp_dir().join(format!("cnt-model-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dict_path = dir.join("dict.cntd");
        let user_path = dir.join("user.dict");
        std::fs::write(&dict_path, &bytes).unwrap();
        PinyinModel::open(&dict_path, &user_path).unwrap()
    }

    #[test]
    fn base_order_is_freq_desc() {
        let m = model_with(
            "base",
            &[
                ("zhan", "站", 100),
                ("zhan", "栈", 900),
                ("zhan", "占", 500),
            ],
        );
        let q = m.query("zhan");
        assert_eq!(q, vec!["栈", "占", "站"]);
    }

    #[test]
    fn user_boost_promotes_selected_word() {
        let m = model_with(
            "boost",
            &[("zhan", "站", 900), ("zhan", "栈", 100)],
        );
        assert_eq!(m.query("zhan"), vec!["站", "栈"]);
        m.bump("zhan", "栈");
        m.bump("zhan", "栈");
        // 栈: 100 + 2*1000 = 2100 > 站: 900
        assert_eq!(m.query("zhan"), vec!["栈", "站"]);
    }

    #[test]
    fn prefix_and_exact_dedup() {
        let m = model_with(
            "dedup",
            &[
                ("ni", "你", 900),
                ("nihao", "你好", 800),
                ("ni", "泥", 100),
            ],
        );
        let q = m.query("ni");
        assert!(q.contains(&"你".to_string()));
        assert!(q.contains(&"你好".to_string()));
        // 去重：你 只出现一次
        assert_eq!(q.iter().filter(|w| *w == "你").count(), 1);
        // 精确 > 前缀，且 你(900) 应排在 你好(800) 前
        assert_eq!(q[0], "你");
        assert_eq!(q[1], "你好");
    }
}
