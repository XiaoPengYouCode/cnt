//! 查询模型：把静态词库 + 用户调频混合排序。
//!
//! 打分公式（一般输入法的通行做法）：
//! ```text
//! score(词 | 拼音) = base_freq(词) + USER_BOOST × user_count(拼音, 词)
//! ```
//! 即：基础词频为主，用户每多选一次该候选，权重 +`USER_BOOST`，让它逐渐往前挤。

use std::borrow::Cow;
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

/// 前缀候选（键比输入更长，如输入 `lian` 命中 `lianxi`/`lianghao`）的词频折扣。
///
/// 前缀词是「猜测用户还没打完」的候选，不该与精确读音的词平起平坐：
/// 联系(24126)/良好(24121) 原先按全额词频压过 `lian` 下的 恋/链/帘，
/// 让精确同音字全被挤出前 10。折扣 4 让高频前缀词仍靠前，但排在常用精确词之后。
pub const PREFIX_DISCOUNT: u64 = 4;

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
    /// 词库是否存在该拼音键。
    #[must_use]
    pub fn has_key(&self, key: &str) -> bool {
        self.dict.contains_key(key)
    }

    /// 词库或用户库是否存在以该拼音为前缀的键（多音节链剪枝用：中间前缀可能无
    /// 独立词条，但更长键存在，如 xuangai → xuangaiji）。
    ///
    /// 必须带上用户库：否则用户学过的复合词（郑爽 / zhengshuang）只能在
    /// 「整串输入 == 该键」时由整键候选命中，beam 不会把它当作句子中间的一个词。
    #[must_use]
    pub fn has_key_prefix(&self, key: &str) -> bool {
        self.dict.contains_prefix(key)
            || self
                .user
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .has_key_prefix(key)
    }

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
            // 前缀词打折：精确读音的词优先（见 PREFIX_DISCOUNT）
            let score =
                u64::from(h.freq) / PREFIX_DISCOUNT + u64::from(cnt) * u64::from(USER_BOOST);
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
    /// 全部用户词及其最大选择次数（跨拼音键聚合）。
    ///
    /// 给语音侧做热词偏置用：通用声学模型不可能知道用户把哪些词当常用词，
    /// 而这份数据正是用户自己一次次选出来的。
    pub fn user_words(&self) -> Vec<(String, u32)> {
        let mut out: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        {
            let user = self
                .user
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (word, count) in user.all_words() {
                let slot = out.entry(word).or_insert(0);
                *slot = (*slot).max(count);
            }
        }
        out.into_iter().collect()
    }

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
    ///
    /// 返回 `(词, 词库频率)`：频率 0 表示用户新词（不在词库），
    /// 频率 1 表示次读音（多音字低频读音）。
    ///
    /// 词用 `Cow`：词库词直接借 mmap（零拷贝），只有用户词是拥有型 ——
    /// 调用方通常要转成自己的表示（如 `Arc<str>`），少一次 `String` 中转。
    #[must_use]
    pub fn ranked_words<'a>(&'a self, key: &str, limit: usize) -> Vec<(Cow<'a, str>, u32)> {
        // 排序键先算好再排（Schwartzian transform）：此前比较器里调 user_count，
        // 每次比较都要抢一次用户库的锁 + 两次哈希查找 —— 20 个候选就是上百次加锁，
        // 而这是 beam 每个词键都会走的热路径。
        let mut scored: Vec<(Cow<'a, str>, u32, u64)> = Vec::new();
        let mut user_pushed: Vec<(Cow<'a, str>, u32, u64)> = Vec::new();
        {
            let user = self
                .user
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // 外层键只查一次（counts_of），之后按词哈希查找
            let counts = user.counts_of(key);
            let rank_of = |word: &str, freq: u32| {
                u64::from(freq) + u64::from(counts.get(word)) * u64::from(USER_BOOST)
            };
            for c in self.dict.exact(key) {
                scored.push((Cow::Borrowed(c.word), c.freq, rank_of(c.word, c.freq)));
            }
            for w in user.words_for_pinyin(key) {
                if !scored.iter().any(|(x, _, _)| x.as_ref() == w.as_str()) {
                    let rank = rank_of(&w, 0);
                    scored.push((Cow::Owned(w.clone()), 0, rank));
                    user_pushed.push((Cow::Owned(w), 0, rank));
                }
            }
        } // 锁只覆盖计数读取，排序不持锁
        scored.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        let mut out: Vec<(Cow<'a, str>, u32, u64)> = scored.into_iter().take(limit).collect();
        // 用户词恒可见（对齐 librime：用户短语权重再低也出现在候选里，可被再次选择确认）：
        // 被截断掉的用户词补到末尾 —— 它们 rank 本就最低，补在末尾不破坏排序。
        for uw in &user_pushed {
            if !out.iter().any(|(x, _, _)| *x == uw.0) {
                out.push(uw.clone());
            }
        }
        out.into_iter().map(|(w, freq, _)| (w, freq)).collect()
    }

    /// 用户选中了候选 (pinyin, word)：记录调频。
    ///
    /// `known`（词在不在静态词库）由本层判定：词库已有 → 直接转正；
    /// 不在词库（自动造词）→ 低可信度起点（见 `UserDb::bump`）。
    pub fn bump(&self, pinyin: &str, word: &str) {
        let known = self.dict.exact(pinyin).iter().any(|c| c.word == word);
        self.user
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .bump(pinyin, word, known);
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
