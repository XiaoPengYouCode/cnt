//! cnt-asr-lm —— 用 n-gram + **用户词库**给语音识别的 n-best 打分。
//!
//! 这是一个**适配器**：把语音侧的 [`cnt_asr::TextScorer`] 端口接到打分侧的
//! [`cnt_score::NgramLm`] 端口上，两边谁都不认识谁。
//!
//! ```text
//!   cnt-voice ──► TextScorer（端口，cnt-asr）
//!                     ▲
//!                     │ 实现
//!                 cnt-asr-lm ──► NgramLm（端口，cnt-score）◄── cnt-lm（mmap n-gram）
//!                     │
//!                     └──► 用户词表（热词加成）
//! ```
//!
//! ## 为什么这一层值得存在
//!
//! 声学模型只听得见「音」，分不清同音异形——用户实测的错误几乎全是这一类：
//!
//! ```text
//!   瓶颈 → 平境      语音 → 原音      现在 → 现来      体验感 → 体验觉
//! ```
//!
//! 而输入法手里正好有一份中文 n-gram（27 万 unigram + 480 万 bigram）。
//! 更关键的是还有**用户自己的词库**：通用模型不可能知道你把「工站」「瓶颈」
//! 当常用词，但 `user.dict` 知道。这是本地方案独有的信息，云端输入法拿不到。
//!
//! ## 热词加成的口径与拼音侧共用
//!
//! 加成 = `min(选择次数, 10) × 0.2`（`cnt_score::policy::user`），与拼音候选
//! 排序用的是**同一把尺**——同一个用户的同一份证据，两条链路没有理由采信程度不同。
//!
//! 按次数而不是「有无」：选过 1 次（+0.2）与选过 10 次（+2.0）的可信度差别很大。
//! 而且是加法不是替换：用户词只是证据之一，遇到明显更通顺的句子时应该让 n-gram 赢。

use std::collections::HashMap;
use std::sync::Arc;

use cnt_asr::TextScorer;
use cnt_score::policy::user;
use cnt_score::{NgramLm, score_text};

/// 用户词至少这么长才参与加成（单字太容易误命中）。
pub const MIN_USER_WORD_CHARS: usize = 2;
/// 未登录字占比超过这个值就**拒绝打分**（返回 None，调用方保持声学顺序）。
///
/// 为什么必须有这道闸：中文 n-gram 给日语/韩语句子打分时，假名与谚文全是未登录字，
/// 分数只反映「文本有多长」——于是重排会系统性地选最短的那条，把句尾吃掉
/// （实测：`…パンを買う` → `…パンを買`）。这与拼音侧「不同覆盖长度的假设不可比」
/// 是同一类错误：**不可比的东西不要比**。
///
/// 0.4 的余量允许中文里夹生僻字、少量外文词，但拒绝整句非中文。
pub const MAX_OOV_RATIO: f32 = 0.4;

/// n-gram + 用户词库文本打分器。
pub struct LmTextScorer<L: NgramLm + ?Sized> {
    lm: Arc<L>,
    /// 用户词 → 选择次数（加成按次数增长，与拼音侧同一口径）。
    user_words: HashMap<String, u32>,
    name: String,
}

impl<L: NgramLm + ?Sized> LmTextScorer<L> {
    /// 只用 n-gram（不做热词加成）。
    #[must_use]
    pub fn new(lm: Arc<L>) -> Self {
        Self {
            lm,
            user_words: HashMap::new(),
            name: "ngram".to_owned(),
        }
    }

    /// 带用户词热词加成（`(词, 选择次数)`）。
    #[must_use]
    pub fn with_user_words<I, S>(lm: Arc<L>, words: I) -> Self
    where
        I: IntoIterator<Item = (S, u32)>,
        S: Into<String>,
    {
        let mut user_words: HashMap<String, u32> = HashMap::new();
        for (word, count) in words {
            let word: String = word.into();
            if word.chars().count() < MIN_USER_WORD_CHARS {
                continue;
            }
            // 同一个词可能挂在多个拼音键下，取最大次数
            let slot = user_words.entry(word).or_insert(0);
            *slot = (*slot).max(count);
        }
        let name = format!("ngram+user({})", user_words.len());
        Self {
            lm,
            user_words,
            name,
        }
    }

    /// 文本里命中的用户词加成总额（不重叠，贪心最长匹配）。
    #[must_use]
    pub fn user_bonus(&self, text: &str) -> f32 {
        if self.user_words.is_empty() {
            return 0.0;
        }
        let chars: Vec<char> = text.chars().collect();
        let mut bonus = 0.0;
        let mut i = 0;
        let mut buf = String::new();
        while i < chars.len() {
            let mut matched = 0;
            // 最长匹配优先：避免「工站台」被算成两次
            for len in (MIN_USER_WORD_CHARS..=8.min(chars.len() - i)).rev() {
                buf.clear();
                buf.extend(&chars[i..i + len]);
                if let Some(count) = self.user_words.get(&buf) {
                    bonus += user::boost(*count);
                    matched = len;
                    break;
                }
            }
            i += if matched > 0 { matched } else { 1 };
        }
        bonus
    }

    /// 文本里命中的用户词个数（诊断用）。
    #[must_use]
    pub fn user_hits(&self, text: &str) -> usize {
        let chars: Vec<char> = text.chars().collect();
        let mut hits = 0;
        let mut i = 0;
        let mut buf = String::new();
        while i < chars.len() {
            let mut matched = 0;
            for len in (MIN_USER_WORD_CHARS..=8.min(chars.len() - i)).rev() {
                buf.clear();
                buf.extend(&chars[i..i + len]);
                if self.user_words.contains_key(&buf) {
                    matched = len;
                    break;
                }
            }
            if matched > 0 {
                hits += 1;
            }
            i += if matched > 0 { matched } else { 1 };
        }
        hits
    }
}

impl<L: NgramLm + ?Sized> TextScorer for LmTextScorer<L> {
    fn logp10(&self, text: &str) -> Option<f32> {
        if text.is_empty() {
            return None;
        }
        let _span = fastrace::local::LocalSpan::enter_with_local_parent("lm_score_text");
        let scored = score_text(self.lm.as_ref(), text);
        let oov = scored.oov_ratio();
        if oov > MAX_OOV_RATIO {
            log::debug!("lm: refusing to score {text:?} (oov ratio {oov:.2})");
            return None;
        }
        Some(scored.logp + self.user_bonus(text))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cnt_asr::TextScorer;
    use cnt_score::lm::{NgramLm, WordId};

    use super::LmTextScorer;

    /// 假 LM：只认几个词，用来验证加成与打分的组合逻辑。
    struct FakeLm;

    impl NgramLm for FakeLm {
        fn word_index(&self, word: &str) -> Option<WordId> {
            match word {
                "工" => Some(0),
                "站" => Some(1),
                "公" => Some(2),
                "工站" => Some(3),
                "公站" => Some(4),
                _ => None,
            }
        }
        fn unigram_by_id(&self, id: WordId) -> Option<(f32, f32)> {
            // 「公站」在通用语料里比「工站」常见（这正是需要用户词库纠偏的情形）
            Some(match id {
                3 => (-5.0, 0.0),
                4 => (-4.0, 0.0),
                _ => (-3.0, 0.0),
            })
        }
        fn bigram_by_id(&self, _prev: WordId, _cur: WordId) -> Option<f32> {
            None
        }
    }

    #[test]
    fn scores_text_without_user_words() {
        let s = LmTextScorer::new(Arc::new(FakeLm));
        assert_eq!(s.name(), "ngram");
        let a = s.logp10("工站").expect("有分");
        let b = s.logp10("公站").expect("有分");
        assert!(b > a, "无用户词时，通用语料里更常见的「公站」胜出");
    }

    #[test]
    fn user_word_flips_homophone_when_used_enough() {
        // 选过 1 次（+0.2）翻不过 1.0 的差距——这是有意的：证据弱就别乱改
        let weak = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 1)]);
        assert!(weak.logp10("公站") > weak.logp10("工站"));
        // 选过 10 次（+2.0）就该翻过来
        let strong = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 10)]);
        let a = strong.logp10("工站").expect("有分");
        let b = strong.logp10("公站").expect("有分");
        assert!(a > b, "常用的用户词应把同音对翻过来");
        assert!((strong.user_bonus("工站") - 2.0).abs() < 0.01);
    }

    #[test]
    fn single_char_user_words_are_ignored() {
        let s = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工", 9), ("站", 9)]);
        assert_eq!(s.user_hits("工站"), 0, "单字不参与热词加成");
        assert!((s.user_bonus("工站") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn user_hits_are_non_overlapping_longest_first() {
        let s = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 3), ("工站台", 3)]);
        assert_eq!(s.user_hits("工站台"), 1, "最长匹配优先，不重复计数");
        assert_eq!(s.user_hits("工站和工站"), 2);
        assert_eq!(s.user_hits("完全无关"), 0);
    }

    #[test]
    fn bonus_scales_with_count() {
        let once = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 1)]);
        let many = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 10)]);
        assert!(many.user_bonus("工站") > once.user_bonus("工站"));
        // 封顶：50 次与 10 次一样
        let capped = LmTextScorer::with_user_words(Arc::new(FakeLm), [("工站", 50)]);
        assert!((capped.user_bonus("工站") - many.user_bonus("工站")).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_text_has_no_score() {
        let s = LmTextScorer::new(Arc::new(FakeLm));
        assert!(s.logp10("").is_none());
    }

    #[test]
    fn refuses_to_score_mostly_unknown_text() {
        // 假 LM 只认那几个中文字；「假名」全未登录 → 拒绝打分，
        // 免得重排按「越短越好」把句尾吃掉（线上踩过：日语句子被截断）
        let s = LmTextScorer::new(Arc::new(FakeLm));
        assert!(s.logp10("うちの中学は").is_none());
        // 全中文照常打分
        assert!(s.logp10("工站").is_some());
    }
}
