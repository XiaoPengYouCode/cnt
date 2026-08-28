//! 文本级语言模型打分：给一段**已经是汉字**的文本算 n-gram 对数概率。
//!
//! 拼音解码天生知道分词（词键就是分词），但语音识别的输出是一串裸汉字，
//! 要用词级 n-gram 给它打分，得先**自己切词**。所以这里做一次维特比分词：
//!
//! ```text
//!   「输入法不准」 ─► 枚举每个位置的候选词（长度 1..=MAX，用 LM 词表判定是否成词）
//!                  ─► DP 取最大 logP 的切分 ─► 返回该切分的分数
//!                     输入法 | 不准     ≈ -7.2      ← 胜出
//!                     输 入 法 不 准    ≈ -14.9
//! ```
//!
//! 这个分数的用途是**在若干条候选文本之间比较**（ASR 的 n-best 重排），
//! 不是绝对概率，所以只要口径一致就有意义。
//!
//! 为什么放在 `cnt-score`：它是纯算法 + 只依赖 [`NgramLm`] 端口，
//! 与「谁来打分」（`cnt-lm` 还是将来的神经模型）和「谁要打分」
//! （拼音解码还是语音识别）都无关。

use crate::lm::NgramLm;

/// 切词时尝试的最大词长（汉字数）。4 覆盖绝大多数中文词与成语。
pub const MAX_WORD_CHARS: usize = 4;
/// 未登录字的对数概率地板：比任何真实 unigram 都低，但不至于让整条路径崩掉。
///
/// 取 -6.0（log10）而不是 -12：语音识别的输出里出现生僻字是正常的，
/// 一个 -12 的悬崖会让「含生僻字但整体正确」的候选永远输给「全是常用字但错」的候选。
pub const OOV_CHAR_LOGP: f32 = -6.0;

/// 一段文本的 n-gram 对数概率（log10），以及它对应的最优切分。
#[derive(Debug, Clone, PartialEq)]
pub struct TextScore {
    /// 最优切分下的总对数概率。
    pub logp: f32,
    /// 最优切分（按顺序的词）。
    pub words: Vec<String>,
    /// 文本总字数。
    pub chars: usize,
    /// 其中**未登录**（不在 LM 词表里）的字数。
    ///
    /// 这个数字是判断「这个分数可不可信」的依据：给日语句子用中文 n-gram 打分时
    /// 假名全部未登录，分数只反映「文本有多长」而不是「有多像一句话」——
    /// 此时应当拒绝打分，而不是给出一个会导致截断的假分数。
    pub oov_chars: usize,
}

impl TextScore {
    /// 未登录字占比（0.0~1.0）。空文本返回 1.0（视为完全不可信）。
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // 字数量级 ≤ 1e3
    pub fn oov_ratio(&self) -> f32 {
        if self.chars == 0 {
            return 1.0;
        }
        self.oov_chars as f32 / self.chars as f32
    }
}

/// 维特比分词 + n-gram 打分。
///
/// `text` 应当是已经归一化的文本（无空格更好；空白与标点会被当作词边界）。
#[must_use]
pub fn score_text<L: NgramLm + ?Sized>(lm: &L, text: &str) -> TextScore {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return TextScore {
            logp: 0.0,
            words: Vec::new(),
            chars: 0,
            oov_chars: 0,
        };
    }
    let n = chars.len();
    // best[i] = 覆盖前 i 个字的最优 (分数, 上一个断点, 该词的 LM id)
    let mut best: Vec<Option<(f32, usize, Option<u32>)>> = vec![None; n + 1];
    best[0] = Some((0.0, 0, None));

    // 逐字节位置预计算，避免在内层反复 collect
    let mut buf = String::with_capacity(MAX_WORD_CHARS * 4);
    for end in 1..=n {
        for len in 1..=MAX_WORD_CHARS.min(end) {
            let start = end - len;
            let Some((prev_score, _, prev_id)) = best[start] else {
                continue;
            };
            buf.clear();
            buf.extend(&chars[start..end]);
            let id = lm.word_index(&buf);
            // 只有单字允许未登录（多字未登录不是词，不该参与切分）
            if id.is_none() && len > 1 {
                continue;
            }
            let step = id.map_or(OOV_CHAR_LOGP, |cur| {
                lm.conditional(prev_id, Some(cur), OOV_CHAR_LOGP)
            });
            let score = prev_score + step;
            if best[end].is_none_or(|(cur, _, _)| score > cur) {
                best[end] = Some((score, start, id));
            }
        }
    }

    // 回溯最优切分
    let Some((logp, _, _)) = best[n] else {
        // 走不到这里（单字总能落地），保守兜底：每字按地板算
        #[allow(clippy::cast_precision_loss)]
        let floor = OOV_CHAR_LOGP * n as f32;
        return TextScore {
            logp: floor,
            words: text.chars().map(String::from).collect(),
            chars: n,
            oov_chars: n,
        };
    };
    let mut words = Vec::new();
    let mut oov_chars = 0usize;
    let mut end = n;
    // 反向可达（best[n] 有值即每一步都有前驱），但仍写成不会 panic 的形式
    while end > 0 {
        let Some((_, start, id)) = best[end] else {
            break;
        };
        if id.is_none() {
            oov_chars += end - start;
        }
        words.push(chars[start..end].iter().collect::<String>());
        end = start;
    }
    words.reverse();
    TextScore {
        logp,
        words,
        chars: n,
        oov_chars,
    }
}

#[cfg(test)]
mod tests {
    use super::{OOV_CHAR_LOGP, score_text};
    use crate::lm::{NgramLm, WordId};

    /// 假 LM：词表 + 手写 unigram/bigram，用来验证 DP 而不依赖真实模型。
    struct FakeLm {
        words: Vec<&'static str>,
        /// (prev, cur) → logP
        bigrams: Vec<(usize, usize, f32)>,
    }

    impl FakeLm {
        fn new() -> Self {
            Self {
                // 0:输入法 1:不准 2:输 3:入 4:法 5:不 6:准 7:我 8:的
                words: vec!["输入法", "不准", "输", "入", "法", "不", "准", "我", "的"],
                bigrams: vec![(0, 1, -1.0)],
            }
        }
    }

    impl NgramLm for FakeLm {
        fn word_index(&self, word: &str) -> Option<WordId> {
            self.words
                .iter()
                .position(|w| *w == word)
                .map(|i| u32::try_from(i).expect("小词表"))
        }
        fn unigram_by_id(&self, id: WordId) -> Option<(f32, f32)> {
            // 多字词 -3，单字 -2.5（单字更常见，正是长度偏置的来源）
            let word = self.words.get(id as usize)?;
            let logp = if word.chars().count() > 1 { -3.0 } else { -2.5 };
            Some((logp, 0.0))
        }
        fn bigram_by_id(&self, prev: WordId, cur: WordId) -> Option<f32> {
            self.bigrams
                .iter()
                .find(|(p, c, _)| *p == prev as usize && *c == cur as usize)
                .map(|(_, _, s)| *s)
        }
    }

    #[test]
    fn empty_text_scores_zero() {
        let s = score_text(&FakeLm::new(), "");
        assert!((s.logp - 0.0).abs() < f32::EPSILON);
        assert!(s.words.is_empty());
    }

    #[test]
    fn prefers_word_segmentation_over_characters() {
        // 输入法(-3) + 不准(bigram -1) = -4  优于  输+入+法+不+准(5×-2.5 = -12.5)
        let s = score_text(&FakeLm::new(), "输入法不准");
        assert_eq!(s.words, vec!["输入法", "不准"]);
        assert!(s.logp > -6.0, "词切分应明显优于逐字：{}", s.logp);
    }

    #[test]
    fn unknown_chars_get_floor_not_cliff() {
        let s = score_text(&FakeLm::new(), "鱛");
        assert!((s.logp - OOV_CHAR_LOGP).abs() < 0.01);
        assert_eq!(s.words, vec!["鱛"]);
    }

    #[test]
    fn oov_ratio_flags_untrustworthy_scores() {
        let lm = FakeLm::new();
        // 全是词表里的词 → 完全可信
        let good = score_text(&lm, "输入法不准");
        assert_eq!(good.chars, 5);
        assert_eq!(good.oov_chars, 0);
        assert!((good.oov_ratio() - 0.0).abs() < f32::EPSILON);
        // 全是未登录字（比如给中文 LM 喂日语假名）→ 完全不可信
        let bad = score_text(&lm, "ををを");
        assert_eq!(bad.oov_chars, 3);
        assert!((bad.oov_ratio() - 1.0).abs() < f32::EPSILON);
        // 空文本按不可信处理
        assert!((score_text(&lm, "").oov_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn shorter_text_wins_when_everything_is_oov() {
        // 这正是「日语句子被截断」的机制：未登录地板让越短的文本分越高。
        // 分数本身没错，错在**拿它做决定**——所以调用方必须看 oov_ratio。
        let lm = FakeLm::new();
        let long = score_text(&lm, "ををを").logp;
        let short = score_text(&lm, "をを").logp;
        assert!(short > long);
    }

    #[test]
    fn multi_char_non_words_are_not_segments() {
        // 「我的」不在词表 → 只能切成 我|的，不能当成一个词
        let s = score_text(&FakeLm::new(), "我的");
        assert_eq!(s.words, vec!["我", "的"]);
    }

    #[test]
    fn mixed_known_and_unknown() {
        let s = score_text(&FakeLm::new(), "输入法鱛不准");
        assert_eq!(s.words, vec!["输入法", "鱛", "不准"]);
    }

    #[test]
    fn longer_text_scores_lower_but_monotonically() {
        let lm = FakeLm::new();
        let short = score_text(&lm, "不准").logp;
        let long = score_text(&lm, "输入法不准").logp;
        assert!(long < short, "更长的文本累积概率更低（这是 n-gram 的性质）");
    }
}
