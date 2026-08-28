//! 整句重排端口（rescoring）：给「已经由动态规划搜出来的 top-N 句子」重新打分。
//!
//! 大厂端上输入法的做法就是这一层：搜索仍然是词图 + beam/Viterbi，
//! 神经模型只作为 **reranker** 介入——既拿到「哪条更像人话」的收益，
//! 又不用让模型承担增量解码与硬约束（用户词、屏蔽词、强制上屏）。

use core::fmt;

/// 候选句的一个片段：`(拼音键, 词)`。
///
/// 重排模型通常只看 `word`，但拼音同时给出，便于做读音一致性/纠错类模型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment<'a> {
    pub pinyin: &'a str,
    pub word: &'a str,
}

/// 一条待重排的候选句。
///
/// 借用形式：重排是冷路径但仍在按键预算内，不做多余的字符串拷贝。
#[derive(Debug, Clone, Copy)]
pub struct SentenceHyp<'a> {
    /// 整句文本（各片段的词拼接）。
    pub text: &'a str,
    /// 分段信息（拼音 → 词）。
    pub segments: &'a [Segment<'a>],
    /// 基线打分器（n-gram + 用户调频 + 各种惩罚）给出的分数，log10 空间。
    pub base_score: f32,
}

/// 整句重排器：对 top-N 候选句给出模型分。
///
/// 约定：
/// - 返回值与 `hyps` **等长、同序**，每项是 log10 空间的模型分（越大越好）；
/// - 返回 `None` 表示本次放弃重排（模型未就绪 / 超时 / 输入不适用），
///   调用方必须保持基线顺序不变——**重排永远不能让输入法变得不可用**；
/// - 实现必须是无副作用、可并发调用的。
pub trait Rescorer: Send + Sync {
    /// 实现名（日志/埋点用，如 `"nnlm-char-6m-int8"`）。
    fn name(&self) -> &'static str;

    /// 给每条候选句打模型分；放弃时返回 `None`。
    fn score_sentences(&self, hyps: &[SentenceHyp<'_>]) -> Option<Vec<f32>>;
}

impl fmt::Debug for dyn Rescorer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rescorer")
            .field("name", &self.name())
            .finish()
    }
}

/// 空实现：永不重排（默认配置 / 关闭开关 / 测试基线）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRescore;

impl Rescorer for NoRescore {
    fn name(&self) -> &'static str {
        "none"
    }

    fn score_sentences(&self, _hyps: &[SentenceHyp<'_>]) -> Option<Vec<f32>> {
        None
    }
}
