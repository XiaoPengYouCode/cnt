//! n-gram 打分端口：beam 内逐步打分所需的最小能力集合。
//!
//! 只暴露解码热路径真正需要的三件事：词 → 词表下标、下标查 unigram、下标查 bigram。
//! 「用下标而不是字符串」是刻意的：beam 每按键要打上千次分，字符串查找会成为瓶颈，
//! 词表下标在候选词缓存阶段解析一次即可复用。

/// 语言模型词表下标（unigram 表中的序号）。
///
/// 打分热路径只传下标，不传字符串——避免每次展开重复做二分查找。
pub type WordId = u32;

/// n-gram 语言模型打分能力（由 `cnt-lm::CntLm` 实现）。
///
/// 实现方只需保证查询是**只读且线程安全**的（mmap 天然满足）。
pub trait NgramLm: Send + Sync {
    /// 词 → 词表下标；不在词表返回 `None`。
    fn word_index(&self, word: &str) -> Option<WordId>;

    /// 按下标取 unigram：`(log10 概率, backoff 权重)`。
    fn unigram_by_id(&self, id: WordId) -> Option<(f32, f32)>;

    /// 按下标取 bigram `log10 P(cur | prev)`；未登录该二元组返回 `None`。
    fn bigram_by_id(&self, prev: WordId, cur: WordId) -> Option<f32>;

    /// 按词文本取 unigram（冷路径便利方法：补全候选、诊断工具）。
    fn unigram(&self, word: &str) -> Option<(f32, f32)> {
        self.word_index(word).and_then(|id| self.unigram_by_id(id))
    }

    /// 条件概率 `log10 P(cur | prev)`，缺失时按 Katz backoff 回退：
    /// `logP(cur) + backoff(prev)`；任一词不在词表时该项按 `unk` / `0.0` 处理。
    ///
    /// 这是 n-gram 的领域规则（不是解码器的），因此放在端口里作为默认实现，
    /// 让所有 n-gram 实现共享同一套回退语义。
    fn conditional(&self, prev: Option<WordId>, cur: Option<WordId>, unk: f32) -> f32 {
        // 命中 bigram 时提前返回：backoff 路径的两次 unigram 查询是热路径开销，
        // 不能提前求值（每按键上千次展开都会走这里）。
        if let (Some(p), Some(c)) = (prev, cur)
            && let Some(logp) = self.bigram_by_id(p, c)
        {
            return logp;
        }
        let backoff = prev.map_or(0.0, |p| self.unigram_by_id(p).map_or(0.0, |(_, b)| b));
        let logp = cur.map_or(unk, |c| self.unigram_by_id(c).map_or(unk, |(p, _)| p));
        logp + backoff
    }
}
