//! cnt-score —— 打分端口层（DDD 意义上的领域端口 / dependency inversion）。
//!
//! 输入法的解码 = **搜索** + **打分**。搜索（词图 + beam / Viterbi 动态规划）
//! 是稳定的算法骨架，不该随打分模型变；打分模型则会演进：
//! n-gram → n-gram + 小神经模型重排。所以把「打分」定义成端口，
//! 让解码器只依赖契约、不依赖具体模型：
//!
//! ```text
//!   cnt-lm (n-gram, mmap)  ─┐
//!   cnt-nnlm (小模型, 未来) ─┼─► cnt-score（本 crate：端口/契约/策略）◄─ cnt-decode（搜索）
//! ```
//!
//! 两个端口对应两条延迟等级完全不同的路径：
//!
//! | 端口 | 调用频次 | 分派 | 实现 |
//! |---|---|---|---|
//! | [`NgramLm`] | beam 内每次展开（每按键上千次） | 静态（泛型单态化） | `cnt-lm::CntLm` |
//! | [`Rescorer`] | 每按键 ≤1 次、仅 top-N 条句子 | 动态（`dyn`） | 未来 `cnt-nnlm` |
//!
//! 热路径必须零抽象开销，所以 [`NgramLm`] 用泛型；重排是冷路径且要能在运行期
//! 按配置装卸，用 `dyn` 更合适（一次调用 ≤20 条候选，虚表开销可忽略）。

pub mod lm;
pub mod policy;
pub mod rescore;

pub use lm::{NgramLm, WordId};
pub use policy::RescorePolicy;
pub use rescore::{NoRescore, Rescorer, Segment, SentenceHyp};
