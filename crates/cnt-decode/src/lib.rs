//! cnt-decode —— 拼音切分 + 整句解码。

//! 结构上分成三块：
//! - `syllable`：拼音 → 音节格（词图构建）
//! - `decoder`：beam search 搜索骨架（打分能力由 `cnt-score` 端口注入）
//! - `rescore`：可选的整句重排阶段（`cnt-score::Rescorer` 编排）

pub mod decoder;
pub(crate) mod rescore;
pub mod syllable;

pub use cnt_score::{NgramLm, RescorePolicy, Rescorer, Segment, SentenceHyp};
pub use decoder::Decoder;
