//! cnt-lm —— n-gram 语言模型。
//!
//! 与词库（cnt-dict）同一套哲学：构建期把 ARPA 文本编译成面向 mmap 的
//! 二进制（`.cntl`），运行期零拷贝二分查找。
//!
//! 格式只做 unigram + bigram（trigram 留作将来扩展，Katz backoff 已预留）。
//!
//! 打分能力通过 `cnt-score::NgramLm` 端口暴露（见 `port` 模块），解码器只依赖
//! 端口，不依赖本 crate 的具体类型。

pub mod format;
pub mod mmap;
pub mod port;
pub mod writer;

pub use cnt_store::StoreError;
pub use mmap::CntLm;
pub use writer::{build, Bigram, Unigram};
