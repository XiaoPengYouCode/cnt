//! 二进制存储错误类型（显式，thiserror）。
//!
//! `cnt-dict` 与 `cnt-lm` 共用：两种格式（.cntd / .cntl）的魔数/版本不同，
//! 但错误形态一致。

use std::io;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// 文件头魔数不对（不是预期的二进制存储文件）
    #[error("bad magic (not a cnt store file)")]
    BadMagic,
    /// 不支持的格式版本
    #[error("unsupported format version {0}")]
    UnsupportedVersion(u32),
    /// 数据被截断（哪个区域）
    #[error("truncated {0}")]
    Truncated(&'static str),
    /// 区域越界（文件损坏）
    #[error("region out of bounds: {0}")]
    Region(&'static str),
    /// 超出格式上限（防止算术溢出）
    #[error("exceeds format limit: {0}")]
    LimitExceeded(&'static str),
    /// IO 错误
    #[error(transparent)]
    Io(#[from] io::Error),
}
