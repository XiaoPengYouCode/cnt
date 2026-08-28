//! cnt-store —— 二进制 mmap 存储的公共底座。
//!
//! `cnt-dict`（词库 .cntd）与 `cnt-lm`（语言模型 .cntl）共享同一套
//! 「mmap + 定长小端表 + 字符串池」存储模式，本 crate 抽出公共部分：
//! 错误类型、字节读取、mmap 文件包装、区域校验。

pub mod bytes;
pub mod error;
pub mod mmap;

pub use bytes::{narrow, read_f32, read_u16, read_u32, read_u64};
pub use error::StoreError;
pub use mmap::{MmapFile, validate_regions};
