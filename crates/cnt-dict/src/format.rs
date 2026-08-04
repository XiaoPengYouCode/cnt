//! 二进制词库（.cntd）的磁盘布局。
//!
//! 面向 mmap 的零拷贝设计：全部是定长小端整数 + 字节偏移，
//! 读取时直接切片，无需反序列化（也因此不用 bincode —— bincode 面向
//! 「反序列化成拥有型结构」，而这里要的是「驻留内存、原地切片」）。
//!
//! 布局（所有整数小端）：
//! ```text
//! [0..8]    magic       b"CNTDICT1"
//! [8..12]   version     u32 = 1
//! [12..16]  reserved    u32
//! [16..24]  entry_count u64
//! [24..32]  entries_offset        u64  ->  entry_count × Entry(16B)，按 key 字节序升序
//! [32..40]  cand_offset           u64  ->  N × CandidateHeader(12B)
//! [40..48]  strings_offset        u64  ->  UTF-8 字节（key 与 word 共用）
//! [48..56]  strings_len           u64
//! [56..64]  reserved    u64
//!
//! Entry（16 字节）：
//!   [0..4]   key_off  u32   strings 内的偏移
//!   [4..6]   key_len  u16
//!   [6..8]   pad      u16
//!   [8..12]  cand_off u32   CandidateHeader 表内的下标（第几个）
//!   [12..16] cand_count u32
//!
//! CandidateHeader（12 字节）：
//!   [0..4]   word_off u32   strings 内的偏移
//!   [4..6]   word_len u16
//!   [6..8]   pad      u16
//!   [8..12]  freq     u32
//! ```

use cnt_store::{StoreError, read_u16, read_u32, read_u64};

pub const MAGIC: &[u8; 8] = b"CNTDICT1";
pub const FORMAT_VERSION: u32 = 1;

pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 16;
pub const CAND_HEADER_SIZE: usize = 12;

// ---- 格式上限（防恶意/损坏文件导致算术溢出）----
// 参考：真实输入法词库规模，常见词 40~60 万条、全量（含专业词库）100~200 万条、
// 二进制体积几十 MB。这里放宽到现实规模的 10 倍以上余量。

/// 拼音键（词条）总数上限：1000 万。
/// 先检查此项，可保证后续 usize 运算（`entry_count * ENTRY_SIZE`、候选数累加）不溢出。
pub const MAX_ENTRIES: usize = 10_000_000;
/// 候选词条总数上限：5000 万。
pub const MAX_CANDIDATES: usize = 50_000_000;
/// 字符串池（key + word 的 UTF-8 字节）上限：512 MB。
pub const MAX_STRINGS_LEN: usize = 512 * 1024 * 1024;

/// 文件头（解析后）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub entry_count: u64,
    pub entries_offset: u64,
    pub cand_offset: u64,
    pub strings_offset: u64,
    pub strings_len: u64,
}

impl Header {
    /// 解析文件头。
    ///
    /// # Errors
    /// 魔数不对、版本不支持或数据不足时返回 [`StoreError`]。
    pub fn parse(bytes: &[u8]) -> Result<Self, StoreError> {
        if bytes.len() < HEADER_SIZE {
            return Err(StoreError::Truncated("header"));
        }
        if &bytes[0..8] != MAGIC {
            return Err(StoreError::BadMagic);
        }
        let version = read_u32(&bytes[8..12]);
        if version != FORMAT_VERSION {
            return Err(StoreError::UnsupportedVersion(version));
        }
        Ok(Self {
            entry_count: read_u64(&bytes[16..24]),
            entries_offset: read_u64(&bytes[24..32]),
            cand_offset: read_u64(&bytes[32..40]),
            strings_offset: read_u64(&bytes[40..48]),
            strings_len: read_u64(&bytes[48..56]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.entry_count.to_le_bytes());
        out.extend_from_slice(&self.entries_offset.to_le_bytes());
        out.extend_from_slice(&self.cand_offset.to_le_bytes());
        out.extend_from_slice(&self.strings_offset.to_le_bytes());
        out.extend_from_slice(&self.strings_len.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
    }
}

/// 词条（见文件头注释）。
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub key_off: u32,
    pub key_len: u16,
    pub cand_off: u32,
    pub cand_count: u32,
}

impl Entry {
    /// 解析一个词条。
    ///
    /// # Errors
    /// 字节不足时返回 [`StoreError::Truncated`]。
    pub fn read(b: &[u8]) -> Result<Self, StoreError> {
        if b.len() < ENTRY_SIZE {
            return Err(StoreError::Truncated("entry"));
        }
        Ok(Self {
            key_off: read_u32(&b[0..4]),
            key_len: read_u16(&b[4..6]),
            cand_off: read_u32(&b[8..12]),
            cand_count: read_u32(&b[12..16]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.key_off.to_le_bytes());
        out.extend_from_slice(&self.key_len.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.cand_off.to_le_bytes());
        out.extend_from_slice(&self.cand_count.to_le_bytes());
    }
}

/// 候选头（见文件头注释）。
#[derive(Debug, Clone, Copy)]
pub struct CandidateHeader {
    pub word_off: u32,
    pub word_len: u16,
    pub freq: u32,
}

impl CandidateHeader {
    /// 解析一个候选头。
    ///
    /// # Errors
    /// 字节不足时返回 [`StoreError::Truncated`]。
    pub fn read(b: &[u8]) -> Result<Self, StoreError> {
        if b.len() < CAND_HEADER_SIZE {
            return Err(StoreError::Truncated("candidate header"));
        }
        Ok(Self {
            word_off: read_u32(&b[0..4]),
            word_len: read_u16(&b[4..6]),
            freq: read_u32(&b[8..12]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.word_off.to_le_bytes());
        out.extend_from_slice(&self.word_len.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.freq.to_le_bytes());
    }
}
