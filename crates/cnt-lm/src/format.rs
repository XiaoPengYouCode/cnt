//! `.cntl` 语言模型二进制布局。
//!
//! 面向 mmap 的零拷贝设计：定长小端整数 + 字节偏移，读取时直接切片。
//!
//! 布局（所有整数小端，f32 用 `from_le_bytes` 安全非对齐读取）：
//! ```text
//! [0..8]    magic       b"CNTLM1\0\0"
//! [8..12]   version     u32 = 1
//! [12..16]  reserved    u32
//! [16..24]  word_count  u64
//! [24..32]  unigram_off u64
//! [32..40]  bigram_off  u64
//! [40..48]  bigram_count u64
//! [48..56]  strings_off u64
//! [56..64]  strings_len u64
//!
//! WordEntry（16 字节，按 word 字节序升序）：
//!   [0..4]   str_off  u32   strings 内偏移
//!   [4..6]   str_len  u16
//!   [6..8]   pad      u16
//!   [8..12]  logprob  f32   log10 概率（ARPA 直接搬过来）
//!   [12..16] backoff  f32   Katz backoff 权重（trigram 扩展用）
//!
//! BigramEntry（12 字节，按 (w1_idx, w2_idx) 升序）：
//!   [0..4]   w1_idx  u32   词表下标
//!   [4..8]   w2_idx  u32
//!   [8..12]  logprob f32
//! ```

use cnt_store::{StoreError, read_f32, read_u16, read_u32, read_u64};

pub const MAGIC: &[u8; 8] = b"CNTLM1\0\0";
pub const FORMAT_VERSION: u32 = 1;

pub const HEADER_SIZE: usize = 64;
pub const WORD_ENTRY_SIZE: usize = 16;
pub const BIGRAM_ENTRY_SIZE: usize = 12;

// ---- 格式上限（防恶意/损坏文件导致算术溢出）----
// 参考：fcitx5 libime 的 LM 约 27 万 unigram + 479 万 bigram。

/// 词表（unigram）总数上限：1000 万。
pub const MAX_WORDS: usize = 10_000_000;
/// bigram 总数上限：1 亿（真实 479 万，20 倍余量）。
pub const MAX_BIGRAMS: usize = 100_000_000;
/// 字符串池上限：1 GB。
pub const MAX_STRINGS_LEN: usize = 1024 * 1024 * 1024;

/// 文件头（解析后）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub word_count: u64,
    pub unigram_off: u64,
    pub bigram_off: u64,
    pub bigram_count: u64,
    pub strings_off: u64,
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
            word_count: read_u64(&bytes[16..24]),
            unigram_off: read_u64(&bytes[24..32]),
            bigram_off: read_u64(&bytes[32..40]),
            bigram_count: read_u64(&bytes[40..48]),
            strings_off: read_u64(&bytes[48..56]),
            strings_len: read_u64(&bytes[56..64]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.word_count.to_le_bytes());
        out.extend_from_slice(&self.unigram_off.to_le_bytes());
        out.extend_from_slice(&self.bigram_off.to_le_bytes());
        out.extend_from_slice(&self.bigram_count.to_le_bytes());
        out.extend_from_slice(&self.strings_off.to_le_bytes());
        out.extend_from_slice(&self.strings_len.to_le_bytes());
    }
}

/// 词条（见文件头注释）。
#[derive(Debug, Clone, Copy)]
pub struct WordEntry {
    pub str_off: u32,
    pub str_len: u16,
    pub logprob: f32,
    pub backoff: f32,
}

impl WordEntry {
    /// 解析一个词条。
    ///
    /// # Errors
    /// 字节不足时返回 [`StoreError::Truncated`]。
    pub fn read(b: &[u8]) -> Result<Self, StoreError> {
        if b.len() < WORD_ENTRY_SIZE {
            return Err(StoreError::Truncated("word entry"));
        }
        Ok(Self {
            str_off: read_u32(&b[0..4]),
            str_len: read_u16(&b[4..6]),
            logprob: read_f32(&b[8..12]),
            backoff: read_f32(&b[12..16]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.str_off.to_le_bytes());
        out.extend_from_slice(&self.str_len.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.logprob.to_le_bytes());
        out.extend_from_slice(&self.backoff.to_le_bytes());
    }
}

/// bigram 条目（见文件头注释）。
#[derive(Debug, Clone, Copy)]
pub struct BigramEntry {
    pub w1: u32,
    pub w2: u32,
    pub logprob: f32,
}

impl BigramEntry {
    /// 解析一个 bigram 条目。
    ///
    /// # Errors
    /// 字节不足时返回 [`StoreError::Truncated`]。
    pub fn read(b: &[u8]) -> Result<Self, StoreError> {
        if b.len() < BIGRAM_ENTRY_SIZE {
            return Err(StoreError::Truncated("bigram entry"));
        }
        Ok(Self {
            w1: read_u32(&b[0..4]),
            w2: read_u32(&b[4..8]),
            logprob: read_f32(&b[8..12]),
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.w1.to_le_bytes());
        out.extend_from_slice(&self.w2.to_le_bytes());
        out.extend_from_slice(&self.logprob.to_le_bytes());
    }
}
