//! `.cntl` 生成器：把 unigram/bigram 三元组编译成二进制。
//!
//! 供 `cnt-dict-tools build-lm`（解析 ARPA 文本后调用）与测试使用。

use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::format::{
    BIGRAM_ENTRY_SIZE, BigramEntry, HEADER_SIZE, Header, MAX_BIGRAMS, MAX_STRINGS_LEN, MAX_WORDS,
    WORD_ENTRY_SIZE, WordEntry,
};

/// 一个 unigram 词条（logprob/backoff 均为 log10）。
#[derive(Debug, Clone)]
pub struct Unigram {
    pub word: String,
    pub logprob: f32,
    pub backoff: f32,
}

/// 一个 bigram 词条（logprob 为 log10 条件概率）。
#[derive(Debug, Clone)]
pub struct Bigram {
    pub w1: String,
    pub w2: String,
    pub logprob: f32,
}

/// 把 `usize` 安全转成格式允许的窄类型；超限报 IO 错误。
fn narrow<T>(v: usize, what: &str) -> io::Result<T>
where
    T: TryFrom<usize>,
{
    T::try_from(v).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{what} too large for the .cntl format ({v} bytes)"),
        )
    })
}

/// 编译语言模型。
///
/// - unigram 按 word 排序去重（保留第一个）；bigram 按词表下标排序
/// - bigram 引用了未收录的词时直接跳过（防御 ARPA 不规范）
///
/// # Errors
/// 词数/bigram 数/字符串池超出格式上限，或类型过窄时返回 IO 错误。
///
/// # Panics
/// 仅当词数超过 `u32` 上限时（已被 `MAX_WORDS` 封顶，不可能）。
pub fn build(unigrams: &[Unigram], bigrams: &[Bigram]) -> io::Result<Vec<u8>> {
    // 1. unigram 表：按 word 排序 + 去重 + 编下标
    let mut sorted: BTreeMap<&str, (f32, f32)> = BTreeMap::new();
    for u in unigrams {
        sorted
            .entry(u.word.as_str())
            .or_insert((u.logprob, u.backoff));
    }
    let word_count = sorted.len();
    if word_count > MAX_WORDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{word_count} words exceed format limit {MAX_WORDS}"),
        ));
    }

    // 2. bigram 表：word -> idx，按 (w1_idx, w2_idx) 排序
    let idx_of: BTreeMap<&str, u32> = sorted
        .keys()
        .enumerate()
        .map(|(i, w)| (*w, u32::try_from(i).expect("word_count <= MAX_WORDS")))
        .collect();
    let mut bigram_map: BTreeMap<(u32, u32), f32> = BTreeMap::new();
    for b in bigrams {
        if let (Some(&w1), Some(&w2)) = (idx_of.get(b.w1.as_str()), idx_of.get(b.w2.as_str())) {
            bigram_map.entry((w1, w2)).or_insert(b.logprob);
        }
    }
    let bigram_count = bigram_map.len();
    if bigram_count > MAX_BIGRAMS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{bigram_count} bigrams exceed format limit {MAX_BIGRAMS}"),
        ));
    }

    // 3. 偏移预算
    let unigram_off = HEADER_SIZE as u64;
    let bigram_off = unigram_off + word_count as u64 * WORD_ENTRY_SIZE as u64;
    let strings_off = bigram_off + bigram_count as u64 * BIGRAM_ENTRY_SIZE as u64;

    // 4. 字符串池 + 词表
    let mut strings: Vec<u8> = Vec::new();
    let str_off = |s: &str, strings: &mut Vec<u8>| -> io::Result<u32> {
        let off = narrow(strings.len(), "strings pool")?;
        strings.extend_from_slice(s.as_bytes());
        Ok(off)
    };
    let mut words: Vec<u8> = Vec::with_capacity(word_count * WORD_ENTRY_SIZE);
    for (word, (logprob, backoff)) in &sorted {
        let e = WordEntry {
            str_off: str_off(word, &mut strings)?,
            str_len: narrow(word.len(), "word")?,
            logprob: *logprob,
            backoff: *backoff,
        };
        e.write(&mut words);
    }
    if strings.len() > MAX_STRINGS_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("strings pool {} bytes exceed format limit", strings.len()),
        ));
    }

    // 5. bigram 表
    let mut bg: Vec<u8> = Vec::with_capacity(bigram_count * BIGRAM_ENTRY_SIZE);
    for ((w1, w2), logprob) in &bigram_map {
        BigramEntry {
            w1: *w1,
            w2: *w2,
            logprob: *logprob,
        }
        .write(&mut bg);
    }

    // 6. 组装
    let header = Header {
        word_count: u64::try_from(word_count)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "word_count overflow"))?,
        unigram_off,
        bigram_off,
        bigram_count: u64::try_from(bigram_count)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bigram_count overflow"))?,
        strings_off,
        strings_len: narrow(strings.len(), "strings pool")?,
    };
    let mut out = Vec::with_capacity(HEADER_SIZE + words.len() + bg.len() + strings.len());
    header.write(&mut out);
    out.extend_from_slice(&words);
    out.extend_from_slice(&bg);
    out.extend_from_slice(&strings);

    let mut w = io::Cursor::new(out);
    w.flush()?;
    Ok(w.into_inner())
}
