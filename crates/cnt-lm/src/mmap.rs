//! mmap 支撑的只读语言模型读取器。
//!
//! 文件头在 `open` 时解析一次并转成 `usize` 偏移（校验过边界与上限），
//! 查询为词表/bigram 表上的二分查找，零拷贝。

use std::path::Path;

use crate::format::{
    BIGRAM_ENTRY_SIZE, BigramEntry, MAX_BIGRAMS, MAX_STRINGS_LEN, MAX_WORDS, WORD_ENTRY_SIZE,
    WordEntry,
};
use cnt_store::{MmapFile, StoreError, validate_regions};

pub struct CntLm {
    mmap: MmapFile,
    word_count: usize,
    unigram_off: usize,
    bigram_off: usize,
    bigram_count: usize,
    strings_off: usize,
    strings_len: usize,
}

impl CntLm {
    /// 打开并 mmap 一个 `.cntl` 文件。
    ///
    /// # Errors
    /// 文件不存在/无法 mmap，或魔数/版本/区域边界/上限不合法时返回 [`StoreError`]。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let mmap = MmapFile::open(path)?;
        let header = crate::format::Header::parse(&mmap)?;
        let lm = Self {
            mmap,
            word_count: usize::try_from(header.word_count)
                .map_err(|_| StoreError::Region("word_count"))?,
            unigram_off: usize::try_from(header.unigram_off)
                .map_err(|_| StoreError::Region("unigrams"))?,
            bigram_off: usize::try_from(header.bigram_off)
                .map_err(|_| StoreError::Region("bigrams"))?,
            bigram_count: usize::try_from(header.bigram_count)
                .map_err(|_| StoreError::Region("bigram_count"))?,
            strings_off: usize::try_from(header.strings_off)
                .map_err(|_| StoreError::Region("strings"))?,
            strings_len: usize::try_from(header.strings_len)
                .map_err(|_| StoreError::Region("strings"))?,
        };
        lm.validate()?;
        Ok(lm)
    }

    /// 校验各区域边界与格式上限，防止坏文件导致切片越界 panic 或算术溢出。
    ///
    /// 顺序：先限制 `word_count` / `bigram_count`（保证乘法/累加不溢出），
    /// 再校验各区域边界。
    fn validate(&self) -> Result<(), StoreError> {
        if self.word_count > MAX_WORDS {
            return Err(StoreError::LimitExceeded("word_count"));
        }
        if self.bigram_count > MAX_BIGRAMS {
            return Err(StoreError::LimitExceeded("bigram_count"));
        }
        if self.strings_len > MAX_STRINGS_LEN {
            return Err(StoreError::LimitExceeded("strings"));
        }
        let len = self.mmap.len();
        validate_regions(
            len,
            &[
                (
                    self.unigram_off,
                    self.word_count * WORD_ENTRY_SIZE,
                    "unigrams",
                ),
                (
                    self.bigram_off,
                    self.bigram_count * BIGRAM_ENTRY_SIZE,
                    "bigrams",
                ),
                (self.strings_off, self.strings_len, "strings"),
            ],
        )
    }

    /// 词表（unigram）总数。
    #[must_use]
    pub const fn word_count(&self) -> usize {
        self.word_count
    }

    /// bigram 总数。
    #[must_use]
    pub const fn bigram_count(&self) -> usize {
        self.bigram_count
    }

    /// 遍历所有 unigram：`(词, log10 概率, backoff)`。
    pub fn unigrams(&self) -> impl Iterator<Item = (&str, f32, f32)> {
        (0..self.word_count).map(|i| {
            let w = self.word_entry(i);
            let word = self.str_at(w.str_off, w.str_len).unwrap_or("");
            (word, w.logprob, w.backoff)
        })
    }

    // ---- 内部切片 ----

    fn strings(&self) -> &[u8] {
        &self.mmap[self.strings_off..self.strings_off + self.strings_len]
    }

    fn word_entry(&self, i: usize) -> WordEntry {
        let off = self.unigram_off + i * WORD_ENTRY_SIZE;
        let b = &self.mmap[off..off + WORD_ENTRY_SIZE];
        WordEntry::read(b).expect("word range validated at open")
    }

    fn str_at(&self, off: u32, len: u16) -> Option<&str> {
        #[allow(clippy::cast_possible_truncation)] // 格式字段 u32→usize，仅支持 ≥32 位目标
        let off = off as usize;
        let len = usize::from(len);
        let bytes = self.strings().get(off..off + len)?;
        std::str::from_utf8(bytes).ok()
    }

    /// 二分：第一个 word >= target 的下标。
    fn lower_bound(&self, target: &str) -> usize {
        let mut lo = 0usize;
        let mut hi = self.word_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let w = self.word_entry(mid);
            let key = self.str_at(w.str_off, w.str_len).unwrap_or("");
            if key < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // ---- 查询 ----

    /// 词在词表中的下标（二分）。
    ///
    /// # Panics
    /// 仅当词表大小超过 `u32` 上限时（已被 `MAX_WORDS` 封顶，不可能）。
    #[must_use]
    pub fn word_index(&self, word: &str) -> Option<u32> {
        let i = self.lower_bound(word);
        if i >= self.word_count {
            return None;
        }
        let w = self.word_entry(i);
        if self.str_at(w.str_off, w.str_len)? != word {
            return None;
        }
        Some(u32::try_from(i).expect("word_count <= MAX_WORDS"))
    }

    /// unigram 概率：返回 `(log10 概率, backoff)`。
    ///
    /// # Panics
    /// 仅当词表大小超过 `u32` 上限时（已被 `MAX_WORDS` 封顶，不可能）。
    #[must_use]
    pub fn unigram(&self, word: &str) -> Option<(f32, f32)> {
        let i = self.word_index(word)?;
        let w = self.word_entry(usize::try_from(i).expect("word_count <= MAX_WORDS"));
        Some((w.logprob, w.backoff))
    }

    /// 按词表下标取 unigram 概率 `(log10 概率, backoff)`。
    ///
    /// # Panics
    /// 下标超过词表大小（仅当传入非法下标时）。
    #[must_use]
    pub fn unigram_by_idx(&self, idx: u32) -> Option<(f32, f32)> {
        let i = usize::try_from(idx).expect("idx fits usize");
        let w = self.word_entry(i);
        Some((w.logprob, w.backoff))
    }

    /// 按下标取 bigram 条件概率 `log P(w2 | w1)`（二分，避免字符串查找）。
    ///
    /// # Panics
    /// 仅当 mmap 内部损坏导致大下标越界切片时（打开时已校验范围，不可能）。
    #[must_use]
    pub fn bigram_by_idx(&self, w1: u32, w2: u32) -> Option<f32> {
        let i = self.bigram_lower_bound(w1, w2);
        if i >= self.bigram_count {
            return None;
        }
        let e = self.bigram_entry(i);
        ((e.w1, e.w2) == (w1, w2)).then_some(e.logprob)
    }

    /// 前词 `w1` 的 bigram 行：`(起始下标, 结束下标)`（左闭右开）。
    ///
    /// bigram 表按 `(w1, w2)` 排序，所以同一前词的后继是连续区间。beam 展开时
    /// 一条假设要对同一前词查几十个后继，先定位一次行、再在行内二分，能把
    /// 「几十次跨 60MB mmap 的随机二分」压成「一次定位 + 行内小范围二分」。
    #[must_use]
    pub fn bigram_row(&self, w1: u32) -> (u32, u32) {
        let lo = self.bigram_lower_bound(w1, 0);
        let hi = self.bigram_lower_bound(w1.saturating_add(1), 0);
        (
            u32::try_from(lo).unwrap_or(u32::MAX),
            u32::try_from(hi).unwrap_or(u32::MAX),
        )
    }

    /// 在 bigram 行内查 `log P(w2 | w1)`（行由 [`Self::bigram_row`] 给出）。
    #[must_use]
    pub fn bigram_in_row(&self, row: (u32, u32), w2: u32) -> Option<f32> {
        let (lo, hi) = (row.0 as usize, row.1 as usize);
        if lo >= hi || hi > self.bigram_count {
            return None;
        }
        let mut left = lo;
        let mut right = hi;
        while left < right {
            let mid = left + (right - left) / 2;
            let e = self.bigram_entry(mid);
            if e.w2 < w2 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left >= hi {
            return None;
        }
        let e = self.bigram_entry(left);
        (e.w2 == w2).then_some(e.logprob)
    }

    /// bigram 表内 `(w1, w2)` 的 lower bound 下标。
    fn bigram_lower_bound(&self, w1: u32, w2: u32) -> usize {
        let mut lo = 0usize;
        let mut hi = self.bigram_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let e = self.bigram_entry(mid);
            if (e.w1, e.w2) < (w1, w2) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// 读第 `i` 条 bigram（下标越界由调用方保证；打开时已校验区间范围）。
    ///
    /// # Panics
    /// 仅当 mmap 内部损坏导致越界切片时（打开时已校验范围，不可能）。
    fn bigram_entry(&self, i: usize) -> BigramEntry {
        let off = self.bigram_off + i * BIGRAM_ENTRY_SIZE;
        BigramEntry::read(&self.mmap[off..off + BIGRAM_ENTRY_SIZE])
            .expect("bigram range validated at open")
    }

    /// bigram 条件概率 `log P(w2 | w1)`。
    ///
    /// # Panics
    /// 仅当词表大小超过 `u32` 上限时（已被 `MAX_WORDS` 封顶，不可能）。
    #[must_use]
    pub fn bigram(&self, w1: &str, w2: &str) -> Option<f32> {
        let (Some(i1), Some(i2)) = (self.word_index(w1), self.word_index(w2)) else {
            return None;
        };
        self.bigram_by_idx(i1, i2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{Bigram, Unigram, build};

    fn lm_with(name: &str, unigrams: &[Unigram], bigrams: &[Bigram]) -> CntLm {
        let bytes = build(unigrams, bigrams).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cnt-lm-test-{}-{}.cntl", std::process::id(), name));
        std::fs::write(&path, &bytes).unwrap();
        let lm = CntLm::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap(); // mmap 已建立，删除文件不影响
        lm
    }

    fn u(word: &str, logprob: f32, backoff: f32) -> Unigram {
        Unigram {
            word: word.to_string(),
            logprob,
            backoff,
        }
    }

    fn b(w1: &str, w2: &str, logprob: f32) -> Bigram {
        Bigram {
            w1: w1.to_string(),
            w2: w2.to_string(),
            logprob,
        }
    }

    #[test]
    fn unigram_and_bigram_roundtrip() {
        let lm = lm_with(
            "basic",
            &[
                u("的", -1.35, -0.63),
                u("我们", -2.4, -0.3),
                u("中国", -2.84, -0.46),
                u("是", -1.94, -0.41),
            ],
            &[b("中国", "是", -0.5), b("我们", "是", -0.8)],
        );
        assert_eq!(lm.word_count(), 4);
        assert_eq!(lm.bigram_count(), 2);
        assert_eq!(lm.unigram("中国"), Some((-2.84, -0.46)));
        assert_eq!(lm.bigram("中国", "是"), Some(-0.5));
        assert_eq!(lm.bigram("我们", "是"), Some(-0.8));
        assert!(lm.unigram("不存在").is_none());
        assert!(lm.bigram("的", "是").is_none()); // 未收录的 bigram
    }

    #[test]
    fn bigram_ordering_independent_of_input_order() {
        let lm = lm_with(
            "order",
            &[u("a", -1.0, 0.0), u("b", -1.0, 0.0), u("c", -1.0, 0.0)],
            &[
                b("c", "a", -0.1), // 乱序输入
                b("a", "b", -0.2),
                b("b", "c", -0.3),
            ],
        );
        assert_eq!(lm.bigram("a", "b"), Some(-0.2));
        assert_eq!(lm.bigram("b", "c"), Some(-0.3));
        assert_eq!(lm.bigram("c", "a"), Some(-0.1));
    }
}
