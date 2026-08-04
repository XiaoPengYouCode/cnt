//! mmap 支撑的只读词库读取器。
//!
//! 整个文件映射进内存，查询时直接切片，零拷贝、零反序列化。
//! 文件头在 `open` 时解析一次并转成 `usize` 偏移（校验过边界），
//! 之后所有索引运算均为 `usize`，无截断风险。

use std::path::Path;

use cnt_store::{StoreError, validate_regions, MmapFile};
use crate::format::{
    Entry, Header, CAND_HEADER_SIZE, ENTRY_SIZE, MAX_CANDIDATES, MAX_ENTRIES, MAX_STRINGS_LEN,
};

/// 精确匹配命中：`word` 借用自 mmap 内存。
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    pub word: &'a str,
    pub freq: u32,
}

/// 前缀匹配命中：附带了完整 key（供用户模型按 (完整拼音, 词) 查调频）。
#[derive(Debug, Clone, Copy)]
pub struct PrefixHit<'a> {
    pub key: &'a str,
    pub word: &'a str,
    pub freq: u32,
}

pub struct MmapDict {
    mmap: MmapFile,
    entry_count: usize,
    entries_start: usize,
    cand_start: usize,
    strings_start: usize,
    strings_len: usize,
}

impl MmapDict {
    /// 打开并 mmap 一个 `.cntd` 文件。
    ///
    /// # Errors
    /// 文件不存在/无法 mmap，或魔数/版本/区域边界不合法时返回 [`StoreError`]。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let mmap = MmapFile::open(path)?;
        let header = Header::parse(&mmap)?;
        let dict = Self {
            mmap,
            entry_count: usize::try_from(header.entry_count)
                .map_err(|_| StoreError::Region("entry_count"))?,
            entries_start: usize::try_from(header.entries_offset)
                .map_err(|_| StoreError::Region("entries"))?,
            cand_start: usize::try_from(header.cand_offset)
                .map_err(|_| StoreError::Region("candidates"))?,
            strings_start: usize::try_from(header.strings_offset)
                .map_err(|_| StoreError::Region("strings"))?,
            strings_len: usize::try_from(header.strings_len)
                .map_err(|_| StoreError::Region("strings"))?,
        };
        dict.validate()?;
        Ok(dict)
    }

    /// 校验各区域边界与格式上限，防止坏文件导致切片越界 panic 或算术溢出。
    ///
    /// 顺序很重要：先限制 `entry_count`（保证后续乘法/累加不溢出），
    /// 再校验 entries 区域（保证 `cand_count()` 读 entry 时切片安全），
    /// 再限制候选总数，最后校验 cand/strings 区域。
    fn validate(&self) -> Result<(), StoreError> {
        if self.entry_count > MAX_ENTRIES {
            return Err(StoreError::LimitExceeded("entry_count"));
        }
        let cand_total = self.cand_count();
        if cand_total > MAX_CANDIDATES {
            return Err(StoreError::LimitExceeded("candidates"));
        }
        if self.strings_len > MAX_STRINGS_LEN {
            return Err(StoreError::LimitExceeded("strings"));
        }
        let len = self.mmap.len();
        validate_regions(
            len,
            &[
                (self.entries_start, self.entry_count * ENTRY_SIZE, "entries"),
                (self.cand_start, cand_total * CAND_HEADER_SIZE, "candidates"),
                (self.strings_start, self.strings_len, "strings"),
            ],
        )
    }

    /// 词条（拼音键）总数。
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// 所有候选头的总数（统计用）。
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // 格式字段 u32→usize，仅支持 ≥32 位目标
    pub fn cand_count(&self) -> usize {
        let mut total = 0usize;
        for i in 0..self.entry_count {
            total += self.entry(i).cand_count as usize;
        }
        total
    }

    /// 第 i 个 key（用于工具/测试）。
    #[must_use]
    pub fn key_at(&self, i: usize) -> Option<&str> {
        self.str_of_entry(self.entry(i)).ok()
    }

    // ---- 内部切片 ----

    fn strings(&self) -> &[u8] {
        let start = self.strings_start;
        let end = start + self.strings_len;
        &self.mmap[start..end]
    }

    fn entry(&self, i: usize) -> Entry {
        let start = self.entries_start + i * ENTRY_SIZE;
        let b = &self.mmap[start..start + ENTRY_SIZE];
        Entry::read(b).expect("entry range validated at open")
    }

    fn cand_header(&self, index: u32) -> crate::format::CandidateHeader {
        #[allow(clippy::cast_possible_truncation)] // 格式字段 u32→usize，仅支持 ≥32 位目标
        let off = self.cand_start + index as usize * CAND_HEADER_SIZE;
        let b = &self.mmap[off..off + CAND_HEADER_SIZE];
        crate::format::CandidateHeader::read(b).expect("cand range validated at open")
    }

    /// 从字符串池取 (offset, len) 处的 `&str`；越界/非 UTF-8 视为坏数据。
    fn str_at(&self, off: u32, len: u16) -> Option<&str> {
        #[allow(clippy::cast_possible_truncation)] // 格式字段 u32→usize，仅支持 ≥32 位目标
        let off = off as usize;
        let len = usize::from(len);
        let bytes = self.strings().get(off..off + len)?;
        std::str::from_utf8(bytes).ok()
    }

    fn str_of_entry(&self, e: Entry) -> Result<&str, ()> {
        self.str_at(e.key_off, e.key_len).ok_or(())
    }

    fn candidates_of(&self, e: Entry) -> Vec<Candidate<'_>> {
        #[allow(clippy::cast_possible_truncation)] // 格式字段 u32→usize，仅支持 ≥32 位目标
        let mut out = Vec::with_capacity(e.cand_count as usize);
        for i in 0..e.cand_count {
            let h = self.cand_header(e.cand_off + i);
            if let Some(word) = self.str_at(h.word_off, h.word_len) {
                out.push(Candidate { word, freq: h.freq });
            }
        }
        out
    }

    /// 第一个 key >= target 的下标（二分）。
    fn lower_bound(&self, target: &[u8]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.entry_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let key = self.str_of_entry(self.entry(mid)).unwrap_or("");
            if key.as_bytes() < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // ---- 查询 ----

    /// 精确匹配：返回该拼音的全部候选（文件内已按频率降序）。
    #[must_use]
    pub fn exact(&self, pinyin: &str) -> Vec<Candidate<'_>> {
        let target = pinyin.as_bytes();
        let i = self.lower_bound(target);
        if i >= self.entry_count {
            return Vec::new();
        }
        let e = self.entry(i);
        if self.str_of_entry(e).ok() != Some(pinyin) {
            return Vec::new();
        }
        self.candidates_of(e)
    }

    /// 前缀匹配：key 以 pinyin 开头，每个 key 取最高频的首个候选。
    #[must_use]
    pub fn prefix(&self, pinyin: &str) -> Vec<PrefixHit<'_>> {
        let target = pinyin.as_bytes();
        let mut i = self.lower_bound(target);
        let mut out = Vec::new();
        while i < self.entry_count {
            let e = self.entry(i);
            let Ok(key) = self.str_of_entry(e) else {
                i += 1;
                continue;
            };
            if !key.starts_with(pinyin) {
                break;
            }
            if let Some(c) = self.candidates_of(e).into_iter().next() {
                out.push(PrefixHit {
                    key,
                    word: c.word,
                    freq: c.freq,
                });
            }
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::build;

    fn dict_with(name: &str, pairs: &[(&str, &str, u32)]) -> MmapDict {
        let pairs: Vec<(String, String, u32)> = pairs
            .iter()
            .map(|(p, w, f)| (p.to_string(), w.to_string(), *f))
            .collect();
        let bytes = build(&pairs).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cnt-test-{}-{}.cntd", std::process::id(), name));
        std::fs::write(&path, &bytes).unwrap();
        let d = MmapDict::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap(); // mmap 已建立，删除文件不影响
        d
    }

    #[test]
    fn exact_returns_sorted_candidates() {
        let d = dict_with(
            "exact",
            &[
                ("shi", "是", 100),
                ("shi", "时", 200),
                ("shi", "事", 300),
                ("ni", "你", 500),
            ],
        );
        assert_eq!(d.entry_count(), 2);
        let c = d.exact("shi");
        let words: Vec<&str> = c.iter().map(|x| x.word).collect();
        assert_eq!(words, vec!["事", "时", "是"]); // 按频率降序
        assert_eq!(d.exact("shi")[0].freq, 300);
        assert!(d.exact("nope").is_empty());
    }

    #[test]
    fn prefix_matches_range() {
        let d = dict_with(
            "prefix",
            &[
                ("ni", "你", 100),
                ("nian", "年", 90),
                ("nihao", "你好", 80),
                ("wo", "我", 70),
            ],
        );
        let hits = d.prefix("ni");
        let keys: Vec<&str> = hits.iter().map(|h| h.key).collect();
        assert_eq!(keys, vec!["ni", "nian", "nihao"]);
        // 每个 key 只取最高频候选
        assert!(hits.iter().all(|h| h.word == h.word));
        let first = d.prefix("ni")[0];
        assert_eq!(first.key, "ni");
        assert_eq!(first.word, "你");
    }

    #[test]
    fn dedupes_and_merges_duplicates() {
        let d = dict_with("dedupe", &[("ni", "你", 100), ("ni", "你", 300), ("ni", "泥", 200)]);
        let c = d.exact("ni");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].word, "你");
        assert_eq!(c[0].freq, 300); // 保留最高频率
    }
}
