//! 词库生成器：把 `(拼音, 词, 频率)` 三元组编译成 `.cntd` 二进制文件。
//!
//! 供 `cnt-dict-tools` CLI 使用，也供测试构造小词库。

use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::format::{
    CAND_HEADER_SIZE, CandidateHeader, ENTRY_SIZE, Entry, HEADER_SIZE, Header, MAX_CANDIDATES,
    MAX_ENTRIES, MAX_STRINGS_LEN,
};

/// 把 `usize` 安全转成格式允许的窄类型；超限报 IO 错误。
fn narrow<T>(v: usize, what: &str) -> io::Result<T>
where
    T: TryFrom<usize>,
{
    T::try_from(v).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{what} too large for the .cntd format ({v} bytes)"),
        )
    })
}

/// 编译词库。`pairs` 可含重复的 (pinyin, word)，会自动合并、去重、取最高频率。
///
/// # Errors
/// 字符串池/词条数超出格式上限（`u32` 偏移 / `u16` 长度 / `MAX_*` 总量上限）时返回 IO 错误。
pub fn build(pairs: &[(String, String, u32)]) -> io::Result<Vec<u8>> {
    let entries: Vec<_> = pairs
        .iter()
        .map(|(k, w, f)| (k.clone(), w.clone(), *f, None))
        .collect();
    build_with_readings(&entries)
}

/// 编译带独立 log10 读音权重的词表；None 保持旧词频语义。
///
/// # Errors
/// 权重非有限值或条目超出格式上限时返回 IO 错误。
#[allow(clippy::too_many_lines)] // 顺序写入四个磁盘区域，保持布局计算相邻。
pub fn build_with_readings(pairs: &[(String, String, u32, Option<f32>)]) -> io::Result<Vec<u8>> {
    // 1. 按拼音分组（BTreeMap 保证 key 有序），候选按频率降序、词升序
    type WeightedWord<'a> = (&'a str, u32, Option<f32>);
    let mut table: BTreeMap<&str, Vec<WeightedWord<'_>>> = BTreeMap::new();
    for (pinyin, word, freq, weight) in pairs {
        if weight.is_some_and(|w| !w.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "non-finite reading weight",
            ));
        }
        let list = table.entry(pinyin.as_str()).or_default();
        match list.iter_mut().find(|(w, _, _)| *w == word.as_str()) {
            Some((_, f, old_weight)) => {
                *f = (*f).max(*freq);
                // 旧三列领域词可补词频，但不能擦掉原始词库的读音信息。
                if let Some(weight) = weight {
                    *old_weight = Some(old_weight.map_or(*weight, |old| old.max(*weight)));
                }
            }
            None => list.push((word.as_str(), *freq, *weight)),
        }
    }
    for list in table.values_mut() {
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    }

    // 2. 预计算各部分大小，以便在 Header 里写偏移
    let entry_count = table.len();
    let cand_count: usize = table.values().map(Vec::len).sum();
    if entry_count > MAX_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{entry_count} entries exceed format limit {MAX_ENTRIES}"),
        ));
    }
    if cand_count > MAX_CANDIDATES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{cand_count} candidates exceed format limit {MAX_CANDIDATES}"),
        ));
    }
    let cand_count_u64 = u64::try_from(cand_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many candidates"))?;
    let entries_offset = u64::try_from(HEADER_SIZE)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header too large"))?;
    let entry_count_u64 = u64::try_from(entry_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "too many entries"))?;
    let cand_offset = entries_offset + entry_count_u64 * ENTRY_SIZE as u64;
    let strings_offset = cand_offset + cand_count_u64 * CAND_HEADER_SIZE as u64;

    // 3. 字符串池（key 与 word 共用），记录每个串的偏移
    let mut strings: Vec<u8> = Vec::new();
    let str_off = |s: &str, strings: &mut Vec<u8>| -> io::Result<u32> {
        let off = narrow(strings.len(), "strings pool")?;
        strings.extend_from_slice(s.as_bytes());
        Ok(off)
    };

    let mut entries: Vec<u8> = Vec::with_capacity(entry_count * ENTRY_SIZE);
    let mut cands: Vec<u8> = Vec::new();
    let mut cand_index = 0u32;

    for (key, list) in &table {
        let key_off = str_off(key, &mut strings)?;
        let key_len = narrow(key.len(), "pinyin key")?;
        let e = Entry {
            key_off,
            key_len,
            cand_off: cand_index,
            cand_count: narrow(list.len(), "candidate count")?,
        };
        e.write(&mut entries);
        for (word, freq, reading_weight) in list {
            let word_off = str_off(word, &mut strings)?;
            let ch = CandidateHeader {
                reading_weight: *reading_weight,
                word_off,
                word_len: narrow(word.len(), "word")?,
                freq: *freq,
            };
            ch.write(&mut cands);
        }
        cand_index = cand_index
            .checked_add(narrow(list.len(), "candidate count")?)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "candidate index overflow")
            })?;
    }

    // 4. 组装：header + entries + cands + strings
    let strings_len = strings.len();
    if strings_len > MAX_STRINGS_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("strings pool {strings_len} bytes exceed format limit {MAX_STRINGS_LEN}"),
        ));
    }
    let header = Header {
        version: crate::format::FORMAT_VERSION,
        entry_count: entry_count_u64,
        entries_offset,
        cand_offset,
        strings_offset,
        strings_len: narrow(strings_len, "strings pool")?,
    };
    let mut out: Vec<u8> =
        Vec::with_capacity(usize::try_from(strings_offset).unwrap_or(0) + strings.len());
    header.write(&mut out);
    out.extend_from_slice(&entries);
    out.extend_from_slice(&cands);
    out.extend_from_slice(&strings);

    let mut w = io::Cursor::new(out);
    w.flush()?;
    Ok(w.into_inner())
}

/// 直接写入文件（供 CLI 使用）。
///
/// # Errors
/// 见 [`build`]；写文件失败时返回 IO 错误。
pub fn write_to_file(pairs: &[(String, String, u32)], path: &std::path::Path) -> io::Result<()> {
    let bytes = build(pairs)?;
    std::fs::write(path, bytes)
}
