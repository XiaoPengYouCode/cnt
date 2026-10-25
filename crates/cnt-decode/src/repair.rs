//! 有界的整词读音修复：一次模糊音，或一次键盘编辑，不叠加两类猜测。
//!
//! 邻键范围参考 libime 的 PinyinCorrectionProfile（QWERTY 同行相邻键）；
//! 与 librime 一样将纠错和精确读音分开，候选数量由调用方限制。
//! 此处只生成键，调用方必须用词库验证；不做任意编辑距离的全词库扫描。

use crate::syllable::SyllableTable;

const MAX_REPAIR_BYTES: usize = 48;

pub fn eligible(input: &str) -> bool {
    (4..=MAX_REPAIR_BYTES).contains(&input.len()) && input.bytes().all(|b| b.is_ascii_lowercase())
}

/// 每个结果恰好包含一次邻键替换、不同字母交换或重复字母删除。
/// 不跨显式音节分隔符，不修改短声母前缀，不同时修复两个错误。
pub fn typo_keys(input: &str) -> Vec<String> {
    if !eligible(input) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut bytes = input.as_bytes().to_vec();
    for i in 0..bytes.len() {
        let original = bytes[i];
        for row in [b"qwertyuiop".as_slice(), b"asdfghjkl", b"zxcvbnm"] {
            if let Some(pos) = row.iter().position(|&b| b == original) {
                for neighbor in [pos.checked_sub(1), (pos + 1 < row.len()).then_some(pos + 1)]
                    .into_iter()
                    .flatten()
                {
                    bytes[i] = row[neighbor];
                    out.push(String::from_utf8(bytes.clone()).expect("ASCII keyboard row"));
                }
                bytes[i] = original;
                break;
            }
        }
        if i + 1 < bytes.len() {
            if bytes[i] == bytes[i + 1] {
                let mut key = input.to_string();
                key.remove(i);
                out.push(key);
            } else {
                bytes.swap(i, i + 1);
                out.push(String::from_utf8(bytes.clone()).expect("ASCII input"));
                bytes.swap(i, i + 1);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 仅替换一个音节的模糊读音；前后必须能按精确音节切分。
/// 因此 an/ang 等规则不会跨音节边界改写另一个声母。
pub fn fuzzy_keys(input: &str, table: &SyllableTable) -> Vec<(String, u8)> {
    if !eligible(input) {
        return Vec::new();
    }
    let lattice = table.lattice(input);
    let n = input.len();
    let mut prefix = vec![false; n + 1];
    let mut suffix = vec![false; n + 1];
    prefix[0] = true;
    suffix[n] = true;
    for pos in 0..n {
        if prefix[pos] {
            for edge in &lattice[pos] {
                prefix[edge.end] = true;
            }
        }
    }
    for pos in (0..n).rev() {
        suffix[pos] = lattice[pos].iter().any(|edge| suffix[edge.end]);
    }
    let mut out = Vec::new();
    for pos in 0..n {
        if !prefix[pos] {
            continue;
        }
        for edge in table.fuzzy_syllables_at(input, pos) {
            if edge.is_fuzzy() && suffix[edge.end] {
                let key = format!("{}{}{}", &input[..pos], edge.syl, &input[edge.end..]);
                out.push((key, edge.cost));
            }
        }
    }
    out.sort_unstable();
    out.dedup_by(|a, b| a.0 == b.0); // 相同键按最低代价保留。
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_keyboard_error_only() {
        for input in ["nihso", "nihoa", "niihao", "nnihao", "nihaoo"] {
            assert!(typo_keys(input).iter().any(|key| key == "nihao"), "{input}");
        }
        assert!(!typo_keys("nihxo").iter().any(|key| key == "nihao"));
        assert!(!typo_keys("niihso").iter().any(|key| key == "nihao"));
        for input in ["", "nh", "ni", "NIHAO", "你好", "ni'hao", &"a".repeat(49)] {
            assert!(typo_keys(input).is_empty(), "{input}");
        }
    }

    #[test]
    fn fuzzy_repair_uses_one_syllable_and_keeps_exact_surroundings() {
        let table = SyllableTable::new();
        let keys = fuzzy_keys("sihou", &table);
        assert!(keys.contains(&("shihou".to_string(), 1)));
        assert!(!keys.iter().any(|(key, _)| key == "shifou"));
        assert_eq!(fuzzy_keys("xi'an", &table), [] as [(String, u8); 0]);
    }
}
