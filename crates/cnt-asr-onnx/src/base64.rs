//! 极小的 base64 解码（sherpa 的字节级 BPE 词表用它编码 token）。
//!
//! 为什么不引 crate：只需要「解码一行短字符串」这一个功能，30 行写完，
//! 而且词表加载是一次性冷路径。与项目里自研 mmap/fbank 的取舍一致。

/// 解码标准 base64（要求正确的 `=` 补齐）；非法输入返回 None。
#[must_use]
pub fn decode(input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut acc: u32 = 0;
        let mut pad = 0;
        for (i, b) in chunk.iter().enumerate() {
            let v = match b {
                b'A'..=b'Z' => u32::from(b - b'A'),
                b'a'..=b'z' => u32::from(b - b'a') + 26,
                b'0'..=b'9' => u32::from(b - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                // `=` 只允许出现在末尾一两位
                b'=' if i >= 2 => {
                    pad += 1;
                    0
                }
                _ => return None,
            };
            acc = (acc << 6) | v;
        }
        let [_, a, b, c] = acc.to_be_bytes();
        out.push(a);
        if pad < 2 {
            out.push(b);
        }
        if pad < 1 {
            out.push(c);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn decodes_ascii() {
        assert_eq!(decode("IQ=="), Some(b"!".to_vec()));
        assert_eq!(decode("Ig=="), Some(b"\"".to_vec()));
        assert_eq!(decode("aGVsbG8="), Some(b"hello".to_vec()));
        assert_eq!(decode("aGVsbG8h"), Some(b"hello!".to_vec()));
    }

    #[test]
    fn decodes_utf8_bytes() {
        // 「你」= E4 BD A0
        assert_eq!(decode("5L2g"), Some(vec![0xe4, 0xbd, 0xa0]));
        // 字节级 BPE 的 token 可能是**半个**汉字，解码结果不必是合法 UTF-8
        assert_eq!(decode("5L0="), Some(vec![0xe4, 0xbd]));
    }

    #[test]
    fn rejects_non_base64() {
        assert_eq!(decode(""), None);
        assert_eq!(decode("s"), None); // 长度不是 4 的倍数
        assert_eq!(decode("the"), None);
        assert_eq!(decode("<unk>"), None);
        assert_eq!(decode("▁the"), None);
        assert_eq!(decode("ab!c"), None); // 非法字符
    }
}
