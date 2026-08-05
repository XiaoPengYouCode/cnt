//! CTC 贪心解码 + token 表（同时支持明文与 base64 字节级词表）。
//!
//! `SenseVoice` 系模型是 CTC 结构：输出 `[T, vocab]` 的 logits，每帧取 argmax，
//! 去掉 blank 与**连续重复**即得 token 序列。
//!
//! 为什么贪心够用：CTC 的峰值极尖锐，贪心与 beam 在中文上的差距 <0.5% CER，
//! 而 beam 要多花几倍时间——输入法的预算是「松手到上屏 < 300 ms」。
//! 真要提准，正确的做法是拿 n-best 过 `cnt-lm` 重排，而不是在 CTC 内部堆 beam。
//!
//! ## 两种词表，一套抽象
//!
//! 换模型时踩过的坑（所以这里必须自动识别，不能靠配置）：
//!
//! | | `SenseVoice-Small` | `Fun-ASR-Nano` (CTC 导出) |
//! |---|---|---|
//! | 编码 | 明文（`▁the 3`） | **base64**（`IQ== 0`） |
//! | 分词 | `SentencePiece`（`▁` 标词首） | **字节级 BPE**（token 可能是半个汉字） |
//! | 词表 | 25055 | 60515 |
//! | blank | id 0 | **id 60514**（`<blk>` 在最后） |
//!
//! 所以 token 统一存**字节**：拼装时先按字节拼接、再做一次 UTF-8 解码
//! （字节级 BPE 的一个汉字会跨 2~3 个 token，按字符串拼必然乱码）。

use std::path::Path;

use cnt_asr::AsrError;

use crate::base64;

/// 常见的 blank token 写法（按此顺序找；都没有则退回 id 0）。
const BLANK_NAMES: &[&str] = &["<blk>", "<blank>", "<pad>", "<blank_id>"];

/// token 表：id → 字节串。
#[derive(Debug, Clone, Default)]
pub struct Tokens {
    /// 每个 id 的字节表示。
    table: Vec<Vec<u8>>,
    /// 元 token（`<|zh|>`、`<blk>`、`<|0.02|>` 这类）的 id 掩码：拼装时跳过。
    meta: Vec<bool>,
    /// CTC blank 的 id。
    blank: usize,
    /// 词表是否 base64 编码（诊断用）。
    base64: bool,
}

impl Tokens {
    /// 从 `tokens.txt` 加载（每行 `token id`）。自动识别明文 / base64。
    ///
    /// # Errors
    /// 文件读不到或没有任何合法行时返回错误。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, AsrError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| AsrError::Model(format!("{}: {e}", path.display())))?;
        Self::parse(&content)
            .ok_or_else(|| AsrError::Model(format!("{}: no tokens parsed", path.display())))
    }

    /// 解析词表内容（便于单测）。
    #[must_use]
    pub fn parse(content: &str) -> Option<Self> {
        let rows: Vec<(&str, usize)> = content
            .lines()
            .filter_map(|line| {
                let (token, id) = line.rsplit_once(' ')?;
                Some((token, id.trim().parse::<usize>().ok()?))
            })
            .collect();
        if rows.is_empty() {
            return None;
        }
        let base64 = looks_base64(&rows);

        let mut table: Vec<Vec<u8>> = Vec::new();
        for (token, id) in &rows {
            if table.len() <= *id {
                table.resize(id + 1, Vec::new());
            }
            table[*id] = if base64 {
                base64::decode(token).unwrap_or_else(|| token.as_bytes().to_vec())
            } else {
                token.as_bytes().to_vec()
            };
        }

        let meta: Vec<bool> = table.iter().map(|t| is_meta_bytes(t)).collect();
        let blank = table
            .iter()
            .position(|t| {
                std::str::from_utf8(t).is_ok_and(|s| BLANK_NAMES.contains(&s))
            })
            .unwrap_or(0);

        Some(Self {
            table,
            meta,
            blank,
            base64,
        })
    }

    /// 词表大小。
    #[must_use]
    pub const fn len(&self) -> usize {
        self.table.len()
    }

    /// 词表是否为空。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// CTC blank 的 id。
    #[must_use]
    pub const fn blank_id(&self) -> usize {
        self.blank
    }

    /// 词表是否 base64（字节级 BPE）。
    #[must_use]
    pub const fn is_base64(&self) -> bool {
        self.base64
    }

    /// id → 可读文本（诊断用；字节级 token 可能不是合法 UTF-8，按 lossy 处理）。
    #[must_use]
    pub fn get(&self, id: usize) -> Option<String> {
        self.table
            .get(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// id → 字节。
    #[must_use]
    pub fn bytes(&self, id: usize) -> Option<&[u8]> {
        self.table.get(id).map(Vec::as_slice)
    }

    /// 该 id 是否是元 token（不属于正文）。
    #[must_use]
    pub fn is_meta(&self, id: usize) -> bool {
        self.meta.get(id).copied().unwrap_or(false)
    }

    /// token → id（线性查找，只在加载期定位特殊 token）。
    #[must_use]
    pub fn id_of(&self, token: &str) -> Option<usize> {
        self.table.iter().position(|t| t == token.as_bytes())
    }

    /// 把 id 序列拼成 `(文本, 语言标签)`。
    ///
    /// 元 token 不进正文；语言取第一个形如 `<|zh|>` 的标记。
    #[must_use]
    pub fn decode_ids(&self, ids: &[usize]) -> (String, Option<String>) {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 3);
        let mut language = None;
        for id in ids {
            let Some(tok) = self.bytes(*id) else { continue };
            if self.is_meta(*id) {
                if language.is_none()
                    && let Ok(s) = std::str::from_utf8(tok)
                    && let Some(lang) = language_tag(s)
                {
                    language = Some(lang.to_owned());
                }
                continue;
            }
            bytes.extend_from_slice(tok);
        }
        (cnt_asr::text::assemble_bytes(&bytes), language)
    }
}

/// 词表是否 base64 编码：抽样若干行，**全部**能解码才算。
///
/// 明文词表里的 `s` / `the` / `<unk>` / `▁the` 都不是合法 base64（长度或字符集
/// 不对），所以这个判据不会把明文误判成 base64。
fn looks_base64(rows: &[(&str, usize)]) -> bool {
    let sample: Vec<&str> = rows
        .iter()
        .map(|(t, _)| *t)
        .filter(|t| !t.is_empty())
        .take(200)
        .collect();
    sample.len() >= 16 && sample.iter().all(|t| base64::decode(t).is_some())
}

/// 元 token 判据：`<|...|>`（`SenseVoice`/whisper 风格）或 `<...>`（`<blk>`、
/// `<yunnan>`、`<sos>` 这类单尖括号写法）。
///
/// 两种都要认：`FireRedASR2` 用单尖括号，`Fun-ASR-Nano` 两种都有
/// （`<|zh|>`、`<|0.02|>` 时间戳，以及 `<blk>`）。
fn is_meta_bytes(token: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(token) else {
        return false;
    };
    s.len() >= 3 && s.starts_with('<') && s.ends_with('>')
}

/// `<|zh|>` → `zh`；情感/事件/时间戳标记返回 None。
fn language_tag(token: &str) -> Option<&str> {
    let inner = token
        .strip_prefix("<|")
        .and_then(|t| t.strip_suffix("|>"))
        .or_else(|| token.strip_prefix('<').and_then(|t| t.strip_suffix('>')))?;
    // 语言码是 2~3 个小写字母（zh/en/ja/ko/yue）；NEUTRAL/Speech/0.02 都不符合
    ((2..=3).contains(&inner.len()) && inner.chars().all(|c| c.is_ascii_lowercase()))
        .then_some(inner)
}

/// CTC 贪心解码：`logits[T, vocab]` → token id 序列（已去 blank 与重复）。
#[must_use]
pub fn greedy_decode(logits: &[f32], vocab: usize, blank: usize) -> Vec<usize> {
    let _span = fastrace::Span::enter_with_local_parent("ctc_decode");
    if vocab == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut prev = usize::MAX;
    for frame in logits.chunks_exact(vocab) {
        let (best, _) = frame
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| {
                if *v > acc.1 { (i, *v) } else { acc }
            });
        if best != blank && best != prev {
            out.push(best);
        }
        prev = best;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{greedy_decode, Tokens};

    fn frame(vocab: usize, hot: usize) -> Vec<f32> {
        let mut f = vec![0.0; vocab];
        f[hot] = 10.0;
        f
    }

    #[test]
    fn greedy_removes_blanks_and_repeats() {
        let vocab = 5;
        let mut logits = Vec::new();
        for hot in [0, 2, 2, 0, 2, 3, 3, 0] {
            logits.extend(frame(vocab, hot));
        }
        assert_eq!(greedy_decode(&logits, vocab, 0), vec![2, 2, 3]);
    }

    #[test]
    fn greedy_honours_non_zero_blank() {
        // Fun-ASR-Nano 的 blank 是最后一个 id，不是 0
        let vocab = 4;
        let mut logits = Vec::new();
        for hot in [3, 0, 0, 3, 1] {
            logits.extend(frame(vocab, hot));
        }
        assert_eq!(greedy_decode(&logits, vocab, 3), vec![0, 1]);
    }

    #[test]
    fn empty_logits_decode_to_nothing() {
        assert!(greedy_decode(&[], 5, 0).is_empty());
        assert!(greedy_decode(&[1.0, 2.0], 0, 0).is_empty());
    }

    #[test]
    fn parses_plain_sentencepiece_vocab() {
        // SenseVoice 风格
        let t = Tokens::parse(
            "<unk> 0\n<s> 1\n</s> 2\n\u{2581}the 3\ns 4\n\u{4f60} 5\n\u{597d} 6\n<|zh|> 7\n<|withitn|> 8\n",
        )
        .expect("parse");
        assert!(!t.is_base64());
        assert_eq!(t.len(), 9);
        assert_eq!(t.blank_id(), 0, "没有 <blk> 时退回 0");
        assert!(t.is_meta(7));
        assert!(!t.is_meta(5));
        let (text, lang) = t.decode_ids(&[7, 8, 5, 6]);
        assert_eq!(text, "\u{4f60}\u{597d}");
        assert_eq!(lang.as_deref(), Some("zh"));
    }

    #[test]
    fn parses_base64_byte_level_vocab() {
        // Fun-ASR-Nano 风格：base64；「你」= E4BDA0 被切成两个 token
        // 5L0= → E4 BD ， oA== → A0
        // 判据需要足够证据（≥16 个样本全部可解码），所以补一批单字节 token
        let vocab = concat!(
            "5L0= 0\noA== 1\n5aW9 2\nPHxaaHw+ 3\nPGJsaz4= 4\n",
            "YQ== 5\nYg== 6\nYw== 7\nZA== 8\nZQ== 9\nZg== 10\nZw== 11\naA== 12\naQ== 13\nag== 14\naw== 15\nbA== 16\nbQ== 17\nbg== 18\nbw== 19\ncA== 20\ncQ== 21\ncg== 22\ncw== 23\ndA== 24\n"
        );
        let t = Tokens::parse(vocab).expect("parse");
        assert!(t.is_base64(), "应识别为 base64 词表");
        assert_eq!(t.len(), 25); // 5 个正题 token + 20 个补足判据的单字节 token
        // <blk> 在 id 4
        assert_eq!(t.blank_id(), 4);
        // 半个汉字 + 另一半 → 拼字节后才是「你」
        let (text, _) = t.decode_ids(&[0, 1, 2]);
        assert_eq!(text, "\u{4f60}\u{597d}");
        // 元 token 不进正文
        assert!(t.is_meta(3));
        assert!(t.is_meta(4));
    }

    #[test]
    fn base64_meta_language_is_extracted() {
        // PHx6aHw+ = "<|zh|>"
        let t = Tokens::parse(concat!("PHx6aHw+ 0\n5aW9 1\n", "YQ== 5\nYg== 6\nYw== 7\nZA== 8\nZQ== 9\nZg== 10\nZw== 11\naA== 12\naQ== 13\nag== 14\naw== 15\nbA== 16\nbQ== 17\nbg== 18\nbw== 19\ncA== 20\ncQ== 21\ncg== 22\ncw== 23\ndA== 24\n")).expect("parse");
        let (text, lang) = t.decode_ids(&[0, 1]);
        assert_eq!(text, "\u{597d}");
        assert_eq!(lang.as_deref(), Some("zh"));
    }

    #[test]
    fn missing_token_file_is_an_error() {
        assert!(Tokens::load("/nonexistent/tokens.txt").is_err());
    }
}
