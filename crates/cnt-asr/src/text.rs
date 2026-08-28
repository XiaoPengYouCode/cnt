//! 文本后处理：token 序列 → 可上屏文本。
//!
//! 与模型无关的纯函数（好单测、好复用）：
//!
//! 1. [`assemble`]：丢掉 `<|zh|>` 一类元 token，处理 BPE 的词首标记 `▁`；
//! 2. [`polish`]：中文场景的空格与标点归一（CJK 之间不留空格、句末标点全角）。
//!
//! 为什么要 polish：模型的 token 表是中英日韩粤混合的 BPE，英文词首带 `▁`，
//! 直接拼出来会出现 `你好 ， 世界` 这种夹空格的结果——中文里空格是错的，
//! 但英文里必须留，所以规则得按字符类别判断，不能一刀切。

/// 元 token 的包裹形式（`<|zh|>`、`<|NEUTRAL|>`、`<|Speech|>`、`<|woitn|>` ……）。
const META_PREFIX: &str = "<|";
const META_SUFFIX: &str = "|>";
/// `SentencePiece` 的词首标记（U+2581 LOWER ONE EIGHTH BLOCK）。
const WORD_START: char = '\u{2581}';

/// 是否是元 token（语言/情感/事件/ITN 标记，不属于正文）。
#[must_use]
pub fn is_meta(token: &str) -> bool {
    token.starts_with(META_PREFIX) && token.ends_with(META_SUFFIX)
}

/// 元 token 去掉包裹后的内容（`<|zh|>` → `zh`）；非元 token 返回 None。
#[must_use]
pub fn meta_value(token: &str) -> Option<&str> {
    if !is_meta(token) {
        return None;
    }
    token
        .strip_prefix(META_PREFIX)
        .and_then(|t| t.strip_suffix(META_SUFFIX))
}

/// 是否是**不需要词间空格**的字符（中文/日文）。
///
/// 注意**不含谚文**（韩文 U+AC00–D7AF）：韩语是分词书写的，
/// 把它当 CJK 会把「조금만 생각을」的空格全吃掉（实测踩过）。
/// 这个函数的语义是「排版上不需要空格」，不是「属于 CJK 区」。
#[must_use]
pub const fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}'   // 假名
        | '\u{3400}'..='\u{4dbf}' // 扩展 A
        | '\u{4e00}'..='\u{9fff}' // 基本区
        | '\u{f900}'..='\u{faff}' // 兼容表意
    )
}

/// 中文标点（全角）：这些符号前面不该有空格。
#[must_use]
pub const fn is_cjk_punct(c: char) -> bool {
    matches!(
        c,
        '，' | '。'
            | '、'
            | '；'
            | '：'
            | '？'
            | '！'
            | '…'
            | '（'
            | '）'
            | '「'
            | '」'
            | '《'
            | '》'
    )
}

/// 字节序列 → 文本（字节级 BPE 的拼装入口）。
///
/// 字节级 BPE（tiktoken 系）的一个汉字会被切成 2~3 个 token，**必须先按字节
/// 拼接、再做一次 UTF-8 解码**；按 token 字符串逐个拼接会得到乱码。
/// 半个字符的残留（截断/识别抖动）用 lossy 解码兜住，不 panic。
#[must_use]
pub fn assemble_bytes(bytes: &[u8]) -> String {
    let raw = String::from_utf8_lossy(bytes);
    // SentencePiece 的词首标记在字节路径里也可能出现（明文词表走同一入口）
    let raw = raw.replace(WORD_START, " ");
    polish(&raw)
}

/// token 序列 → 原始文本（处理 `▁` 词首标记，丢弃元 token）。
///
/// 返回 `(文本, 语言标签)`：语言标签取第一个形如 `<|zh|>` 的元 token。
#[must_use]
pub fn assemble(tokens: &[String]) -> (String, Option<String>) {
    let mut out = String::with_capacity(tokens.len() * 3);
    let mut language = None;
    for token in tokens {
        if let Some(value) = meta_value(token) {
            // 语言标签是两三个字母的小写码（zh/en/ja/ko/yue），
            // 情感/事件标记（NEUTRAL/Speech）不符合这个形状，据此区分。
            if language.is_none()
                && (2..=3).contains(&value.len())
                && value.chars().all(|c| c.is_ascii_lowercase())
            {
                language = Some(value.to_owned());
            }
            continue;
        }
        if let Some(rest) = token.strip_prefix(WORD_START) {
            out.push(' ');
            out.push_str(rest);
        } else {
            out.push_str(token);
        }
    }
    (polish(&out), language)
}

/// 空格与标点归一：CJK 之间不留空格，标点前不留空格，半角标点在中文语境转全角。
#[must_use]
pub fn polish(raw: &str) -> String {
    // 1. 折叠连续空白并去首尾
    let mut compact = String::with_capacity(raw.len());
    let mut prev_space = true; // 开头等价于「前面是空格」→ 吃掉前导空白
    for c in raw.chars() {
        if c.is_whitespace() {
            if !prev_space {
                compact.push(' ');
            }
            prev_space = true;
        } else {
            compact.push(c);
            prev_space = false;
        }
    }
    while compact.ends_with(' ') {
        compact.pop();
    }

    // 2. 逐字符决定是否保留空格 + 半角标点转全角
    let chars: Vec<char> = compact.chars().collect();
    let mut out = String::with_capacity(compact.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let before = prev_visible(&chars, i);
            let after = chars.get(i + 1).copied();
            // CJK 与任意字符之间、标点之前，都不需要空格
            let drop_space = match (before, after) {
                (Some(b), Some(a)) => {
                    is_cjk(b) || is_cjk(a) || is_cjk_punct(a) || is_ascii_tail_punct(a)
                }
                _ => true,
            };
            if !drop_space {
                out.push(' ');
            }
            continue;
        }
        let cjk_context = prev_visible(&chars, i).is_some_and(is_cjk)
            || next_visible(&chars, i).is_some_and(is_cjk);
        out.push(if cjk_context {
            widen_punct(c).unwrap_or(c)
        } else {
            c
        });
    }
    out
}

/// 半角句读 → 全角（仅在中文语境用）。
const fn widen_punct(c: char) -> Option<char> {
    Some(match c {
        ',' => '，',
        '.' => '。',
        '?' => '？',
        '!' => '！',
        ':' => '：',
        ';' => '；',
        _ => return None,
    })
}

/// 会「贴在前一个词尾」的半角标点（前面不留空格）。
const fn is_ascii_tail_punct(c: char) -> bool {
    matches!(c, ',' | '.' | '?' | '!' | ':' | ';' | ')' | ']' | '%')
}

/// 位置 i 之前最近的非空格字符。
fn prev_visible(chars: &[char], i: usize) -> Option<char> {
    chars[..i].iter().rev().copied().find(|c| *c != ' ')
}

/// 位置 i 之后最近的非空格字符。
fn next_visible(chars: &[char], i: usize) -> Option<char> {
    chars.get(i + 1..)?.iter().copied().find(|c| *c != ' ')
}

#[cfg(test)]
mod tests {
    use super::{assemble, is_meta, meta_value, polish};

    fn toks(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn meta_tokens_are_recognized() {
        assert!(is_meta("<|zh|>"));
        assert!(!is_meta("你"));
        assert_eq!(meta_value("<|NEUTRAL|>"), Some("NEUTRAL"));
        assert_eq!(meta_value("好"), None);
    }

    #[test]
    fn assemble_drops_meta_and_keeps_language() {
        let (text, lang) = assemble(&toks(&[
            "<|zh|>",
            "<|NEUTRAL|>",
            "<|Speech|>",
            "<|woitn|>",
            "今",
            "天",
            "天",
            "气",
            "不",
            "错",
        ]));
        assert_eq!(text, "今天天气不错");
        assert_eq!(lang.as_deref(), Some("zh"));
    }

    #[test]
    fn assemble_handles_english_word_starts() {
        let (text, _) = assemble(&toks(&["\u{2581}HELLO", "\u{2581}WORLD"]));
        assert_eq!(text, "HELLO WORLD");
    }

    #[test]
    fn assemble_mixes_chinese_and_english_without_stray_spaces() {
        // 中英混排：英文词之间留空格，中英之间不留
        let (text, _) = assemble(&toks(&["我", "用", "\u{2581}RUST", "写", "输", "入", "法"]));
        assert_eq!(text, "我用RUST写输入法");
    }

    #[test]
    fn polish_normalizes_punctuation_in_chinese_context() {
        assert_eq!(polish("你好 , 世界 ."), "你好，世界。");
        // 纯英文语境不动半角标点，也保留词间空格
        assert_eq!(polish("hello , world ."), "hello, world.");
    }

    #[test]
    fn polish_keeps_korean_word_spacing() {
        // 韩语分词书写：空格不能吃（谚文不算「无需空格」的 CJK）
        assert_eq!(polish("조금만 생각을 하면서"), "조금만 생각을 하면서");
    }

    #[test]
    fn polish_trims_and_collapses_whitespace() {
        assert_eq!(polish("  hello   world  "), "hello world");
        assert_eq!(polish(""), "");
        assert_eq!(polish("   "), "");
    }
}
