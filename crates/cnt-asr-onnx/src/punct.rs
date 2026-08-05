//! CT-Transformer 标点恢复（`FunASR` 的 `ct-punc`，ONNX）。
//!
//! ```text
//!   光板文本 ─► 切成「词」（汉字逐字，拉丁词整体）─► id 序列
//!            ─► 分段(20)推理 ─► 每个位置 argmax 出标点类别 ─► 插回文本
//! ```
//!
//! ## 为什么需要它
//!
//! 语音输入的文本要直接进聊天框/文档，没有句读等于每句都要手动补。
//! 而「声学模型会不会吐标点」是偶然的（见 `cnt-asr` 的端口说明），
//! 所以标点独立成一段可插拔的推理。
//!
//! ## 契约（来自模型仓库的 `test.py` / `show-model-input-output.py`）
//!
//! | | |
//! |---|---|
//! | 输入 | `inputs: int32[B, L]`（词 id）、`text_lengths: int32[B]` |
//! | 输出 | `logits: float32[B, L, 6]`，最后一维 argmax = 标点类别 |
//! | metadata | `tokens`（`\|` 分隔，27 万条）、`punctuations`（`\|` 分隔）、`unk_symbol` |
//!
//! ## 分段策略（与官方实现对齐）
//!
//! 每 20 个词推一次，但**不是**简单切块：每段推完从后往前找最后一个句号/问号，
//! 只采纳到那里为止，剩下的词退回下一段重推。这样句子不会被段边界切断
//! （切断会导致边界处的标点判断失去右侧上下文，句号乱放）。
//! 超过 `MAX_LEN` 个词还找不到句号，就把最后一个逗号升级成句号——
//! 否则一段没有句读的长语音会让整段无限累积。

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use cnt_asr::{AsrError, Punctuator};
use fastrace::local::LocalSpan;
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};

/// 每次推理的词数。
const SEGMENT: usize = 20;
/// 强制断句的词数上限。
const MAX_LEN: usize = 200;
/// 「此处无标点」的类别在不同导出里写法不一（`_`、`<unk>`、空串都见过），
/// 所以判据反过来：**只有真正的标点字符才允许写进文本**（白名单）。
///
/// 这条是踩出来的：本模型的 `punctuations` 元数据是 `<unk>|_|，|。|？|、`，
/// 类别 0 就是字面量 `<unk>`，且它正是「无标点」的默认类。用黑名单（只跳过 `_`）
/// 会把 `<unk>` 当标点插进每个字之间——上屏文本变成「不<unk>不<unk>不」。
fn is_real_punct(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            matches!(c, '，' | '。' | '、' | '；' | '：' | '？' | '！' | '…')
                || matches!(c, ',' | '.' | ';' | ':' | '?' | '!')
        })
}

/// CT-Transformer 标点模型。
pub struct CtPunctuator {
    session: Mutex<Session>,
    /// 词 → id。
    vocab: HashMap<String, i32>,
    /// 未登录词 id。
    unk: i32,
    /// 类别 id → 标点串（`_` 表示不加）。
    puncts: Vec<String>,
    /// 句末标点的类别 id（句号/问号）。
    sentence_end: Vec<usize>,
    /// 逗号的类别 id（超长时升级成句号用）。
    comma: Option<usize>,
    /// 句号的类别 id。
    period: Option<usize>,
    /// 「无标点」的类别 id（用于补齐尾部；本模型是 0 = `<unk>`）。
    none_class: usize,
    /// 输入名（按模型声明取，不硬编码）。
    ids_input: String,
    lens_input: String,
    lens_dtype: TensorElementType,
    name: String,
}

impl CtPunctuator {
    /// 加载标点模型（`model.int8.onnx` 或 `model.onnx`）。
    ///
    /// # Errors
    /// 模型打不开、metadata 缺少 `tokens`/`punctuations` 时返回错误。
    pub fn open(model: impl AsRef<Path>, threads: usize) -> Result<Self, AsrError> {
        let _span = LocalSpan::enter_with_local_parent("punct_open");
        let model = model.as_ref();
        let mut builder = Session::builder().map_err(err)?;
        if threads > 0 {
            builder = builder.with_intra_threads(threads).map_err(err)?;
        }
        let session = builder
            .commit_from_file(model)
            .map_err(|e| AsrError::Model(format!("{}: {e}", model.display())))?;

        let meta = session.metadata().map_err(err)?;
        let tokens = meta.custom("tokens").ok_or_else(|| {
            AsrError::Model(format!("{}: missing `tokens` metadata", model.display()))
        })?;
        let punctuations = meta.custom("punctuations").ok_or_else(|| {
            AsrError::Model(format!(
                "{}: missing `punctuations` metadata",
                model.display()
            ))
        })?;
        let unk_symbol = meta
            .custom("unk_symbol")
            .unwrap_or_else(|| "<unk>".to_owned());
        drop(meta);

        let mut vocab: HashMap<String, i32> = HashMap::new();
        for (i, token) in tokens.split('|').enumerate() {
            vocab.insert(token.to_owned(), i32::try_from(i).unwrap_or(0));
        }
        let unk = *vocab.get(unk_symbol.trim()).ok_or_else(|| {
            AsrError::Model(format!("{}: unk symbol not in vocab", model.display()))
        })?;

        let puncts: Vec<String> = punctuations.split('|').map(str::to_owned).collect();
        let find = |p: &str| puncts.iter().position(|x| x == p);
        let period = find("。");
        let comma = find("，");
        let sentence_end: Vec<usize> = [period, find("？"), find("！")]
            .into_iter()
            .flatten()
            .collect();
        // 「无标点」类：取第一个不是真标点的类别（`<unk>` 或 `_`）
        let none_class = puncts.iter().position(|p| !is_real_punct(p)).unwrap_or(0);

        // 输入名与 dtype 按声明取（其他导出可能叫 text/text_len）
        let mut ids_input = "inputs".to_owned();
        let mut lens_input = "text_lengths".to_owned();
        let mut lens_dtype = TensorElementType::Int32;
        for outlet in session.inputs() {
            let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
                continue;
            };
            // rank ≥2 = [B, L] 的 id 序列；rank 1 = [B] 的长度
            if shape.len() >= 2 {
                outlet.name().clone_into(&mut ids_input);
            } else {
                outlet.name().clone_into(&mut lens_input);
                lens_dtype = *ty;
            }
        }
        log::info!("punct model inputs: {ids_input} / {lens_input} ({lens_dtype:?}), classes={puncts:?}");

        let name = format!(
            "ct-punct({})",
            model
                .file_name()
                .map_or_else(|| "?".into(), |n| n.to_string_lossy())
        );
        log::info!(
            "punct backend: {name}; vocab={}, classes={puncts:?}, none={none_class}",
            vocab.len()
        );

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            unk,
            puncts,
            sentence_end,
            comma,
            period,
            none_class,
            ids_input,
            lens_input,
            lens_dtype,
            name,
        })
    }

    /// 一段词 id 的推理：返回每个位置的标点类别。
    // significant_drop_tightening: outputs 借用 session，锁必须覆盖 run + 取输出
    #[allow(clippy::significant_drop_tightening)]
    fn infer(&self, ids: &[i32]) -> Result<Vec<usize>, AsrError> {
        let len = i64::try_from(ids.len()).unwrap_or(0);
        let tokens = Tensor::from_array((vec![1_i64, len], ids.to_vec()))
            .map_err(err)?
            .into_dyn();
        let lengths = match self.lens_dtype {
            TensorElementType::Int64 => Tensor::from_array((vec![1_i64], vec![len]))
                .map_err(err)?
                .into_dyn(),
            _ => Tensor::from_array((vec![1_i64], vec![i32::try_from(ids.len()).unwrap_or(0)]))
                .map_err(err)?
                .into_dyn(),
        };

        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session
            .run(vec![
                (
                    std::borrow::Cow::Owned(self.ids_input.clone()),
                    ort::session::SessionInputValue::from(tokens),
                ),
                (
                    std::borrow::Cow::Owned(self.lens_input.clone()),
                    ort::session::SessionInputValue::from(lengths),
                ),
            ])
            .map_err(err)?;
        let logits = outputs
            .get("logits")
            .ok_or_else(|| AsrError::Backend("punct model produced no logits".to_owned()))?;
        let (shape, data) = logits.try_extract_tensor::<f32>().map_err(err)?;
        let classes = shape
            .last()
            .and_then(|d| usize::try_from(*d).ok())
            .filter(|c| *c > 0)
            .ok_or_else(|| AsrError::Backend(format!("bad punct logits shape: {shape:?}")))?;
        Ok(data
            .chunks_exact(classes)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |acc, (i, v)| {
                        if *v > acc.1 { (i, *v) } else { acc }
                    })
                    .0
            })
            .collect())
    }
}

impl Punctuator for CtPunctuator {
    fn restore(&self, text: &str) -> Result<String, AsrError> {
        let _span = LocalSpan::enter_with_local_parent("punctuate");
        let words = split_words(text);
        if words.is_empty() {
            return Ok(String::new());
        }
        let ids: Vec<i32> = words
            .iter()
            .map(|w| *self.vocab.get(w.text).unwrap_or(&self.unk))
            .collect();

        // 分段推理 + 窗口回溯（详见模块文档）
        let n = ids.len();
        let mut classes: Vec<usize> = Vec::with_capacity(n);
        // 上一次已确认到的位置：下一段从这里重新开始推（而不是丢掉这一段的标点）
        let mut resume: Option<usize> = None;
        let segments = n.div_ceil(SEGMENT);
        for i in 0..segments {
            let start = resume.unwrap_or(i * SEGMENT);
            let end = ((i + 1) * SEGMENT).min(n);
            if start >= end {
                continue;
            }
            let chunk = &ids[start..end];
            let mut out = self.infer(chunk)?;

            // 从后往前找句末标点
            let mut cut = out.iter().rposition(|c| self.sentence_end.contains(c));
            if cut.is_none() && chunk.len() >= MAX_LEN {
                // 太长还没句号：把最后一个逗号升级成句号，否则窗口无限增长
                if let (Some(comma), Some(period)) = (self.comma, self.period)
                    && let Some(pos) = out.iter().rposition(|c| *c == comma)
                {
                    out[pos] = period;
                    cut = Some(pos);
                }
            }

            if let Some(pos) = cut {
                // 采纳到句末为止；重叠部分以本次结果为准
                classes.truncate(start);
                classes.extend_from_slice(&out[..=pos]);
                resume = Some(start + pos + 1);
            } else {
                // 没找到句末：窗口保留起点，连着下一段一起重推
                if resume.is_none() {
                    resume = Some(start);
                }
                if i + 1 == segments {
                    // 最后一段：全部采纳
                    classes.truncate(start);
                    classes.extend_from_slice(&out);
                }
            }
        }
        // 尾部补「无标点」：用第一个非标点类别（本模型是 0 = `<unk>`）
        classes.resize(n, self.none_class);

        Ok(join(&words, &classes, &self.puncts))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 一个切出来的词，以及它在原文里**前面是否有空白**。
///
/// 记住空白是契约要求的：端口承诺「只加标点、不改字」，而空格属于原文。
/// 早先的实现把空白丢掉、只在两个拉丁词之间补回来，结果韩语（分词书写）
/// 的空格被全部吃掉：`조금만 생각을 하면서` → `조금만생각을하면서`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word<'a> {
    pub text: &'a str,
    /// 原文中该词之前有空白。
    pub space_before: bool,
}

/// 把文本切成模型认识的「词」：CJK 逐字，拉丁按字母数字边界成词。
///
/// 与官方实现同口径（汉字一个 id，英文单词一个 id），否则 id 序列错位、
/// 标点会落到奇怪的位置。空白不产生词，但会记在下一个词的 `space_before` 上。
#[must_use]
pub fn split_words(text: &str) -> Vec<Word<'_>> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut pending_space = false;
    while i < text.len() {
        let Some(c) = text[i..].chars().next() else {
            break;
        };
        let width = c.len_utf8();
        if c.is_whitespace() {
            pending_space = true;
            i += width;
            continue;
        }
        let (text_slice, next) = if c.is_ascii_alphanumeric() {
            // 拉丁词/数字：吃到非字母数字为止
            let mut j = i;
            while j < text.len() && bytes[j].is_ascii_alphanumeric() {
                j += 1;
            }
            (&text[i..j], j)
        } else {
            (&text[i..i + width], i + width)
        };
        out.push(Word {
            text: text_slice,
            space_before: pending_space,
        });
        pending_space = false;
        i = next;
    }
    out
}

/// 词 + 标点类别 → 文本。**原文的空白原样保留**（只加标点，不改字、不改空格）。
fn join(words: &[Word<'_>], classes: &[usize], puncts: &[String]) -> String {
    let mut out = String::with_capacity(words.len() * 4);
    for (i, w) in words.iter().enumerate() {
        let word = w.text;
        if w.space_before && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
        if let Some(p) = classes.get(i).and_then(|c| puncts.get(*c))
            && is_real_punct(p)
        {
            // 纯拉丁语境用半角：模型是中英混合训练的，但类别表只有全角标点，
            // 英文句子后面跟一个「。」很刺眼（实测 en.wav 就是这样）
            let latin = word.chars().next().is_some_and(|c| c.is_ascii_alphanumeric());
            match (latin, narrow(p)) {
                (true, Some(half)) => out.push_str(half),
                _ => out.push_str(p),
            }
        }
    }
    out
}

/// 全角标点 → 半角（拉丁语境用）。
fn narrow(p: &str) -> Option<&'static str> {
    Some(match p {
        "。" => ".",
        "，" | "、" => ",",
        "？" => "?",
        "！" => "!",
        "：" => ":",
        "；" => ";",
        _ => return None,
    })
}

/// ort 错误 → 端口错误。
fn err<E: std::fmt::Display>(e: E) -> AsrError {
    AsrError::Backend(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{join, split_words};

    fn texts<'a>(words: &[super::Word<'a>]) -> Vec<&'a str> {
        words.iter().map(|w| w.text).collect()
    }

    fn w(text: &str, space_before: bool) -> super::Word<'_> {
        super::Word { text, space_before }
    }

    #[test]
    fn splits_chinese_by_character() {
        assert_eq!(texts(&split_words("你好世界")), vec!["你", "好", "世", "界"]);
    }

    #[test]
    fn splits_latin_by_word() {
        assert_eq!(
            texts(&split_words("hello world 123")),
            vec!["hello", "world", "123"]
        );
    }

    #[test]
    fn splits_mixed_text() {
        assert_eq!(
            texts(&split_words("我用 RUST 写输入法")),
            vec!["我", "用", "RUST", "写", "输", "入", "法"]
        );
    }

    #[test]
    fn records_original_spacing() {
        let words = split_words("조금만 생각을 하면서");
        // 韩语是分词书写的，空格必须记下来
        assert!(!words[0].space_before);
        let spaced: Vec<bool> = words.iter().map(|w| w.space_before).collect();
        assert!(spaced.iter().filter(|s| **s).count() >= 2, "{spaced:?}");
    }

    #[test]
    fn ignores_whitespace_and_empty() {
        assert!(split_words("").is_empty());
        assert!(split_words("   ").is_empty());
        assert_eq!(texts(&split_words(" 你  好 ")), vec!["你", "好"]);
    }

    #[test]
    fn join_preserves_original_spacing() {
        // 只加标点，不动空格（端口契约）
        let puncts = vec!["_".to_owned(), "。".to_owned()];
        let words = [w("조금만", false), w("생각을", true), w("하면서", true)];
        assert_eq!(
            join(&words, &[0, 0, 1], &puncts),
            "조금만 생각을 하면서。"
        );
    }

    #[test]
    fn join_inserts_punctuation() {
        let puncts = vec![
            "_".to_owned(),
            "，".to_owned(),
            "。".to_owned(),
            "？".to_owned(),
        ];
        let words = [
            w("今", false),
            w("天", false),
            w("天", false),
            w("气", false),
            w("不", false),
            w("错", false),
        ];
        // 「今天天气」后逗号，末尾句号
        assert_eq!(join(&words, &[0, 0, 0, 1, 0, 2], &puncts), "今天天气，不错。");
    }

    #[test]
    fn join_keeps_latin_word_spacing() {
        let puncts = vec!["_".to_owned(), "，".to_owned()];
        let words = [w("hello", false), w("world", true), w("你", false), w("好", false)];
        // 英文词后面的逗号用半角；空格来自原文
        assert_eq!(join(&words, &[0, 1, 0, 0], &puncts), "hello world,你好");
    }

    #[test]
    fn join_uses_halfwidth_after_latin() {
        let puncts = vec!["_".to_owned(), "。".to_owned()];
        assert_eq!(join(&[w("hello", false)], &[1], &puncts), "hello.");
        // 中文语境仍是全角
        assert_eq!(join(&[w("好", false)], &[1], &puncts), "好。");
    }

    /// 本模型真实的类别表：`<unk>` 在 id 0，且它就是「无标点」的默认类。
    fn real_puncts() -> Vec<String> {
        ["<unk>", "_", "，", "。", "？", "、"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    #[test]
    fn unk_class_is_never_written_into_text() {
        // 这是线上踩到的 bug：上屏文本变成「不<unk>不<unk>不」
        let puncts = real_puncts();
        let words = [w("不", false), w("不", false), w("不", false)];
        assert_eq!(join(&words, &[0, 0, 0], &puncts), "不不不");
        // `_` 同样不输出
        assert_eq!(join(&words, &[1, 1, 1], &puncts), "不不不");
        // 真标点照常输出
        assert_eq!(join(&words, &[0, 2, 3], &puncts), "不不，不。");
    }

    #[test]
    fn none_class_detection_picks_non_punct() {
        let puncts = real_puncts();
        let none = puncts.iter().position(|p| !super::is_real_punct(p));
        assert_eq!(none, Some(0), "「无标点」类应是 id 0（<unk>）");
    }

    #[test]
    fn is_real_punct_whitelist() {
        assert!(super::is_real_punct("。"));
        assert!(super::is_real_punct("，"));
        assert!(super::is_real_punct("?"));
        assert!(!super::is_real_punct("<unk>"));
        assert!(!super::is_real_punct("_"));
        assert!(!super::is_real_punct(""));
        assert!(!super::is_real_punct("的"));
    }

    #[test]
    fn join_tolerates_short_class_list() {
        // 类别数组比词短（不该发生）也不 panic
        let puncts = vec!["_".to_owned()];
        assert_eq!(join(&[w("你", false), w("好", false)], &[0], &puncts), "你好");
    }
}
