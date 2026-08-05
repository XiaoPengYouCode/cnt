//! cnt-asr-onnx —— `SenseVoice` (ONNX Runtime) 识别后端，实现 `cnt_asr::Recognizer`。
//!
//! ```text
//!   16k f32 ─► fbank(80) ─► LFR(7,6) ─► CMVN ─► ONNX encoder+CTC ─► 贪心解码 ─► 文本
//!             └─ fbank.rs ─┴──── frontend.rs ────┘   ort（CPU EP）   └─ ctc.rs ─┘
//! ```
//!
//! 选 `SenseVoiceSmall` 的理由（不是随便挑的）：
//!
//! - **非流式、单次前向**：按住说话/VAD 切句天然是「一段音频进、一段文本出」，
//!   不需要流式模型那套 chunk 状态管理，代码少一个数量级；
//! - **自带标点与 ITN**：`textnorm=withitn` 时输出就带「，。？」和数字规整，
//!   省掉一个独立的标点模型（否则语音输入几乎不可用）；
//! - **中英日韩粤同模型**：中英混说不用切引擎；
//! - CPU 上 RTF ≈ 0.05~0.15（4 线程），一句 3 秒的话推理 150~450 ms。
//!
//! **换 NPU 的路径**：本文件不含任何 EP 相关代码——`ort` 的 execution provider
//! 是构建期 feature + 运行期 `with_execution_providers`，届时只在
//! [`SenseVoice::open`] 里加一行注册，前后端与调用方都不用改。
//!
//! 模型文件（约 500 MB，不入库，按 `data/` 的规矩独立分发）：
//! `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-*` 里的 `model.int8.onnx` + `tokens.txt`。

pub mod base64;
pub mod ctc;
pub mod fbank;
pub mod frontend;
pub mod inspect;
pub mod punct;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use fastrace::local::LocalSpan;
use fastrace::{Event, Span};
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, Value, ValueType};

use cnt_asr::{AsrError, Recognizer, Transcript, SAMPLE_RATE};

use crate::ctc::{prefix_beam_search, Tokens, PREFIX_BEAM};
use crate::fbank::{Fbank, FbankOptions};
use crate::frontend::{Cmvn, Lfr};

/// 特征输入的默认名（模型没声明浮点输入时的兜底）。
const IN_SPEECH: &str = "speech";
/// 输出名。
const OUT_LOGITS: &str = "logits";

/// 太短的音频不送模型（低于此长度只会得到噪声 token）。
const MIN_SAMPLES: usize = SAMPLE_RATE as usize / 10; // 100 ms

/// 语言 id 的兜底值（模型 metadata 缺失时用；与 `SenseVoice` 训练侧一致）。
const FALLBACK_LANG: &[(&str, i64)] = &[
    ("auto", 0),
    ("zh", 3),
    ("en", 4),
    ("yue", 7),
    ("ja", 11),
    ("ko", 12),
    ("nospeech", 13),
];
/// ITN 开关的兜底 id。
const FALLBACK_ITN: (i64, i64) = (14, 15); // (with_itn, without_itn)

/// 后端配置。
#[derive(Debug, Clone)]
pub struct SenseVoiceConfig {
    /// `model.onnx` / `model.int8.onnx` 路径。
    pub model: PathBuf,
    /// `tokens.txt` 路径。
    pub tokens: PathBuf,
    /// 推理线程数（0 = 交给 ORT 自己定）。
    pub threads: usize,
    /// 语言：`auto`/`zh`/`en`/`ja`/`ko`/`yue`。
    pub language: String,
    /// 是否开启 ITN（数字/日期规整 + 标点）。
    pub itn: bool,
    /// 覆盖幅度口径（None = 按模型 metadata 判断）。
    pub sample_scale: Option<f32>,
}

impl SenseVoiceConfig {
    /// 从模型目录推断文件名：`<dir>/model.int8.onnx`（不存在则 `model.onnx`）+ `<dir>/tokens.txt`。
    #[must_use]
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let int8 = dir.join("model.int8.onnx");
        let model = if int8.exists() {
            int8
        } else {
            dir.join("model.onnx")
        };
        Self {
            model,
            tokens: dir.join("tokens.txt"),
            threads: 4,
            language: "auto".to_owned(),
            itn: true,
            sample_scale: None,
        }
    }
}

/// `SenseVoice` 识别器。
///
/// `Session::run` 需要 `&mut self`，而端口是 `&self`（多线程共享），
/// 所以 session 用 `Mutex` 包起来：一次调用几百毫秒，串行化不是瓶颈，
/// 反而避免了同时跑两次推理把 CPU 打满、两句都变慢。
pub struct SenseVoice {
    session: Mutex<Session>,
    tokens: Tokens,
    fbank: Fbank,
    lfr: Lfr,
    cmvn: Cmvn,
    /// 特征输入名（不同导出叫 `speech` / `x` / `feats`）。
    speech_input: String,
    /// 其余输入：**只喂模型声明了的**（`Fun-ASR-Nano` 的 CTC 导出就没有
    /// `language`/`textnorm`）。dtype 也按声明来（int32 / int64 都见过）。
    aux_inputs: Vec<(String, TensorElementType, AuxKind)>,
    language_id: i64,
    textnorm_id: i64,
    name: String,
}

/// 辅助输入的语义（值从哪来）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuxKind {
    /// 特征帧数。
    Lengths,
    /// 语言 id。
    Language,
    /// ITN 开关 id。
    Textnorm,
    /// 认不出来的整型输入：喂 0 并告警（比直接崩掉好，也留下线索）。
    Unknown,
}

impl AuxKind {
    /// 按输入名猜语义（sherpa 的各家导出命名不统一，但都很直白）。
    fn classify(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        if n.contains("len") {
            Self::Lengths
        } else if n.contains("lang") {
            Self::Language
        } else if n.contains("textnorm") || n.contains("itn") {
            Self::Textnorm
        } else {
            Self::Unknown
        }
    }
}

impl SenseVoice {
    /// 加载模型与词表。
    ///
    /// # Errors
    /// 模型/词表打不开、metadata 不合规或 ORT 初始化失败时返回错误。
    pub fn open(config: &SenseVoiceConfig) -> Result<Self, AsrError> {
        let _span = Span::enter_with_local_parent("asr_open");
        let tokens = Tokens::load(&config.tokens)?;

        let mut builder = Session::builder().map_err(ort_err)?;
        if config.threads > 0 {
            builder = builder.with_intra_threads(config.threads).map_err(ort_err)?;
        }
        let session = builder
            .commit_from_file(&config.model)
            .map_err(|e| AsrError::Model(format!("{}: {e}", config.model.display())))?;

        // ---- 从 ONNX metadata 取前端参数（sherpa-onnx 导出的模型都带）----
        let meta = session.metadata().map_err(ort_err)?;
        let get = |key: &str| meta.custom(key);
        let lfr = Lfr {
            window: get("lfr_window_size")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(7),
            shift: get("lfr_window_shift")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(6),
        };
        let cmvn = if let (Some(m), Some(s)) = (get("neg_mean"), get("inv_stddev")) {
            Cmvn::parse(&m, &s)
        } else {
            // 不是错误：`Fun-ASR-Nano` 的导出把归一化折进了模型权重，
            // metadata 里就没有这两项（实测输出正常）。只在这里留一条记录。
            log::info!("model has no CMVN metadata (neg_mean/inv_stddev); skipping normalization");
            Cmvn::default()
        };
        // normalize_samples=1 表示模型吃 [-1,1]；否则要还原到 int16 量级
        let normalize = get("normalize_samples").is_some_and(|v| v.trim() == "1");
        let sample_scale = config
            .sample_scale
            .unwrap_or(if normalize { 1.0 } else { 32_768.0 });

        let language_id = get(&format!("lang_{}", config.language))
            .and_then(|v| v.trim().parse::<i64>().ok())
            .or_else(|| {
                FALLBACK_LANG
                    .iter()
                    .find(|(k, _)| *k == config.language)
                    .map(|(_, v)| *v)
            })
            .ok_or_else(|| AsrError::Config(format!("unknown language: {}", config.language)))?;
        let textnorm_key = if config.itn { "with_itn" } else { "without_itn" };
        let textnorm_id = get(textnorm_key)
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(if config.itn {
                FALLBACK_ITN.0
            } else {
                FALLBACK_ITN.1
            });
        drop(meta);

        let (speech_input, aux_inputs) = discover_inputs(&session);

        let fbank = Fbank::new(FbankOptions {
            sample_scale,
            ..FbankOptions::default()
        });

        let name = format!(
            "sensevoice({}, lang={}, itn={})",
            config
                .model
                .file_name()
                .map_or_else(|| "?".into(), |n| n.to_string_lossy()),
            config.language,
            config.itn
        );
        log::info!(
            "asr backend: {name}; vocab={} ({}), blank={}, lfr=({},{}), cmvn={}, scale={sample_scale}",
            tokens.len(),
            if tokens.is_base64() { "base64 byte-BPE" } else { "plain" },
            tokens.blank_id(),
            lfr.window,
            lfr.shift,
            if cmvn.is_valid() { "yes" } else { "MISSING" }
        );

        Ok(Self {
            session: Mutex::new(session),
            tokens,
            fbank,
            lfr,
            cmvn,
            speech_input,
            aux_inputs,
            language_id,
            textnorm_id,
            name,
        })
    }

    /// 声学前端：样本 → `(特征, 帧数, 维数)`。
    fn features(&self, samples: &[f32]) -> (Vec<f32>, usize, usize) {
        let _span = Span::enter_with_local_parent("asr_frontend");
        let dim = self.fbank.num_bins();
        let feats = self.fbank.compute(samples);
        let mut feats = self.lfr.apply(&feats, dim);
        let out_dim = self.lfr.out_dim(dim);
        self.cmvn.apply(&mut feats);
        let frames = feats.len().checked_div(out_dim).unwrap_or(0);
        feats.shrink_to_fit();
        (feats, frames, out_dim)
    }

    /// 按模型声明的 dtype 造一个 `[1]` 的整型标量张量。
    fn scalar(value: i64, dtype: TensorElementType) -> Result<Value, AsrError> {
        let shape = vec![1_i64];
        let v = match dtype {
            TensorElementType::Int64 => Tensor::from_array((shape, vec![value]))
                .map_err(ort_err)?
                .into_dyn(),
            _ => Tensor::from_array((shape, vec![i32::try_from(value).unwrap_or(0)]))
                .map_err(ort_err)?
                .into_dyn(),
        };
        Ok(v)
    }
}

impl Recognizer for SenseVoice {
    // significant_drop_tightening: session 锁必须覆盖 run + 取输出（outputs 借用 session）
    #[allow(clippy::significant_drop_tightening)]
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
        let _span = Span::enter_with_local_parent("asr_transcribe");
        if samples.len() < MIN_SAMPLES {
            return Ok(Transcript::default());
        }
        LocalSpan::add_event(
            Event::new("audio").with_property(|| ("samples", samples.len().to_string())),
        );

        let (feats, frames, dim) = self.features(samples);
        if frames == 0 {
            return Ok(Transcript::default());
        }
        LocalSpan::add_event(Event::new("features").with_property(|| ("frames", frames.to_string())));

        let shape = vec![
            1_i64,
            i64::try_from(frames).unwrap_or(i64::MAX),
            i64::try_from(dim).unwrap_or(i64::MAX),
        ];
        let speech = Tensor::from_array((shape, feats))
            .map_err(ort_err)?
            .into_dyn();
        let mut inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(1 + self.aux_inputs.len());
        inputs.push((
            std::borrow::Cow::Owned(self.speech_input.clone()),
            SessionInputValue::from(speech),
        ));
        for (name, dtype, kind) in &self.aux_inputs {
            let value = match kind {
                AuxKind::Lengths => i64::try_from(frames).unwrap_or(i64::MAX),
                AuxKind::Language => self.language_id,
                AuxKind::Textnorm => self.textnorm_id,
                AuxKind::Unknown => 0,
            };
            inputs.push((
                std::borrow::Cow::Owned(name.clone()),
                SessionInputValue::from(Self::scalar(value, *dtype)?),
            ));
        }

        let (hyps, vocab) = {
            let _span = Span::enter_with_local_parent("asr_infer");
            // outputs 借用 session，锁必须活到取完 logits 为止 —— 这正是
            // 「一次推理串行化」的设计意图，不是可以收紧的临时借用。
            let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
            let outputs = session.run(inputs).map_err(ort_err)?;
            // 输出名以 `logits` 为准；导出脚本改过名时退回第一个输出
            let logits = match outputs.get(OUT_LOGITS) {
                Some(v) => v,
                None if outputs.len() > 0 => &outputs[0],
                None => {
                    return Err(AsrError::Backend("model produced no logits".to_owned()));
                }
            };
            let (shape, data) = logits.try_extract_tensor::<f32>().map_err(ort_err)?;
            let vocab = shape
                .last()
                .and_then(|d| usize::try_from(*d).ok())
                .ok_or_else(|| AsrError::Backend(format!("bad logits shape: {shape:?}")))?;
            (
                prefix_beam_search(data, vocab, self.tokens.blank_id(), PREFIX_BEAM),
                vocab,
            )
        };

        // n-best：每条前缀各自拼成文本（去重后保留声学分数）
        let mut alternatives: Vec<cnt_asr::Hypothesis> = Vec::with_capacity(hyps.len());
        let mut language = None;
        for hyp in &hyps {
            let (text, lang) = self.tokens.decode_ids(&hyp.ids);
            if text.is_empty() {
                continue;
            }
            if language.is_none() {
                language = lang;
            }
            if alternatives.iter().any(|a| a.text == text) {
                continue; // 不同 token 路径可能拼出同一文本
            }
            alternatives.push(cnt_asr::Hypothesis {
                text,
                acoustic: hyp.logp,
            });
        }
        let ids: &[usize] = hyps.first().map_or(&[], |h| h.ids.as_slice());
        LocalSpan::add_event(
            Event::new("decoded")
                .with_property(|| ("tokens", ids.len().to_string()))
                .with_property(|| ("nbest", alternatives.len().to_string()))
                .with_property(|| ("vocab", vocab.to_string())),
        );
        // tokens 保留可读形式，供诊断（乱码时一眼看出是拼装还是识别问题）
        let tokens = ids
            .iter()
            .filter_map(|id| self.tokens.get(*id))
            .collect::<Vec<_>>();
        Ok(Transcript {
            text: alternatives.first().map(|a| a.text.clone()).unwrap_or_default(),
            tokens,
            language,
            alternatives,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 发现输入契约：谁是特征、还有哪些辅助输入、各自什么 dtype。
///
/// 教训：`SenseVoice-Small` 声明 `speech`/`speech_lengths`/`language`/`textnorm` 四个输入，
/// 而 `Fun-ASR-Nano` 的 CTC 导出只有一个 `x`。硬编码输入名，换模型就直接报错；
/// 更糟的是名字碰巧对上而语义不对——那会得到「能跑但输出乱码」。
fn discover_inputs(session: &Session) -> (String, Vec<(String, TensorElementType, AuxKind)>) {
    let mut speech_input = IN_SPEECH.to_owned();
    let mut aux_inputs: Vec<(String, TensorElementType, AuxKind)> = Vec::new();
    for outlet in session.inputs() {
        let ValueType::Tensor { ty, .. } = outlet.dtype() else {
            continue;
        };
        let name = outlet.name().to_owned();
        if matches!(ty, TensorElementType::Float32 | TensorElementType::Float16) {
            speech_input = name;
        } else {
            let kind = AuxKind::classify(&name);
            if kind == AuxKind::Unknown {
                log::warn!("model input {name:?} ({ty:?}) not recognized; feeding 0");
            }
            aux_inputs.push((name, *ty, kind));
        }
    }
    log::info!(
        "model inputs: {speech_input} (features) + {:?}",
        aux_inputs
            .iter()
            .map(|(n, t, k)| format!("{n}:{t:?}={k:?}"))
            .collect::<Vec<_>>()
    );
    (speech_input, aux_inputs)
}

/// ort 错误 → 端口错误（端口层不该依赖 ort 的类型）。
fn ort_err<E: std::fmt::Display>(e: E) -> AsrError {
    AsrError::Backend(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::SenseVoiceConfig;

    #[test]
    fn config_from_dir_prefers_int8_model() {
        let dir = std::env::temp_dir().join("cnt-asr-onnx-cfg");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let cfg = SenseVoiceConfig::from_dir(&dir);
        // 目录里没有 int8 → 退回 model.onnx
        assert!(cfg.model.ends_with("model.onnx"));
        assert!(cfg.tokens.ends_with("tokens.txt"));

        std::fs::write(dir.join("model.int8.onnx"), b"x").expect("write");
        let cfg = SenseVoiceConfig::from_dir(&dir);
        assert!(cfg.model.ends_with("model.int8.onnx"));
        std::fs::remove_file(dir.join("model.int8.onnx")).ok();
    }
}
