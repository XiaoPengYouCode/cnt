//! cnt-asr —— 语音领域的**端口层**（零依赖，仿 `cnt-score` 的做法）。
//!
//! ## 领域划分
//!
//! 语音输入这条链上有四件事，各自变化速率完全不同，所以拆成四个 crate：
//!
//! ```text
//!   采集(cnt-audio) ─► 识别(端口) ─► 标点(端口) ─► 编排(cnt-voice) ─► 上屏(cnt-engine)
//!                        ▲             ▲
//!                        └─ cnt-asr-onnx 实现两者（ort/CPU，将来 NPU）
//! ```
//!
//! 本 crate 只定义**契约**，不含任何 IO 与模型代码：
//!
//! | 端口 | 一次调用的粒度 | 频次 | 分派 |
//! |---|---|---|---|
//! | [`Recognizer`] | 一整段语音（0.3~60 s） | 每句 1 次 | `dyn` |
//! | [`Punctuator`] | 一句文本（几十字） | 每句 ≤1 次 | `dyn` |
//!
//! 两个端口都用 `dyn`：调用频次是「每句一次」，虚表开销完全可忽略，
//! 换来的是**运行期按配置装卸**（没装标点模型就退化成 [`NoPunct`]，
//! 声学模型自带标点时也用 [`NoPunct`]）。
//!
//! ## 为什么标点是独立端口，而不是识别器的内部细节
//!
//! 实测发现：`Fun-ASR-Nano` 的 CTC 导出词表里**有**标点 token，但训练目标无标点，
//! 输出是光板汉字流；`SenseVoice-Small` 自带标点；`FireRedASR2` 全家词表就没有标点。
//! 也就是说「会不会标点」是**声学模型的偶然属性**，而「上屏文本必须有标点」是
//! 输入法的**固有需求**。把它编码成独立端口，两边就能各自演进：
//! 换声学模型不用重做标点，换标点模型不用碰声学。
//! 这也是 `FunASR` 官方 pipeline 的做法（Paraformer + ct-punc 两段）。
//!
//! 另外本 crate 承担**纯文本后处理**（[`text`]：字节级 BPE 拼装、中英混排空格与
//! 标点归一），它与模型无关、可单测，不该藏在后端里。

pub mod text;

pub use text::{assemble, assemble_bytes, polish};

/// 识别端口统一采样率（16 kHz 单声道 f32，取值范围 `[-1, 1]`）。
pub const SAMPLE_RATE: u32 = 16_000;

/// 语音识别错误。
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    /// 模型/词表文件打不开或格式不对。
    #[error("model: {0}")]
    Model(String),
    /// 推理后端报错（ort / EP）。
    #[error("backend: {0}")]
    Backend(String),
    /// 输入音频不合法（太短、采样率不符等）。
    #[error("audio: {0}")]
    Audio(String),
    /// 配置项不合法。
    #[error("config: {0}")]
    Config(String),
}

/// 一条候选假设：文本 + 声学对数概率（自然对数，越大越好）。
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub text: String,
    /// 声学模型给的对数概率（**自然对数**，CTC 口径）。
    pub acoustic: f32,
}

/// 一次识别的结果。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    /// 后处理完成、可直接上屏的文本。
    pub text: String,
    /// 原始 token 序列（含 `<|zh|>` 一类特殊 token），用于诊断。
    pub tokens: Vec<String>,
    /// 模型识别出的语言标签（如 `zh`/`en`），未知时为 None。
    pub language: Option<String>,
    /// n-best（含 #1，按声学分数降序）。
    ///
    /// 语音识别的主要错误是**音对字错**（`瓶颈`→`平境`、`语音`→`原音`），
    /// 正确答案常常就在第 2、3 名里。只有把备选交出来，语言模型才有翻盘的机会；
    /// 只给 1-best 等于把纠错的可能性提前丢掉。
    pub alternatives: Vec<Hypothesis>,
}

impl Transcript {
    /// 空结果（静音 / 全是特殊 token）。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// 文本打分端口：给一段文本一个语言模型对数概率（log10）。
///
/// 用途是**在 n-best 之间比较**，不是绝对概率。声学模型只听得见「音」，
/// 分不清同音异形；而这条端口背后是中文的 n-gram 统计（`cnt-lm` 的 27 万 unigram
/// + 480 万 bigram），恰好补上这一块。
///
/// 实现要遵守：
/// - **单位是 log10**（与 `cnt-lm`/`cnt-score` 一致；CTC 的自然对数由调用方换算）；
/// - 打不了分（模型没装、文本为空）时返回 `None`，调用方保持声学顺序。
pub trait TextScorer: Send + Sync {
    /// 文本的语言模型对数概率（log10）；无法打分时 None。
    fn logp10(&self, text: &str) -> Option<f32>;

    /// 后端名字（日志/诊断用）。
    fn name(&self) -> &str;
}

/// 标点恢复端口：光板文本 → 带标点文本。
///
/// 契约要求（实现必须遵守，编排层依赖这些性质）：
///
/// - **只加标点，不改字**：实现不得增删或替换汉字/单词，否则用户会看到
///   「我说的不是这个」——那比没有标点更糟；
/// - **失败即原样返回**：标点是增强而非必需，模型出错时必须退回入参，
///   绝不能让一句话丢掉（[`NoPunct`] 就是这个语义的平凡实现）；
/// - 幂等友好：输入已带标点时不应重复添加。
pub trait Punctuator: Send + Sync {
    /// 给一段文本加标点。
    ///
    /// # Errors
    /// 推理失败时返回错误；调用方应当**忽略错误并使用原文**。
    fn restore(&self, text: &str) -> Result<String, AsrError>;

    /// 后端名字（日志/诊断用）。
    fn name(&self) -> &str;
}

/// 直通实现：不加标点（未装标点模型，或声学模型自带标点时用）。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPunct;

impl Punctuator for NoPunct {
    fn restore(&self, text: &str) -> Result<String, AsrError> {
        Ok(text.to_owned())
    }

    fn name(&self) -> &'static str {
        "none"
    }
}

/// 识别端口：一段完整音频 → 文本。
///
/// 实现必须是 `Send + Sync`：编排层在工作线程里调用，引擎线程只等结果。
/// 内部若有可变状态（ONNX session 需要 `&mut`），实现自己用锁包起来。
pub trait Recognizer: Send + Sync {
    /// 识别一段 16 kHz 单声道音频。
    ///
    /// # Errors
    /// 音频不合法或推理失败时返回错误。
    fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError>;

    /// 后端名字（日志/诊断用）。
    fn name(&self) -> &str;

    /// 后端期望的采样率（默认 16 kHz）。
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}
