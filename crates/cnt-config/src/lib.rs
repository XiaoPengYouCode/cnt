//! cnt-config —— 配置加载（TOML）。
//!
//! 配置文件：`$XDG_CONFIG_HOME/cnt/config.toml`（默认 `~/.config/cnt/config.toml`）：
//!
//! ```toml
//! # 每页候选词数
//! page_size = 10
//!
//! # 语音输入（默认关闭：模型不随程序分发）
//! [voice]
//! enabled = true
//! model_dir = "~/.local/share/cnt/asr"
//! language = "auto"          # auto/zh/en/ja/ko/yue
//! itn = true                 # 数字规整 + 标点
//! threads = 4
//! device = ""                # 麦克风设备名子串，空 = 系统默认
//! punctuation = true         # 标点恢复（独立的小模型，见 cnt-asr 的 Punctuator 端口）
//! punct_dir = "~/.local/share/cnt/punct"
//! ptt_key = "Alt_R"                    # 按住说话（右 Alt）
//! toggle_key = "Control+Shift+space"   # 切换式常开
//! max_seconds = 60.0
//! trailing_silence_ms = 700  # 常开模式：停顿多久算一句结束
//! vad_margin_db = 10.0       # 高于噪声底多少 dB 算语音
//! ```
//!
//! 解析子集而不是结构体反序列化：配置项少、且希望**单项出错不影响
//! 其他项**（输入法不能因为一个写错的热键就不能打字）。

use std::fs;
use std::path::{Path, PathBuf};

/// 默认每页候选数。
pub const DEFAULT_PAGE_SIZE: usize = 10;
/// 候选词数允许范围（上限 16：`ibus_lookup_table_new` 断言 `page_size <= 16`）。
const PAGE_SIZE_RANGE: std::ops::RangeInclusive<usize> = 5..=16;

/// 应用配置。
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// 每页候选词数。
    pub page_size: usize,
    /// 语音输入。
    pub voice: VoiceSettings,
}

/// 语音输入配置。
///
/// 默认 `enabled = false`：模型是两百多 MB 的外部文件，没装就不应该因为
/// “想试试语音”而影响拼音输入的启动。
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceSettings {
    /// 是否启用语音输入。
    pub enabled: bool,
    /// 模型目录（含 `model.int8.onnx` / `model.onnx` 与 `tokens.txt`）。
    pub model_dir: Option<PathBuf>,
    /// 识别语言：`auto`/`zh`/`en`/`ja`/`ko`/`yue`。
    pub language: String,
    /// 是否开启 ITN（数字/日期规整 + 标点）。
    pub itn: bool,
    /// 推理线程数。
    pub threads: usize,
    /// 麦克风设备名子串（None = 系统默认）。
    pub device: Option<String>,
    /// 是否开启标点恢复（声学模型输出光板文本，标点是独立一段推理）。
    pub punctuation: bool,
    /// 标点模型目录（None = 声学目录的同级 `punct/`）。
    pub punct_dir: Option<PathBuf>,
    /// 按住说话的键。
    pub ptt_key: String,
    /// 切换式常开的组合键。
    pub toggle_key: String,
    /// PTT 单次最长录音（秒）。
    pub max_seconds: f32,
    /// 常开模式：多长的静音算一句结束（毫秒）。
    pub trailing_silence_ms: usize,
    /// VAD 相对噪声底的余量（dB）。
    pub vad_margin_db: f32,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model_dir: None,
            language: "auto".to_owned(),
            itn: true,
            threads: 4,
            device: None,
            punctuation: true,
            punct_dir: None,
            // 右 Alt：按住不放对应用基本无副作用，是 push-to-talk 的理想人选。
            // （不用右 Ctrl：很多新键盘已经把它换成 Copilot 键了）
            ptt_key: "Alt_R".to_owned(),
            toggle_key: "Control+Shift+space".to_owned(),
            max_seconds: 60.0,
            trailing_silence_ms: 700,
            vad_margin_db: 10.0,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            voice: VoiceSettings::default(),
        }
    }
}

impl Config {
    /// 默认配置路径：`$XDG_CONFIG_HOME/cnt/config.toml`（默认 `~/.config/cnt/config.toml`）。
    #[must_use]
    pub fn default_path() -> PathBuf {
        if let Ok(dir) = std::env::var("XDG_CONFIG_HOME")
            && !dir.is_empty()
        {
            return PathBuf::from(dir).join("cnt/config.toml");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".config/cnt/config.toml");
        }
        PathBuf::from("config.toml")
    }

    /// 从路径加载配置；文件不存在时返回默认值，解析失败返回错误。
    ///
    /// # Errors
    /// 文件存在但不是合法 TOML 时返回解析错误。
    pub fn load(path: impl AsRef<Path>) -> Result<Self, toml::de::Error> {
        let path = path.as_ref();
        let Ok(content) = fs::read_to_string(path) else {
            return Ok(Self::default()); // 文件不存在 → 默认配置
        };
        let mut config = Self::default();
        // toml::Value::from_str 解析的是「单个值」，文档要用反序列化入口
        let value: toml::Value = toml::from_str(&content)?;
        if let Some(v) = value.get("page_size")
            && let Some(n) = v.as_integer()
        {
            config.page_size = clamp_page_size(n);
        }
        if let Some(voice) = value.get("voice") {
            config.voice.merge_toml(voice);
        }
        Ok(config)
    }

    /// 解析 `page_size` 字段（TOML 整数 → 夹紧到允许范围）。
    #[must_use]
    pub fn with_page_size(n: i64) -> Self {
        Self {
            page_size: clamp_page_size(n),
            ..Self::default()
        }
    }
}

impl VoiceSettings {
    /// 用 TOML 的 `[voice]` 表覆盖默认值（未识别/类型不对的项保持默认）。
    ///
    /// 配置里的数值都是「秒/分贝/线程数」这类小量，f64 → f32 的精度损失
    /// 与 i64 → f64 的截断在这里都没有实际意义（clamp 会先把范围收住）。
    #[allow(clippy::cast_possible_truncation)]
    fn merge_toml(&mut self, table: &toml::Value) {
        if let Some(v) = table.get("enabled").and_then(toml::Value::as_bool) {
            self.enabled = v;
        }
        if let Some(v) = table.get("model_dir").and_then(toml::Value::as_str) {
            self.model_dir = Some(expand_tilde(v));
        }
        if let Some(v) = table.get("language").and_then(toml::Value::as_str) {
            v.clone_into(&mut self.language);
        }
        if let Some(v) = table.get("itn").and_then(toml::Value::as_bool) {
            self.itn = v;
        }
        if let Some(v) = table.get("threads").and_then(toml::Value::as_integer) {
            self.threads = usize::try_from(v.clamp(0, 64)).unwrap_or(4);
        }
        if let Some(v) = table.get("device").and_then(toml::Value::as_str) {
            self.device = (!v.is_empty()).then(|| v.to_owned());
        }
        if let Some(v) = table.get("punctuation").and_then(toml::Value::as_bool) {
            self.punctuation = v;
        }
        if let Some(v) = table.get("punct_dir").and_then(toml::Value::as_str) {
            self.punct_dir = Some(expand_tilde(v));
        }
        if let Some(v) = table.get("ptt_key").and_then(toml::Value::as_str) {
            v.clone_into(&mut self.ptt_key);
        }
        if let Some(v) = table.get("toggle_key").and_then(toml::Value::as_str) {
            v.clone_into(&mut self.toggle_key);
        }
        if let Some(v) = number(table.get("max_seconds")) {
            self.max_seconds = (v as f32).clamp(1.0, 600.0);
        }
        if let Some(v) = table
            .get("trailing_silence_ms")
            .and_then(toml::Value::as_integer)
        {
            self.trailing_silence_ms = usize::try_from(v.clamp(100, 5_000)).unwrap_or(700);
        }
        if let Some(v) = number(table.get("vad_margin_db")) {
            self.vad_margin_db = (v as f32).clamp(1.0, 40.0);
        }
    }
}

/// TOML 里的数字（浮点或整数都接：用户写 `max_seconds = 60` 不应该被忽略）。
#[allow(clippy::cast_precision_loss)]
fn number(v: Option<&toml::Value>) -> Option<f64> {
    let v = v?;
    v.as_float()
        .or_else(|| v.as_integer().map(|i| i as f64))
}

/// 展开路径开头的 `~`（配置文件里写 `~/...` 是人的习惯，不能当相对路径）。
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
}

fn clamp_page_size(n: i64) -> usize {
    let n = n.clamp(
        i64::from(u32::try_from(*PAGE_SIZE_RANGE.start()).unwrap_or(5)),
        i64::from(u32::try_from(*PAGE_SIZE_RANGE.end()).unwrap_or(30)),
    );
    usize::try_from(n).unwrap_or(DEFAULT_PAGE_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_when_file_missing() {
        let cfg = Config::load("/nonexistent/cnt-config.toml").unwrap();
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn parses_page_size() {
        let dir = std::env::temp_dir().join(format!("cnt-config-{}", std::process::id()));
        let path = dir.join("config.toml");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, "page_size = 8\n").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.page_size, 8);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn clamps_out_of_range() {
        assert_eq!(Config::with_page_size(3).page_size, 5);
        assert_eq!(Config::with_page_size(999).page_size, 16);
        assert_eq!(Config::with_page_size(0).page_size, 5);
    }

    #[test]
    fn invalid_toml_errors() {
        let dir = std::env::temp_dir().join(format!("cnt-config-bad-{}", std::process::id()));
        let path = dir.join("config.toml");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, "page_size = [unclosed").unwrap();
        assert!(Config::load(&path).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn voice_is_disabled_by_default() {
        let v = VoiceSettings::default();
        assert!(!v.enabled);
        // 标点默认开：声学模型输出无标点，关了就等于不可用
        assert!(v.punctuation);
        assert_eq!(v.ptt_key, "Alt_R");
        assert_eq!(v.toggle_key, "Control+Shift+space");
    }

    #[test]
    fn parses_voice_section() {
        let dir = std::env::temp_dir().join(format!("cnt-config-voice-{}", std::process::id()));
        let path = dir.join("config.toml");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            "page_size = 7\n\
             [voice]\n\
             enabled = true\n\
             model_dir = \"/tmp/asr\"\n\
             language = \"zh\"\n\
             threads = 8\n\
             ptt_key = \"Alt_R\"\n\
             max_seconds = 30\n\
             vad_margin_db = 12.5\n",
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.page_size, 7);
        assert!(cfg.voice.enabled);
        assert_eq!(cfg.voice.model_dir.as_deref(), Some(Path::new("/tmp/asr")));
        assert_eq!(cfg.voice.language, "zh");
        assert_eq!(cfg.voice.threads, 8);
        assert_eq!(cfg.voice.ptt_key, "Alt_R");
        // 整数写法也该被接受
        assert!((cfg.voice.max_seconds - 30.0).abs() < f32::EPSILON);
        assert!((cfg.voice.vad_margin_db - 12.5).abs() < f32::EPSILON);
        // 未写的项保持默认
        assert!(cfg.voice.itn);
        assert_eq!(cfg.voice.trailing_silence_ms, 700);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_voice_values_fall_back_to_defaults() {
        let dir = std::env::temp_dir().join(format!("cnt-config-vbad-{}", std::process::id()));
        let path = dir.join("config.toml");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // 类型全写错 + 越界值：不能报错，也不能让输入法拿到不合法参数
        fs::write(
            &path,
            "[voice]\nenabled = \"yes\"\nthreads = 999\nmax_seconds = -5\n",
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        assert!(!cfg.voice.enabled);
        assert_eq!(cfg.voice.threads, 64);
        assert!((cfg.voice.max_seconds - 1.0).abs() < f32::EPSILON);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tilde_in_model_dir_is_expanded() {
        let home = std::env::var("HOME").unwrap_or_default();
        let expanded = super::expand_tilde("~/asr");
        assert!(expanded.starts_with(&home));
        assert!(expanded.ends_with("asr"));
        // 绝对路径不动
        assert_eq!(super::expand_tilde("/tmp/x"), PathBuf::from("/tmp/x"));
    }
}
