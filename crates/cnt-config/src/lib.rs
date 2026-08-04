//! cnt-config —— 配置加载（TOML）。
//!
//! 配置文件：`$XDG_CONFIG_HOME/cnt/config.toml`（默认 `~/.config/cnt/config.toml`），
//! 当前仅一项配置：
//! ```toml
//! # 每页候选词数
//! page_size = 10
//! ```

use std::fs;
use std::path::{Path, PathBuf};

/// 默认每页候选数。
pub const DEFAULT_PAGE_SIZE: usize = 10;
/// 候选词数允许范围（上限 16：`ibus_lookup_table_new` 断言 `page_size <= 16`）。
const PAGE_SIZE_RANGE: std::ops::RangeInclusive<usize> = 5..=16;

/// 应用配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// 每页候选词数。
    pub page_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
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
        Ok(config)
    }

    /// 解析 `page_size` 字段（TOML 整数 → 夹紧到允许范围）。
    #[must_use]
    pub fn with_page_size(n: i64) -> Self {
        Self {
            page_size: clamp_page_size(n),
        }
    }
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
}
