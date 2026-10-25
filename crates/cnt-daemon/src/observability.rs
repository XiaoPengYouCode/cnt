//! `IBus` 可能丢弃 stderr，文件日志必须独立于启动进程的重定向。

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use fastrace::collector::{Config, ConsoleReporter};
use logforth::append::file::FileBuilder;
use logforth::filter::rustlog::RustLogFilterBuilder;

const LOG_NAME: &str = "cnt-daemon.log";

#[derive(Debug, thiserror::Error)]
pub enum ObservabilityError {
    #[error("set CNT_LOG_DIR, XDG_STATE_HOME or HOME to an absolute path")]
    MissingDirectory,
    #[error("cannot open log directory {path}: {source}")]
    File {
        path: PathBuf,
        source: logforth::Error,
    },
    #[error("cannot initialize logger: {0}")]
    Logger(#[from] logforth::Error),
}

fn log_dir() -> Result<PathBuf, ObservabilityError> {
    for (name, suffix) in [
        ("CNT_LOG_DIR", ""),
        ("XDG_STATE_HOME", "cnt"),
        ("HOME", ".local/state/cnt"),
    ] {
        if let Some(value) = std::env::var_os(name) {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                return Ok(path.join(suffix));
            }
        }
    }
    Err(ObservabilityError::MissingDirectory)
}

fn file_appender(dir: &Path) -> Result<logforth::append::File, ObservabilityError> {
    // 同步写入，无用户态队列；进程异常退出前的最后一条诊断也可读取。
    FileBuilder::new(dir, LOG_NAME)
        .layout(logforth::layout::TextLayout::default().no_color())
        .rollover_size(NonZeroUsize::new(5 * 1024 * 1024).expect("nonzero log size"))
        .max_log_files(NonZeroUsize::new(3).expect("nonzero log count"))
        .build()
        .map_err(|source| ObservabilityError::File {
            path: dir.to_path_buf(),
            source,
        })
}

pub fn init() -> Result<(), ObservabilityError> {
    let dir = log_dir()?;
    let file = file_appender(&dir)?;
    fastrace::set_reporter(ConsoleReporter, Config::default());
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(RustLogFilterBuilder::from_default_env_or("info,cnt_engine=debug").build())
                .diagnostic(logforth::diagnostic::FastraceDiagnostic::default())
                .append(file)
                .append(
                    logforth::append::Stderr::default()
                        .with_layout(logforth::layout::TextLayout::default()),
                )
        })
        .dispatch(|d| d.append(logforth::append::FastraceEvent::default()))
        .try_apply()?;
    log::info!(
        "observability: pid={} executable={:?} log={} RUST_LOG={:?}",
        std::process::id(),
        std::env::current_exe(),
        dir.join(LOG_NAME).display(),
        std::env::var_os("RUST_LOG")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;

    #[test]
    fn file_log_is_visible_immediately_and_survives_reopen() {
        let dir = std::env::temp_dir().join(format!("cnt-log-test-{}", std::process::id()));
        for message in ["first key received", "second key received"] {
            // 使用独立 logger；测试不初始化 daemon、不连接 IBus。
            let logger = logforth::bridge::log::LogBridge::new(
                logforth::core::builder()
                    .dispatch(|d| d.append(file_appender(&dir).unwrap()))
                    .build(),
            );
            logger.log(
                &log::Record::builder()
                    .level(log::Level::Info)
                    .target("cnt_engine")
                    .args(format_args!("{message}"))
                    .build(),
            );
            // 不依赖 flush/drop 才可见，也不依赖 stderr。
            let text = std::fs::read_to_string(dir.join(LOG_NAME)).unwrap();
            assert!(text.contains(message));
            assert!(text.contains("first key received"));
            assert!(!text.contains('\u{1b}'));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
