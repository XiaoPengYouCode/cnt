//! cnt —— 一个用 Rust 写的简单简体中文拼音输入法（IBus 引擎）主程序。

mod observability;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use fastrace::Span;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use fastrace::local::LocalSpan;
use zbus::connection::Builder;
use zbus::fdo::RequestNameFlags;

use cnt_config::Config as AppConfig;
use cnt_decode::Decoder;
use cnt_dict::{DEFAULT_DICT_FILE, DEFAULT_USER_FILE, PinyinModel};
use cnt_engine::{Factory, VoiceAvailability, VoiceRuntime};
use cnt_lm::CntLm;
use cnt_store::StoreError;

const ENGINE_NAME: &str = "cnt";
const ENGINE_LONGNAME: &str = "Cnt 拼音 (Rust)";
const ENGINE_DESCRIPTION: &str = "一个简单的简体中文拼音输入法（Rust 实现）";
const COMPONENT_NAME: &str = "org.freedesktop.IBus.Cnt";
/// 默认语言模型文件名（用户数据目录下）。
const DEFAULT_LM_FILE: &str = "lm.cntl";
/// 默认语音模型目录（用户数据目录下）。
const DEFAULT_ASR_DIR: &str = "asr";
/// 默认标点模型目录（用户数据目录下）。
const DEFAULT_PUNCT_DIR: &str = "punct";

/// 主程序错误（显式，thiserror）。
#[derive(Debug, thiserror::Error)]
enum DaemonError {
    #[error("observability: {0}")]
    Observability(#[from] observability::ObservabilityError),
    #[error("usage: cnt-daemon [--standalone] (dynamic registration for development only)")]
    Arguments,
    #[error("ibus: {0}")]
    Ibus(#[from] cnt_ibus::IbusError),
    #[error("zbus: {0}")]
    Zbus(#[from] zbus::Error),
    #[error("zbus fdo: {0}")]
    Fdo(#[from] zbus::fdo::Error),
    #[error("store: {0}")]
    Store(#[from] StoreError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// 用户数据目录：`$XDG_DATA_HOME/cnt`（默认 `~/.local/share/cnt`）。
fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return PathBuf::from(dir).join("cnt");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".local/share/cnt");
    }
    PathBuf::from(".")
}

/// 词库路径：`CNT_DICT` 环境变量优先，否则默认用户数据目录。
fn dict_path() -> PathBuf {
    std::env::var_os("CNT_DICT").map_or_else(|| data_dir().join(DEFAULT_DICT_FILE), PathBuf::from)
}

/// 用户数据路径：`CNT_USER_DB` 环境变量优先，否则默认用户数据目录。
fn user_path() -> PathBuf {
    std::env::var_os("CNT_USER_DB")
        .map_or_else(|| data_dir().join(DEFAULT_USER_FILE), PathBuf::from)
}

/// 语言模型路径：`CNT_LM` 环境变量优先，否则默认用户数据目录。
fn lm_path() -> PathBuf {
    std::env::var_os("CNT_LM").map_or_else(|| data_dir().join(DEFAULT_LM_FILE), PathBuf::from)
}

type DecoderResources = (Arc<Decoder>, Arc<PinyinModel>, Option<Arc<CntLm>>);

/// 加载词库、用户数据与语言模型，构造解码器（返回解码器 + LM 供语音侧复用）。
///
/// 词库是硬依赖（缺失即退出）；语言模型缺失时降级为单字/词候选。
fn load_decoder(config: &AppConfig) -> Result<DecoderResources, DaemonError> {
    let _span = LocalSpan::enter_with_local_parent("load_decoder");
    // 词库 + 用户数据（mmap 二进制词库）
    let dict_path = dict_path();
    let user_path = user_path();
    let model = match PinyinModel::open(&dict_path, &user_path) {
        Ok(m) => m,
        Err(e) => {
            log::error!("cannot open dictionary at {}: {e}", dict_path.display());
            log::error!(
                "build it first:\n  cargo run -p cnt-dict-tools -- build data/wordlist.tsv {}",
                dict_path.display()
            );
            return Err(e.into());
        }
    };
    let model = Arc::new(model);
    log::info!("dictionary: {}", dict_path.display());
    log::info!("user data : {}", user_path.display());

    // 语言模型（可选：缺失时降级为单字/词候选，无整句）
    let lm_path = lm_path();
    let lm = match CntLm::open(&lm_path) {
        Ok(lm) => {
            log::info!(
                "lm: {} ({} words, {} bigrams)",
                lm_path.display(),
                lm.word_count(),
                lm.bigram_count()
            );
            Some(Arc::new(lm))
        }
        Err(e) => {
            log::warn!(
                "cannot open lm at {}: {e} (sentence candidates disabled; build it with: \n  cargo run -p cnt-dict-tools -- build-lm <lm.arpa> {})",
                lm_path.display(),
                lm_path.display()
            );
            None
        }
    };
    Ok((
        Arc::new(
            Decoder::new(model.clone(), lm.clone(), true)
                .with_typo_correction(config.typo_correction),
        ),
        model,
        lm,
    ))
}

/// 语音的 n-best 重排器：**复用拼音那份 LM 与用户词库**。
///
/// 这是本地方案独有的优势：通用声学模型不可能知道你把「工站」当常用词，
/// 而这份用户数据正是你自己一次次选出来的。两条链路共用同一把尺
/// （`cnt_score::policy::user`），同一份证据没有理由采信程度不同。
fn load_rescorer(
    lm: Option<&Arc<CntLm>>,
    model: &Arc<PinyinModel>,
) -> Option<Arc<dyn cnt_asr::TextScorer>> {
    let _span = LocalSpan::enter_with_local_parent("load_voice_rescorer");
    let lm = lm?;
    let user_words = model.user_words();
    let n_user = user_words.len();
    let scorer = cnt_asr_lm::LmTextScorer::with_user_words(Arc::clone(lm), user_words);
    log::info!(
        "voice rescorer: {} ({n_user} user words, {} lm words)",
        cnt_asr::TextScorer::name(&scorer),
        lm.word_count()
    );
    Some(Arc::new(scorer))
}

/// 加载语音输入（可选）。
///
/// **任何一步失败都只是「没有语音」，绝不能影响拼音输入**——模型没下载、
/// 没有麦克风、设备被占用都属于正常情况，输入法必须照常可用。
/// 加载标点模型（可选）。
///
/// 缺失/失败都只是「没有标点」——`NoPunct` 直通，语音仍然可用（只是难读）。
fn load_punctuator(settings: &cnt_config::VoiceSettings) -> Arc<dyn cnt_asr::Punctuator> {
    use cnt_asr_onnx::punct::CtPunctuator;

    let _span = LocalSpan::enter_with_local_parent("load_punctuator");
    if !settings.punctuation {
        log::info!("voice: punctuation disabled by config");
        return Arc::new(cnt_asr::NoPunct);
    }
    let dir = std::env::var_os("CNT_PUNCT_DIR").map_or_else(
        || {
            settings
                .punct_dir
                .clone()
                .unwrap_or_else(|| data_dir().join(DEFAULT_PUNCT_DIR))
        },
        PathBuf::from,
    );
    let int8 = dir.join("model.int8.onnx");
    let model = if int8.exists() {
        int8
    } else {
        dir.join("model.onnx")
    };
    if !model.exists() {
        log::warn!(
            "voice: punctuation model not found in {} (run scripts/fetch-asr-model.sh); \
             output will have no punctuation",
            dir.display()
        );
        return Arc::new(cnt_asr::NoPunct);
    }
    match CtPunctuator::open(&model, settings.threads) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            log::error!(
                "voice: cannot load punctuation model: {e}; output will have no punctuation"
            );
            Arc::new(cnt_asr::NoPunct)
        }
    }
}

fn load_voice(
    settings: &cnt_config::VoiceSettings,
    scorer: Option<Arc<dyn cnt_asr::TextScorer>>,
) -> Option<Arc<VoiceRuntime>> {
    use cnt_asr_onnx::{SenseVoice, SenseVoiceConfig};
    use cnt_audio::CaptureConfig;
    use cnt_audio::vad::VadConfig;
    use cnt_voice::{Voice, VoiceConfig};

    let _span = LocalSpan::enter_with_local_parent("load_voice");
    LocalSpan::add_property(|| ("enabled", settings.enabled.to_string()));
    LocalSpan::add_property(|| ("threads", settings.threads.to_string()));
    if !settings.enabled {
        log::info!("voice input: disabled (set [voice] enabled = true to turn on)");
        return None;
    }

    let dir = std::env::var_os("CNT_ASR_DIR").map_or_else(
        || {
            settings
                .model_dir
                .clone()
                .unwrap_or_else(|| data_dir().join(DEFAULT_ASR_DIR))
        },
        PathBuf::from,
    );
    let mut asr = SenseVoiceConfig::from_dir(&dir);
    asr.language.clone_from(&settings.language);
    asr.itn = settings.itn;
    asr.threads = settings.threads;
    if !asr.model.exists() || !asr.tokens.exists() {
        log::warn!(
            "voice input: model not found in {} (run scripts/fetch-asr-model.sh); voice disabled",
            dir.display()
        );
        return None;
    }

    let loaded = {
        let _span = LocalSpan::enter_with_local_parent("load_asr_model");
        SenseVoice::open(&asr)
    };
    let model = match loaded {
        Ok(m) => m,
        Err(e) => {
            log::error!("voice input: cannot load model: {e}; voice disabled");
            return None;
        }
    };
    log::info!("voice input: model loaded");

    let voice_config = VoiceConfig {
        capture: CaptureConfig {
            device: settings.device.clone(),
        },
        vad: VadConfig {
            margin_db: settings.vad_margin_db,
            trailing_silence_ms: settings.trailing_silence_ms,
            ..VadConfig::default()
        },
        max_seconds: settings.max_seconds,
        ..VoiceConfig::default()
    };
    let punctuator = load_punctuator(settings);
    match Voice::new(Arc::new(model), punctuator, scorer, voice_config) {
        Ok(voice) => Some(Arc::new(VoiceRuntime::new(
            voice,
            &settings.ptt_key,
            &settings.toggle_key,
        ))),
        Err(e) => {
            log::error!("voice input: cannot open microphone: {e}; voice disabled");
            None
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), DaemonError> {
    observability::init()?;
    let result = run().await;
    if let Err(error) = &result {
        log::error!("daemon stopped: {error}");
    }
    // 启动失败也要排空：名称冲突/连接失败的 trace 同样是诊断证据。
    fastrace::flush();
    log::logger().flush();
    result
}

async fn run() -> Result<(), DaemonError> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let standalone = match args.as_slice() {
        [] => false,
        [arg] if arg == "--standalone" => true,
        _ => return Err(DaemonError::Arguments),
    };
    // 独立 root 只量进程初始化到注册 IO 完成，不把整个常驻会话挂在它下面。
    let startup = Span::root("daemon_startup", SpanContext::random())
        .with_property(|| ("standalone", standalone.to_string()));
    let Started {
        conn,
        decoder,
        voice,
        voice_task,
    } = start(standalone).in_span(startup).await?;
    fastrace::flush();

    log::info!("running (Ctrl-C to quit)");
    let flush_decoder = decoder.clone();
    let flush_task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        tick.tick().await;
        loop {
            tick.tick().await;
            if let Err(e) = flush_decoder.flush_user() {
                log::error!("flush user data failed: {e}");
            }
        }
    });

    let shutdown = tokio::signal::ctrl_c().await;
    log::info!("received Ctrl-C, flushing user data");
    flush_task.abort();
    // 加载中退出：占住结果槽，防止后台完成后再发布；ONNX 初始化不可中断，等待回收。
    voice.complete(None);
    if let Some(runtime) = voice.runtime() {
        runtime.cancel().await;
    }
    if let Some(task) = voice_task
        && let Err(error) = task.await
    {
        log::error!("voice initialization task failed: {error}");
    }
    shutdown?;
    decoder.flush_user()?;
    // 保持连接直到用户数据已落盘；关闭连接会释放组件名称。
    drop(conn);
    Ok(())
}

struct Started {
    conn: zbus::Connection,
    decoder: Arc<Decoder>,
    voice: Arc<VoiceAvailability>,
    voice_task: Option<tokio::task::JoinHandle<()>>,
}

/// 模型 IO/ONNX 初始化运行在阻塞线程；完成后发布给包括旧上下文在内的所有引擎。
fn start_voice_loading(
    settings: cnt_config::VoiceSettings,
    model: Arc<PinyinModel>,
    lm: Option<Arc<CntLm>>,
    availability: Arc<VoiceAvailability>,
) -> Option<tokio::task::JoinHandle<()>> {
    if !settings.enabled {
        return None;
    }
    Some(tokio::spawn(async move {
        let result = tokio::task::spawn_blocking(move || {
            // 跨线程单独建 root，量到运行时发布完成，不与拼音就绪时间混在一起。
            let root = Span::root("voice_startup", SpanContext::random());
            let runtime = {
                let _parent = root.set_local_parent();
                let scorer = load_rescorer(lm.as_ref(), &model);
                load_voice(&settings, scorer)
            };
            (runtime, root)
        })
        .await;
        match result {
            Ok((runtime, root)) => {
                let ready = runtime.is_some();
                root.add_property(|| ("ready", ready.to_string()));
                if availability.complete(runtime) {
                    log::info!("voice initialization published: ready={ready}");
                } else {
                    log::info!("voice initialization discarded during shutdown");
                }
                drop(root);
            }
            Err(error) => {
                availability.complete(None);
                log::error!("voice initialization failed: {error}");
            }
        }
        fastrace::flush();
    }))
}

async fn start(standalone: bool) -> Result<Started, DaemonError> {
    // 先加载配置，解码器与语音初始化均使用同一份设置。
    let config_path =
        std::env::var_os("CNT_CONFIG").map_or_else(AppConfig::default_path, PathBuf::from);
    let config = match AppConfig::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            log::error!("cannot parse config at {}: {e}", config_path.display());
            AppConfig::default()
        }
    };
    log::info!(
        "config: {} (page_size={}, typo_correction={})",
        config_path.display(),
        config.page_size,
        config.typo_correction
    );

    let (decoder, model, lm) = load_decoder(&config)?;

    let voice = Arc::new(VoiceAvailability::new(
        config.voice.enabled,
        &config.voice.ptt_key,
        &config.voice.toggle_key,
    ));

    // 1. 找到 IBus 私有总线地址并连接
    let addr = cnt_ibus::find_address()?;
    log::info!("connecting to IBus: {addr}");

    // 异步阶段用 Span + in_span，确保连接与注册 IO 都归入启动树。
    let conn = Builder::address(addr.as_str())?
        .build()
        .in_span(Span::enter_with_local_parent("connect_ibus"))
        .await?;
    log::info!("connected (unique name: {:?})", conn.unique_name());

    // 2. 先提供 Factory，再申请名称；IBus 看到 NameOwnerChanged 后即可调用它。
    conn.object_server()
        .at(
            cnt_engine::FACTORY_OBJ_PATH,
            Factory::new(
                conn.clone(),
                decoder.clone(),
                config.page_size,
                voice.clone(),
            ),
        )
        .in_span(Span::enter_with_local_parent("serve_factory"))
        .await?;
    log::info!("factory served at {}", cnt_engine::FACTORY_OBJ_PATH);

    // XML 自动启动靠名称持有事件唤醒等待中的静态组件。
    // RegisterComponent 只创建另一个动态组件，无法完成这次静态激活握手。
    // 不抢占、不排队：重复进程应失败，避免两个进程同时写用户词库。
    conn.request_name_with_flags(COMPONENT_NAME, RequestNameFlags::DoNotQueue.into())
        .in_span(Span::enter_with_local_parent("request_component_name"))
        .await?;
    log::info!("component name acquired: {COMPONENT_NAME}");
    if standalone {
        register_standalone(&conn)
            .in_span(Span::enter_with_local_parent("register_standalone"))
            .await?;
    }
    let voice_task = start_voice_loading(config.voice, model, lm, voice.clone());
    Ok(Started {
        conn,
        decoder,
        voice,
        voice_task,
    })
}

/// 显式开发模式：允许没有 XML 安装记录时手动启动。
async fn register_standalone(conn: &zbus::Connection) -> Result<(), DaemonError> {
    let exec_path = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let engines = vec![cnt_ibus::engine_desc(
        ENGINE_NAME,
        ENGINE_LONGNAME,
        ENGINE_DESCRIPTION,
        "zh_CN", // language
        "MIT",   // license
        "cnt",   // author
        "",      // icon
        "us",    // layout
        50,      // rank
        "",      // hotkeys
        "",      // symbol
        "",      // setup
        "",      // layout_variant
        "",      // layout_option
        "0.1.0", // version
        "",      // textdomain
        "",      // icon_prop_key
    )];
    let component = cnt_ibus::component(
        COMPONENT_NAME,
        "Cnt Pinyin Component",
        "0.1.0",
        "MIT",
        "cnt",
        "",         // homepage
        &exec_path, // exec：本程序路径，ibus 需要时可重新拉起
        "",         // textdomain
        vec![],     // observed_paths
        engines,
    );

    conn.call_method(
        Some("org.freedesktop.IBus"),
        "/org/freedesktop/IBus",
        Some("org.freedesktop.IBus"),
        "RegisterComponent",
        &(component,),
    )
    .await?;
    log::info!("component registered: {ENGINE_NAME}");

    Ok(())
}
