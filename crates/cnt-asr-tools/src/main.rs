//! cnt-asr-tools —— 语音链路的开发/诊断 CLI。
//!
//! 设计原则：**能不开麦克风就不开麦克风**。`transcribe`/`bench` 完全离线
//! （wav 文件进、文本出），是日常调模型与看性能的主路径；只有 `record`/`live`
//! 会打开麦克风，且会在 stderr 明确提示。
//!
//! ```text
//! cnt-asr-tools info      <model-dir>
//! cnt-asr-tools transcribe <model-dir> <a.wav> [b.wav ...] [--lang zh] [--no-itn] [--threads N]
//! cnt-asr-tools bench     <model-dir> <a.wav> [rounds]
//! cnt-asr-tools record    <out.wav> [seconds]              # 打开麦克风
//! cnt-asr-tools live      <model-dir> [seconds|--continuous] # 打开麦克风
//! ```

// CLI 里的数值转换（样本位深/声道数/秒）都是本质工作，不逐处加 allow。
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cnt_asr::{AsrError, Punctuator, Recognizer, SAMPLE_RATE};
use cnt_asr_onnx::punct::CtPunctuator;
use cnt_asr_onnx::{SenseVoice, SenseVoiceConfig};
use cnt_audio::{samples_to_secs, CaptureConfig, Recorder, Resampler};
use cnt_voice::{Mode, Voice, VoiceConfig, VoiceEvent};

/// CLI 错误。
#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("usage error: {0}")]
    Usage(String),
    #[error("asr: {0}")]
    Asr(#[from] AsrError),
    #[error("audio: {0}")]
    Audio(#[from] cnt_audio::AudioError),
    #[error("voice: {0}")]
    Voice(#[from] cnt_voice::VoiceError),
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

type CliResult = Result<(), CliError>;

fn main() {
    init_logging();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("info") => cmd_info(&args[1..]),
        Some("transcribe") => cmd_transcribe(&args[1..]),
        Some("bench") => cmd_bench(&args[1..]),
        Some("record") => cmd_record(&args[1..]),
        Some("live") => cmd_live(&args[1..]),
        _ => {
            usage();
            return;
        }
    };
    fastrace::flush();
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "cnt-asr-tools —— 语音链路 CLI\n\
         \n\
         离线（不开麦克风）：\n\
         \x20 info       <model-dir>\n\
         \x20 transcribe <model-dir> <a.wav> [b.wav ...] [--lang auto|zh|en|ja|ko|yue] [--no-itn] [--threads N]\n\
         \x20 bench      <model-dir> <a.wav> [rounds]\n\
         \n\
         需要麦克风（会明确提示）：\n\
         \x20 record     <out.wav> [seconds]\n\
         \x20 live       <model-dir> [seconds|--continuous]\n\
         \n\
         model-dir 里需要 model.int8.onnx（或 model.onnx）+ tokens.txt。"
    );
}

fn init_logging() {
    fastrace::set_reporter(
        fastrace::collector::ConsoleReporter,
        fastrace::collector::Config::default(),
    );
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.diagnostic(logforth::diagnostic::FastraceDiagnostic::default())
                .append(
                    logforth::append::Stderr::default()
                        .with_layout(logforth::layout::TextLayout::default()),
                )
        })
        .apply();
}

// ---------------------------------------------------------------------------
// 参数解析（够用即可，不引 clap）
// ---------------------------------------------------------------------------

/// 从参数里摘出 `--flag value` 与 `--flag`，剩下的是位置参数。
struct Args {
    positional: Vec<String>,
    lang: String,
    itn: bool,
    threads: usize,
    continuous: bool,
    /// 标点模型目录（默认与声学目录同级的 punct/）。
    punct: Option<String>,
    /// 关掉标点恢复。
    no_punct: bool,
}

fn parse_args(args: &[String]) -> Result<Args, CliError> {
    let mut out = Args {
        positional: Vec::new(),
        lang: "auto".to_owned(),
        itn: true,
        threads: 4,
        continuous: false,
        punct: None,
        no_punct: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--lang" => {
                out.lang.clone_from(
                    it.next()
                        .ok_or_else(|| CliError::Usage("--lang needs a value".into()))?,
                );
            }
            "--threads" => {
                out.threads = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| CliError::Usage("--threads needs a number".into()))?;
            }
            "--no-itn" => out.itn = false,
            "--no-punct" => out.no_punct = true,
            "--punct" => {
                out.punct = Some(
                    it.next()
                        .ok_or_else(|| CliError::Usage("--punct needs a directory".into()))?
                        .clone(),
                );
            }
            "--continuous" => out.continuous = true,
            other if other.starts_with("--") => {
                return Err(CliError::Usage(format!("unknown flag {other}")));
            }
            other => out.positional.push(other.to_owned()),
        }
    }
    Ok(out)
}

/// 加载标点模型：`--punct` 指定，否则找 `<声学目录>/../punct`。
/// 找不到就返回 None（标点是增强，不是必需）。
fn load_punct(acoustic_dir: &str, args: &Args) -> Option<Box<dyn Punctuator>> {
    if args.no_punct {
        return None;
    }
    let dir = args.punct.clone().map_or_else(
        || {
            Path::new(acoustic_dir)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("punct")
        },
        std::path::PathBuf::from,
    );
    let int8 = dir.join("model.int8.onnx");
    let model = if int8.exists() { int8 } else { dir.join("model.onnx") };
    if !model.exists() {
        eprintln!("（未找到标点模型 {}，输出将没有标点）", model.display());
        return None;
    }
    match CtPunctuator::open(&model, args.threads) {
        Ok(p) => Some(Box::new(p)),
        Err(e) => {
            eprintln!("（标点模型加载失败：{e}，输出将没有标点）");
            None
        }
    }
}

fn load_model(dir: &str, args: &Args) -> Result<SenseVoice, CliError> {
    let mut cfg = SenseVoiceConfig::from_dir(dir);
    cfg.language.clone_from(&args.lang);
    cfg.itn = args.itn;
    cfg.threads = args.threads;
    if !cfg.model.exists() {
        return Err(CliError::Usage(format!(
            "model not found: {} —— 先下载模型（见 README「语音输入」一节）",
            cfg.model.display()
        )));
    }
    let t = Instant::now();
    let model = SenseVoice::open(&cfg)?;
    println!("模型加载 {:?}：{}", t.elapsed(), model.name());
    Ok(model)
}

// ---------------------------------------------------------------------------
// 命令
// ---------------------------------------------------------------------------

fn cmd_info(args: &[String]) -> CliResult {
    let args = parse_args(args)?;
    let dir = args
        .positional
        .first()
        .ok_or_else(|| CliError::Usage("info <model-dir>".into()))?;
    let cfg = SenseVoiceConfig::from_dir(dir);
    if !cfg.model.exists() {
        return Err(CliError::Usage(format!(
            "model not found: {} —— 先跑 scripts/fetch-asr-model.sh",
            cfg.model.display()
        )));
    }

    // 先探查契约（不依赖任何假设），再尝试按 SenseVoice 契约加载
    let info = cnt_asr_onnx::inspect::inspect(&cfg.model)?;
    println!("模型文件 : {}", cfg.model.display());
    println!("输入：");
    for i in &info.inputs {
        println!("  {:<18} {:<10} {:?}", i.name, i.dtype, i.shape);
    }
    println!("输出：");
    for o in &info.outputs {
        println!("  {:<18} {:<10} {:?}", o.name, o.dtype, o.shape);
    }
    println!("metadata（{} 项）：", info.metadata.len());
    for (k, v) in &info.metadata {
        // CMVN 向量很长，只看前几个数确认存在与量级
        let shown: String = if v.len() > 80 {
            format!("{}… ({} 字节)", &v[..80], v.len())
        } else {
            v.clone()
        };
        println!("  {k:<20} = {shown}");
    }

    let tokens = cnt_asr_onnx::ctc::Tokens::load(&cfg.tokens)?;
    println!("词表     : {} 个 token（{}）", tokens.len(), cfg.tokens.display());
    println!("  id 0    = {:?}", tokens.get(0));
    println!("  末尾    = {:?}", tokens.get(tokens.len() - 1));
    println!("  blank   = {:?}", tokens.blank_id());
    println!("  编码    = {}", if tokens.is_base64() { "base64（字节级 BPE）" } else { "明文" });
    Ok(())
}

fn cmd_transcribe(args: &[String]) -> CliResult {
    let args = parse_args(args)?;
    let (dir, files) = args
        .positional
        .split_first()
        .ok_or_else(|| CliError::Usage("transcribe <model-dir> <a.wav> ...".into()))?;
    if files.is_empty() {
        return Err(CliError::Usage("需要至少一个 wav 文件".into()));
    }
    let model = load_model(dir, &args)?;
    let punct = load_punct(dir, &args);
    for file in files {
        let samples = read_wav(file)?;
        let secs = samples_to_secs(samples.len());
        let root = fastrace::Span::root(
            "cli_transcribe",
            fastrace::collector::SpanContext::random(),
        );
        let guard = root.set_local_parent();
        let t = Instant::now();
        let result = model.transcribe(&samples)?;
        let elapsed = t.elapsed();
        drop(guard);
        println!(
            "{file}: {:.2}s 音频，{:?}（RTF {:.3}）",
            secs,
            elapsed,
            elapsed.as_secs_f32() / secs.max(f32::EPSILON)
        );
        let text = punct.as_ref().map_or_else(
            || result.text.clone(),
            |p| {
                let t = Instant::now();
                let out = p.restore(&result.text).unwrap_or_else(|e| {
                    eprintln!("标点恢复失败（用原文）：{e}");
                    result.text.clone()
                });
                println!("  标点 {:?}（{}）", t.elapsed(), p.name());
                out
            },
        );
        println!("  → {text}");
        if let Some(lang) = &result.language {
            println!("  语言 {lang}，{} tokens", result.tokens.len());
        }
    }
    Ok(())
}

fn cmd_bench(args: &[String]) -> CliResult {
    let args = parse_args(args)?;
    let dir = args
        .positional
        .first()
        .ok_or_else(|| CliError::Usage("bench <model-dir> <a.wav> [rounds]".into()))?;
    let file = args
        .positional
        .get(1)
        .ok_or_else(|| CliError::Usage("bench 需要一个 wav 文件".into()))?;
    let rounds: usize = args
        .positional
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    // 聚合 reporter：逐条 Debug 输出在批量下不可读，阶段聚合表才能定位瓶颈
    let stats = Arc::new(Mutex::new(Aggregate::default()));
    fastrace::set_reporter(
        AggregateReporter {
            stats: Arc::clone(&stats),
        },
        fastrace::collector::Config::default(),
    );

    let model = load_model(dir, &args)?;
    let samples = read_wav(file)?;
    let audio_secs = samples_to_secs(samples.len());

    // 预热：第一次前向包含 ORT 的 arena 分配与权重换入，不该计入分位数
    for _ in 0..2 {
        model.transcribe(&samples)?;
    }

    let mut latencies = Vec::with_capacity(rounds);
    for i in 0..rounds {
        let root = fastrace::Span::root("bench", fastrace::collector::SpanContext::random());
        let guard = root.set_local_parent();
        let t = Instant::now();
        model.transcribe(&samples)?;
        latencies.push(t.elapsed());
        drop(guard);
        if i != 0 {
            root.cancel(); // 只保留第一轮的 span 树，避免聚合重复计数
        }
    }
    fastrace::flush();

    latencies.sort_unstable();
    let n = latencies.len();
    let avg = latencies.iter().sum::<Duration>() / u32::try_from(n).unwrap_or(1);
    println!();
    println!(
        "{file}: {audio_secs:.2}s 音频 × {n} 轮（{} 线程）",
        args.threads
    );
    println!(
        "  平均 {avg:?}（RTF {:.3}）  中位 {:?}  90% {:?}  最差 {:?}",
        avg.as_secs_f32() / audio_secs.max(f32::EPSILON),
        latencies[n / 2],
        latencies[n * 9 / 10],
        latencies[n - 1]
    );
    println!();
    println!("fastrace 阶段分解（首轮 span 树）：");
    let table = stats.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut names: Vec<&String> = table.by_name.keys().collect();
    names.sort_by_key(|n| stage_rank(n));
    for name in names {
        let a = &table.by_name[name];
        println!(
            "  {name:<20} n={:<3} avg={:>9.1}µs  min={:>9.1}µs  max={:>9.1}µs",
            a.count,
            ns_to_us(a.total_ns / a.count.max(1)),
            ns_to_us(a.min_ns),
            ns_to_us(a.max_ns)
        );
    }
    drop(table);
    Ok(())
}

fn cmd_record(args: &[String]) -> CliResult {
    let args = parse_args(args)?;
    let out = args
        .positional
        .first()
        .ok_or_else(|| CliError::Usage("record <out.wav> [seconds]".into()))?;
    let secs: f32 = args
        .positional
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);

    eprintln!("⚠ 即将打开麦克风录音 {secs:.1}s → {out}");
    let recorder = Recorder::spawn(&CaptureConfig::default())?;
    recorder.start()?;
    eprintln!("录音中……（设备：{}）", recorder.device_name());
    std::thread::sleep(Duration::from_secs_f32(secs));
    recorder.stop()?;
    let samples = recorder.drain();
    let (total, overflow) = recorder.stats();
    write_wav(out, &samples)?;
    eprintln!(
        "已写入 {out}：{:.2}s（采集 {total} 样本{}）",
        samples_to_secs(samples.len()),
        if overflow { "，发生过缓冲溢出" } else { "" }
    );
    Ok(())
}

fn cmd_live(args: &[String]) -> CliResult {
    let args = parse_args(args)?;
    let dir = args
        .positional
        .first()
        .ok_or_else(|| CliError::Usage("live <model-dir> [seconds|--continuous]".into()))?;
    let secs: f32 = args
        .positional
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(if args.continuous { 30.0 } else { 5.0 });
    let model = load_model(dir, &args)?;
    let mode = if args.continuous {
        Mode::Continuous
    } else {
        Mode::PushToTalk
    };

    // live 也走标点：和引擎里的链路一致，才能拿它当预演
    let punct: Arc<dyn Punctuator> = load_punct(dir, &args)
        .map_or_else(|| Arc::new(cnt_asr::NoPunct) as Arc<dyn Punctuator>, Arc::from);
    let voice = Voice::new(Arc::new(model), punct, VoiceConfig::default())?;
    eprintln!(
        "⚠ 即将打开麦克风（{}，{secs:.1}s；设备：{}）",
        mode.as_str(),
        voice.device()
    );

    // CLI 没有按键上下文，所以用「定时」模拟 PTT 的按下/松开
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut session = voice.start(mode)?;
        let stopper = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs_f32(secs)).await;
        });
        let deadline = tokio::time::Instant::now() + Duration::from_secs_f32(secs);
        loop {
            tokio::select! {
                event = session.next() => match event {
                    Some(VoiceEvent::Text(text)) => println!("→ {text}"),
                    Some(VoiceEvent::Empty) => println!("（没听到内容）"),
                    Some(VoiceEvent::Error(e)) => eprintln!("识别出错: {e}"),
                    Some(VoiceEvent::Recognizing) => eprintln!("识别中……"),
                    Some(VoiceEvent::Level { secs, db }) => {
                        eprint!("\r录音 {secs:5.1}s  {db:6.1} dBFS   ");
                    }
                    Some(VoiceEvent::Started(m)) => eprintln!("开始（{}）", m.as_str()),
                    Some(VoiceEvent::Stopped) | None => break,
                },
                () = tokio::time::sleep_until(deadline) => {
                    voice.stop()?;
                }
            }
        }
        stopper.abort();
        Ok::<(), CliError>(())
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// wav 读写
// ---------------------------------------------------------------------------

/// 读 wav → 16 kHz 单声道 f32（自动 downmix + 重采样）。
fn read_wav(path: impl AsRef<Path>) -> Result<Vec<f32>, CliError> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / f32::from(i16::MAX);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| shrink_to_i16(v, spec.bits_per_sample) * scale))
                .collect::<Result<_, _>>()?
        }
    };
    // downmix
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    let mut out = Vec::with_capacity(mono.len() * SAMPLE_RATE as usize / spec.sample_rate as usize);
    Resampler::new(spec.sample_rate, SAMPLE_RATE).process(&mono, &mut out);
    Ok(out)
}

/// 不同位深的整型样本统一折到 i16 量级。
fn shrink_to_i16(v: i32, bits: u16) -> f32 {
    let shift = i32::from(bits.saturating_sub(16));
    (v >> shift) as f32
}

/// 写 16 kHz 单声道 16-bit wav。
fn write_wav(path: impl AsRef<Path>, samples: &[f32]) -> Result<(), CliError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path.as_ref(), spec)?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// fastrace 聚合
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SpanAgg {
    count: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
}

#[derive(Default)]
struct Aggregate {
    by_name: std::collections::HashMap<String, SpanAgg>,
}

struct AggregateReporter {
    stats: Arc<Mutex<Aggregate>>,
}

impl fastrace::collector::Reporter for AggregateReporter {
    // 聚合锁的作用域就是整个 report（后台上报线程独占），不需要收紧
    #[allow(clippy::significant_drop_tightening)]
    fn report(&mut self, spans: Vec<fastrace::collector::SpanRecord>) {
        let mut st = self
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for s in spans {
            let e = st.by_name.entry(s.name.to_string()).or_default();
            e.count += 1;
            e.total_ns += s.duration_ns;
            e.max_ns = e.max_ns.max(s.duration_ns);
            e.min_ns = if e.min_ns == 0 || s.duration_ns < e.min_ns {
                s.duration_ns
            } else {
                e.min_ns
            };
        }
    }
}

/// 阶段在表里的排序（按链路顺序，读起来才像一条流水线）。
fn stage_rank(name: &str) -> usize {
    match name {
        "bench" | "cli_transcribe" => 0,
        "asr_transcribe" => 1,
        "asr_frontend" => 2,
        "fbank" => 3,
        "lfr" => 4,
        "cmvn" => 5,
        "asr_infer" => 6,
        "ctc_decode" => 7,
        _ => 8,
    }
}

/// `u64` 纳秒计数远小于 2^53，转 `f64` 无损。
#[allow(clippy::cast_precision_loss)]
fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1000.0
}
