//! cnt-audio —— 麦克风采集 + 重采样 + 语音活动检测（VAD）。
//!
//! 三件事，一条链：
//!
//! ```text
//!   cpal 回调（设备原生格式，48k 多声道 i16/f32）
//!     └─► downmix 单声道 ─► [`resample`] 重采样到 16k ─► 环形缓冲
//!                                                          └─► [`vad`] 切句（常开模式）
//! ```
//!
//! **隐私优先**：只在会话期间打开设备（PTT 按下 / 常开开启），会话结束立刻
//! `drop(stream)` 关闭；音频只在内存里流动，除显式 `record` 命令外不落盘。
//!
//! **线程模型**：cpal 的 `Stream` 在部分平台不是 `Send`，所以设备的打开/关闭
//! 全部发生在 [`Recorder`] 自己的采集线程内，外部只通过命令通道驱动。

// 采样数 ↔ 秒的换算贯穿全 crate，转换是本质工作。
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

pub mod resample;
pub mod vad;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, StreamConfig, SupportedStreamConfig};

use cnt_asr::SAMPLE_RATE;

pub use resample::Resampler;
pub use vad::{Segmenter, VadConfig};

/// 采集缓冲上限（秒）：超过就丢最老的数据，防止「忘了关」把内存打满。
const MAX_BUFFER_SECS: usize = 120;
/// 采集缓冲上限（样本数）。
const MAX_BUFFER_SAMPLES: usize = MAX_BUFFER_SECS * SAMPLE_RATE as usize;

/// 音频子系统错误。
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// 找不到输入设备（没有麦克风 / 权限被拒）。
    #[error("no input device available")]
    NoDevice,
    /// cpal 报错。
    #[error("cpal: {0}")]
    Cpal(String),
    /// 设备的采样格式我们不支持。
    #[error("unsupported sample format: {0}")]
    Format(String),
    /// 采集线程已退出。
    #[error("capture thread is gone")]
    ThreadGone,
}

impl From<cpal::Error> for AudioError {
    fn from(e: cpal::Error) -> Self {
        Self::Cpal(e.to_string())
    }
}

/// 采集配置。
#[derive(Debug, Clone, Default)]
pub struct CaptureConfig {
    /// 指定输入设备名（子串匹配）；None = 系统默认设备。
    pub device: Option<String>,
}

/// 采集线程的共享状态（回调线程写、控制线程读）。
#[derive(Debug, Default)]
struct Shared {
    /// 已重采样为 16k 单声道的样本。
    buffer: Mutex<Vec<f32>>,
    /// 累计采集样本数（诊断用，不随 drain 归零）。
    total: AtomicUsize,
    /// 是否发生过缓冲溢出（丢数据）。
    overflow: AtomicBool,
    /// 采集是否正在进行。
    active: AtomicBool,
}

impl Shared {
    fn lock_buffer(&self) -> std::sync::MutexGuard<'_, Vec<f32>> {
        self.buffer.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// 回调线程：追加样本（超上限丢最老的）。
    fn push(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.total.fetch_add(samples.len(), Ordering::Relaxed);
        let overflowed = {
            let mut buf = self.lock_buffer();
            buf.extend_from_slice(samples);
            let over = buf.len() > MAX_BUFFER_SAMPLES;
            if over {
                let excess = buf.len() - MAX_BUFFER_SAMPLES;
                buf.drain(..excess);
            }
            over
        };
        if overflowed {
            self.overflow.store(true, Ordering::Relaxed);
        }
    }
}

/// 采集线程命令。
enum Cmd {
    /// 打开设备开始采集（清空缓冲）。
    Start,
    /// 关闭设备停止采集（缓冲保留，供 drain）。
    Stop,
    /// 退出线程。
    Quit,
}

/// 麦克风采集器：会话式开关，输出 16 kHz 单声道 f32。
///
/// `start`/`stop` 是同步的（等采集线程确认），因为「按下说话」的语义要求
/// 松手时确定已经拿到全部音频，不能有竞态。
pub struct Recorder {
    cmd: SyncSender<(Cmd, SyncSender<Result<(), AudioError>>)>,
    shared: Arc<Shared>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// 设备描述（日志展示用）。
    device_name: String,
}

impl Recorder {
    /// 启动采集线程（不打开设备）。
    ///
    /// 会先探测一次设备与格式：**启动即失败**总比 PTT 按下时才发现没麦克风好。
    ///
    /// # Errors
    /// 没有可用输入设备或设备格式不支持时返回错误。
    pub fn spawn(config: &CaptureConfig) -> Result<Self, AudioError> {
        let device_name = {
            let (device, cfg) = open_device(config)?;
            let name = describe(&device);
            log::info!(
                "audio input: {name} ({} ch, {} Hz, {:?})",
                cfg.channels(),
                cfg.sample_rate(),
                cfg.sample_format()
            );
            name
        };

        let shared = Arc::new(Shared::default());
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        let thread_shared = Arc::clone(&shared);
        let thread_config = config.clone();
        let handle = std::thread::Builder::new()
            .name("cnt-audio".into())
            .spawn(move || capture_loop(&thread_config, &thread_shared, &rx))
            .map_err(|e| AudioError::Cpal(e.to_string()))?;

        Ok(Self {
            cmd: tx,
            shared,
            handle: Some(handle),
            device_name,
        })
    }

    /// 设备名（日志用）。
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// 打开设备开始采集（清空历史缓冲）。
    ///
    /// # Errors
    /// 设备打不开或采集线程已退出时返回错误。
    pub fn start(&self) -> Result<(), AudioError> {
        self.request(Cmd::Start)
    }

    /// 停止采集并关闭设备（缓冲内容保留，用 [`Recorder::drain`] 取）。
    ///
    /// # Errors
    /// 采集线程已退出时返回错误。
    pub fn stop(&self) -> Result<(), AudioError> {
        self.request(Cmd::Stop)
    }

    /// 是否正在采集。
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.shared.active.load(Ordering::Relaxed)
    }

    /// 取走当前缓冲的全部样本（16 kHz 单声道）。
    #[must_use]
    pub fn drain(&self) -> Vec<f32> {
        std::mem::take(&mut *self.shared.lock_buffer())
    }

    /// 当前缓冲的样本数（不取走）。
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.shared.lock_buffer().len()
    }

    /// 累计采集样本数 / 是否发生过溢出（诊断）。
    #[must_use]
    pub fn stats(&self) -> (usize, bool) {
        (
            self.shared.total.load(Ordering::Relaxed),
            self.shared.overflow.load(Ordering::Relaxed),
        )
    }

    fn request(&self, cmd: Cmd) -> Result<(), AudioError> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.cmd
            .send((cmd, tx))
            .map_err(|_| AudioError::ThreadGone)?;
        rx.recv_timeout(Duration::from_secs(5))
            .map_err(|_| AudioError::ThreadGone)?
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.request(Cmd::Quit);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// 采集线程主循环：设备只在这个线程里打开/关闭（`Stream` 不跨线程）。
fn capture_loop(
    config: &CaptureConfig,
    shared: &Arc<Shared>,
    rx: &Receiver<(Cmd, SyncSender<Result<(), AudioError>>)>,
) {
    let mut stream: Option<cpal::Stream> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok((Cmd::Start, reply)) => {
                shared.lock_buffer().clear();
                let result = build_stream(config, shared).map(|s| {
                    stream = Some(s);
                    shared.active.store(true, Ordering::Relaxed);
                });
                let _ = reply.send(result);
            }
            Ok((Cmd::Stop, reply)) => {
                drop(stream.take()); // 关闭设备：麦克风指示灯灭
                shared.active.store(false, Ordering::Relaxed);
                let _ = reply.send(Ok(()));
            }
            Ok((Cmd::Quit, reply)) => {
                drop(stream.take());
                shared.active.store(false, Ordering::Relaxed);
                let _ = reply.send(Ok(()));
                return;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                drop(stream.take());
                shared.active.store(false, Ordering::Relaxed);
                return;
            }
        }
    }
}

/// 选设备 + 选配置：优先能直接给 16 kHz 的配置（省一次重采样）。
fn open_device(config: &CaptureConfig) -> Result<(cpal::Device, SupportedStreamConfig), AudioError> {
    let host = cpal::default_host();
    let device = match &config.device {
        Some(want) => host
            .input_devices()
            .map_err(AudioError::from)?
            .find(|d| describe(d).to_lowercase().contains(&want.to_lowercase()))
            .ok_or(AudioError::NoDevice)?,
        None => host.default_input_device().ok_or(AudioError::NoDevice)?,
    };
    let default = device.default_input_config().map_err(AudioError::from)?;
    let chosen = pick_16k(&device).unwrap_or(default);
    Ok((device, chosen))
}

/// 设备是否支持直接吐 16 kHz？支持就用（f32/i16 优先）。
fn pick_16k(device: &cpal::Device) -> Option<SupportedStreamConfig> {
    let ranges = device.supported_input_configs().ok()?;
    let mut best: Option<SupportedStreamConfig> = None;
    for range in ranges {
        if range.min_sample_rate() > SAMPLE_RATE || range.max_sample_rate() < SAMPLE_RATE {
            continue;
        }
        if !matches!(
            range.sample_format(),
            SampleFormat::F32 | SampleFormat::I16 | SampleFormat::I32 | SampleFormat::U16
        ) {
            continue;
        }
        let cand = range.with_sample_rate(SAMPLE_RATE);
        // 声道越少越好（省 downmix），格式 f32 优先
        let better = best.as_ref().is_none_or(|b| {
            (cand.channels(), cand.sample_format() != SampleFormat::F32)
                < (b.channels(), b.sample_format() != SampleFormat::F32)
        });
        if better {
            best = Some(cand);
        }
    }
    best
}

fn describe(device: &cpal::Device) -> String {
    device
        .description()
        .map_or_else(|_| "<unknown>".to_owned(), |d| d.name().to_owned())
}

/// 建立输入流：回调里 downmix + 重采样 + 入缓冲。
fn build_stream(config: &CaptureConfig, shared: &Arc<Shared>) -> Result<cpal::Stream, AudioError> {
    let (device, supported) = open_device(config)?;
    let channels = usize::from(supported.channels());
    let in_rate = supported.sample_rate();
    let format = supported.sample_format();
    let stream_config: StreamConfig = supported.into();

    let err_fn = |e: cpal::Error| log::warn!("audio stream error: {e}");

    macro_rules! build {
        ($t:ty) => {{
            let shared = Arc::clone(shared);
            let mut pipe = Pipeline::new(channels, in_rate);
            device.build_input_stream(
                stream_config,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    shared.push(pipe.process(data));
                },
                err_fn,
                None,
            )
        }};
    }

    let stream = match format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::U16 => build!(u16),
        other => return Err(AudioError::Format(other.to_string())),
    }
    .map_err(AudioError::from)?;
    stream.play().map_err(AudioError::from)?;
    Ok(stream)
}

/// 回调线程内的处理流水线：多声道 → 单声道 → 16 kHz。
struct Pipeline {
    channels: usize,
    mono: Vec<f32>,
    resampler: Option<Resampler>,
    out: Vec<f32>,
}

impl Pipeline {
    fn new(channels: usize, in_rate: u32) -> Self {
        Self {
            channels: channels.max(1),
            mono: Vec::new(),
            resampler: (in_rate != SAMPLE_RATE).then(|| Resampler::new(in_rate, SAMPLE_RATE)),
            out: Vec::new(),
        }
    }

    /// 交错样本 → 16k 单声道切片（返回内部缓冲的借用，稳态下零分配）。
    fn process<T: Copy>(&mut self, interleaved: &[T]) -> &[f32]
    where
        f32: FromSample<T>,
    {
        // 采集回调是实时线程：这里只做 downmix + 重采样，没有锁、没有系统调用。
        // fastrace 的 span 在无上层 context 时是零开销的，回调里也能安全埋点。
        let _span = fastrace::Span::enter_with_local_parent("audio_pipeline");
        self.mono.clear();
        if self.channels == 1 {
            self.mono
                .extend(interleaved.iter().map(|s| f32::from_sample(*s)));
        } else {
            let scale = 1.0 / self.channels as f32;
            for frame in interleaved.chunks(self.channels) {
                let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                self.mono.push(sum * scale);
            }
        }
        match self.resampler.as_mut() {
            Some(r) => {
                self.out.clear();
                r.process(&self.mono, &mut self.out);
                &self.out
            }
            None => &self.mono,
        }
    }
}

/// 秒 → 样本数（16 kHz）。
#[must_use]
pub fn secs_to_samples(secs: f32) -> usize {
    (secs.max(0.0) * SAMPLE_RATE as f32) as usize
}

/// 样本数 → 秒（16 kHz）。
#[must_use]
pub fn samples_to_secs(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE as f32
}
