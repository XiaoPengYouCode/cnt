//! cnt-voice —— 语音输入编排：把「采集」和「识别」串成两种交互模式。
//!
//! ```text
//!   引擎（cnt-engine）
//!     │ start(PushToTalk) / start(Continuous) / stop()
//!     ▼
//!   Voice ──命令通道──► 编排线程 ──► cnt-audio::Recorder（cpal 线程）
//!     ◄──事件通道(tokio)──┘   └──► dyn Recognizer（cnt-asr-onnx）
//! ```
//!
//! ## 两种模式，两种取舍
//!
//! | 模式 | 触发 | 切句 | 延迟 | 适用 |
//! |---|---|---|---|---|
//! | [`Mode::PushToTalk`] | 按住某键说话，松手识别 | 由用户的手决定 | 松手 → 上屏 | 短句、确定性最高 |
//! | [`Mode::Continuous`] | 组合键切换常开 | VAD 自动切句 | 停顿 700 ms → 上屏 | 长段口述 |
//!
//! PTT 是默认路径：**「什么时候在录」由手指决定**，既没有 VAD 误判，也没有
//! 「忘了关」的隐私风险；常开模式适合连续口述，代价是要靠 VAD 猜句子边界。
//!
//! ## 线程与延迟
//!
//! 编排在独立 OS 线程上跑，因为推理是几百毫秒的**阻塞** CPU 工作，绝不能占
//! zbus 的 tokio 执行器（会把整个输入法的按键响应堵住）。事件用 tokio 通道
//! 回传，引擎侧 `await` 即可。
//!
//! ## 可观测性
//!
//! 每句话一棵 root span `voice_utterance`，子 span 覆盖前端/推理/解码，
//! 属性带上音频长度、识别耗时与 RTF（实时率）。语音链路的性能判断全靠这棵树：
//! 「慢」到底慢在采集尾巴、fbank 还是 encoder，不看 span 树只能猜。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver as SyncReceiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fastrace::collector::SpanContext;
use fastrace::local::LocalSpan;
use fastrace::{Event, Span};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use cnt_asr::{AsrError, Punctuator, Recognizer, TextScorer};
use cnt_audio::vad::{Segment, Segmenter, VadConfig};
use cnt_audio::{samples_to_secs, AudioError, CaptureConfig, Recorder};

/// 编排线程的轮询间隔：足够细（不给尾字增加可感延迟），又不至于空转烧 CPU。
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// 音量/时长上报间隔：够跟手，又不会把 D-Bus 刷爆。
const LEVEL_INTERVAL: Duration = Duration::from_millis(200);

/// 语言模型在融合分里的权重（shallow fusion 的 λ）。
///
/// 与拼音侧重排的默认 λ 一致（`cnt_score::RescorePolicy`）：融合而不是替代 ——
/// 声学分仍是主，语言模型只在它拿不定主意时起决定作用。
const LM_WEIGHT: f32 = 0.5;
/// 触发重排的声学分差门限（log10）。
///
/// #1 比 #2 领先超过这个幅度就不动：声学已经很确定，此时让语言模型插手
/// 只会把「说得不常见但确实说了」的话改成「常见但不是我说的」。
const RESCORE_GAP: f32 = 1.5;
/// 自然对数 → log10（CTC 用自然对数，`cnt-lm` 用 log10，必须统一口径）。
const LN_TO_LOG10: f32 = std::f32::consts::LOG10_E;

/// 语音输入错误。
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    /// 音频子系统错误。
    #[error("audio: {0}")]
    Audio(#[from] AudioError),
    /// 识别后端错误。
    #[error("asr: {0}")]
    Asr(#[from] AsrError),
    /// 已经有会话在进行。
    #[error("a voice session is already active")]
    Busy,
    /// 编排线程已退出。
    #[error("voice worker is gone")]
    WorkerGone,
}

/// 交互模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 按住说话：松手即识别整段。
    PushToTalk,
    /// 常开：VAD 自动切句，逐句上屏。
    Continuous,
}

impl Mode {
    /// 模式名（日志/UI 用）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PushToTalk => "push-to-talk",
            Self::Continuous => "continuous",
        }
    }
}

/// 会话事件（引擎据此更新预编辑与上屏）。
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceEvent {
    /// 麦克风已打开，开始采集。
    Started(Mode),
    /// 采集中的实时状态：已录时长（秒）+ 当前音量（dBFS）。
    ///
    /// 非流式模型没有中间结果，界面上必须有**别的**东西证明「它在听」，
    /// 否则用户不知道该不该继续说。这条事件就是那个证据。
    Level { secs: f32, db: f32 },
    /// 采集结束，正在识别（PTT 松手后 UI 应显示「识别中」）。
    Recognizing,
    /// 一段识别结果（常开模式一句一条）。
    Text(String),
    /// 这一段没有识别出内容（静音 / 只有噪声）。
    Empty,
    /// 出错（麦克风被占用、模型报错……）。
    Error(String),
    /// 会话结束，麦克风已关闭。
    Stopped,
}

/// 语音输入配置。
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// 采集配置（设备选择）。
    pub capture: CaptureConfig,
    /// VAD / 切句配置（常开模式用）。
    pub vad: VadConfig,
    /// PTT 单次最长录音（秒）：防按键卡住导致无限录音。
    pub max_seconds: f32,
    /// 短于此长度（秒）的 PTT 录音直接忽略（误触）。
    pub min_seconds: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            capture: CaptureConfig::default(),
            vad: VadConfig::default(),
            max_seconds: 60.0,
            min_seconds: 0.25,
        }
    }
}

/// 一次会话的事件流（引擎持有；drop 即不再关心后续事件）。
pub struct VoiceSession {
    rx: UnboundedReceiver<VoiceEvent>,
}

impl VoiceSession {
    /// 等下一个事件（会话结束后返回 None）。
    pub async fn next(&mut self) -> Option<VoiceEvent> {
        self.rx.recv().await
    }
}

/// 编排线程命令。
enum Cmd {
    Start(Mode, UnboundedSender<VoiceEvent>),
    /// 结束采集并识别（PTT 松手 / 关常开）。
    Stop,
    /// 放弃：停止采集且**不识别、不上屏**（说错了、按错了）。
    Cancel,
    Quit,
}

/// 语音输入门面：引擎只用这几个方法。
pub struct Voice {
    cmd: SyncSender<Cmd>,
    active: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
    backend: String,
    punct: String,
    rescorer: String,
    device: String,
}

impl Voice {
    /// 建立语音输入（打开麦克风设备探测一次，启动编排线程）。
    ///
    /// # Errors
    /// 没有可用输入设备时返回错误（此时上层应把语音功能标记为不可用，
    /// 但输入法的其他功能必须照常工作）。
    pub fn new(
        recognizer: Arc<dyn Recognizer>,
        punctuator: Arc<dyn Punctuator>,
        scorer: Option<Arc<dyn TextScorer>>,
        config: VoiceConfig,
    ) -> Result<Self, VoiceError> {
        let recorder = Recorder::spawn(&config.capture)?;
        let device = recorder.device_name().to_owned();
        let backend = recognizer.name().to_owned();
        let punct = punctuator.name().to_owned();
        let rescorer = scorer
            .as_ref()
            .map_or_else(|| "none".to_owned(), |s| s.name().to_owned());
        let active = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let worker_active = Arc::clone(&active);
        let handle = std::thread::Builder::new()
            .name("cnt-voice".into())
            .spawn(move || {
                let worker = Worker {
                    recorder,
                    recognizer,
                    punctuator,
                    scorer,
                    config,
                    active: worker_active,
                };
                worker.run(&rx);
            })
            .map_err(|e| VoiceError::Audio(AudioError::Cpal(e.to_string())))?;
        Ok(Self {
            cmd: tx,
            active,
            handle: Some(handle),
            backend,
            punct,
            rescorer,
            device,
        })
    }

    /// 识别后端名。
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// 标点后端名。
    #[must_use]
    pub fn punctuator(&self) -> &str {
        &self.punct
    }

    /// 重排器名（`none` = 未装语言模型）。
    #[must_use]
    pub fn rescorer(&self) -> &str {
        &self.rescorer
    }

    /// 输入设备名。
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }

    /// 是否有会话在进行。
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// 开一个会话，返回事件流。
    ///
    /// # Errors
    /// 已有会话在进行（[`VoiceError::Busy`]）或编排线程已退出。
    pub fn start(&self, mode: Mode) -> Result<VoiceSession, VoiceError> {
        if self.active.swap(true, Ordering::SeqCst) {
            return Err(VoiceError::Busy);
        }
        let (tx, rx) = unbounded_channel();
        self.cmd.send(Cmd::Start(mode, tx)).map_err(|_| {
            self.active.store(false, Ordering::SeqCst);
            VoiceError::WorkerGone
        })?;
        Ok(VoiceSession { rx })
    }

    /// 结束采集（PTT 松手 / 关掉常开）。识别结果随后通过事件流送达。
    ///
    /// # Errors
    /// 编排线程已退出时返回错误。
    pub fn stop(&self) -> Result<(), VoiceError> {
        self.cmd.send(Cmd::Stop).map_err(|_| VoiceError::WorkerGone)
    }

    /// 放弃当前会话：停止采集，**不识别也不上屏**。
    ///
    /// # Errors
    /// 编排线程已退出时返回错误。
    pub fn cancel(&self) -> Result<(), VoiceError> {
        self.cmd
            .send(Cmd::Cancel)
            .map_err(|_| VoiceError::WorkerGone)
    }
}

impl Drop for Voice {
    fn drop(&mut self) {
        let _ = self.cmd.send(Cmd::Quit);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// 编排线程状态。
struct Worker {
    recorder: Recorder,
    recognizer: Arc<dyn Recognizer>,
    /// 标点恢复（没装模型时是 `NoPunct`，零开销直通）。
    punctuator: Arc<dyn Punctuator>,
    /// n-best 重排的语言模型（None = 只用声学 1-best）。
    scorer: Option<Arc<dyn TextScorer>>,
    config: VoiceConfig,
    active: Arc<AtomicBool>,
}

impl Worker {
    fn run(&self, rx: &SyncReceiver<Cmd>) {
        loop {
            match rx.recv() {
                Ok(Cmd::Start(mode, tx)) => self.session(mode, &tx, rx),
                // 无会话时的 stop/cancel：忽略
                Ok(Cmd::Stop | Cmd::Cancel) => {}
                Ok(Cmd::Quit) | Err(_) => return,
            }
        }
    }

    /// 一次会话的完整生命周期（采集 → 识别 → 结束）。
    fn session(&self, mode: Mode, tx: &UnboundedSender<VoiceEvent>, rx: &SyncReceiver<Cmd>) {
        let started = Instant::now();
        if let Err(e) = self.recorder.start() {
            log::error!("voice: cannot open microphone: {e}");
            let _ = tx.send(VoiceEvent::Error(e.to_string()));
            let _ = tx.send(VoiceEvent::Stopped);
            self.active.store(false, Ordering::SeqCst);
            return;
        }
        log::info!("voice: session started ({}, {})", mode.as_str(), self.recorder.device_name());
        let _ = tx.send(VoiceEvent::Started(mode));

        let mut segmenter = Segmenter::new(self.config.vad);
        let mut ptt_buffer: Vec<f32> = Vec::new();
        let max_samples = cnt_audio::secs_to_samples(self.config.max_seconds);
        let mut quit = false;
        let mut cancelled = false;
        let mut recorded: usize = 0;
        let mut last_level = Instant::now();

        // ---- 采集阶段 ----
        loop {
            match rx.recv_timeout(POLL_INTERVAL) {
                Ok(Cmd::Stop) => break,
                Ok(Cmd::Cancel) => {
                    cancelled = true;
                    break;
                }
                // 会话中重复 start：忽略（引擎已用 active 挡住）
                Ok(Cmd::Start(..)) | Err(RecvTimeoutError::Timeout) => {}
                Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => {
                    quit = true;
                    cancelled = true;
                    break;
                }
            }

            let chunk = self.recorder.drain();
            recorded += chunk.len();
            // 每 ~200 ms 报一次时长与音量（界面据此画「在听」的证据）
            if !chunk.is_empty() && last_level.elapsed() >= LEVEL_INTERVAL {
                last_level = Instant::now();
                let _ = tx.send(VoiceEvent::Level {
                    secs: samples_to_secs(recorded),
                    db: cnt_audio::vad::frame_db(&chunk),
                });
            }
            match mode {
                Mode::PushToTalk => {
                    ptt_buffer.extend_from_slice(&chunk);
                    if ptt_buffer.len() >= max_samples {
                        log::warn!(
                            "voice: hit max recording length ({} s), finishing",
                            self.config.max_seconds
                        );
                        break;
                    }
                }
                Mode::Continuous => {
                    for segment in segmenter.push(&chunk) {
                        // 常开模式：每切出一句就是一次完整的「听到→上屏」，
                        // 各自一棵 root（会话可能持续几分钟，不能等它结束才上报）
                        let seg_root = Span::root("voice_segment", SpanContext::random())
                            .with_property(|| ("start_secs", format!("{:.1}", segment.start_secs)));
                        let _guard = seg_root.set_local_parent();
                        self.recognize_segment(&segment, tx);
                    }
                }
            }
        }

        // ---- 收尾：关麦克风 ----
        if let Err(e) = self.recorder.stop() {
            log::warn!("voice: stopping capture failed: {e}");
        }
        let tail = self.recorder.drain();

        // 放弃：录到的音频直接丢掉，一个字都不上屏
        if cancelled {
            log::info!(
                "voice: cancelled, discarding {:.2}s of audio",
                samples_to_secs(recorded + tail.len())
            );
            drop(tail);
            drop(ptt_buffer);
            let _ = tx.send(VoiceEvent::Stopped);
            self.active.store(false, Ordering::SeqCst);
            return;
        }
        // ★ 用户真正感受到的延迟：**松手到上屏**。
        //
        // 此前只埋了 transcribe 的耗时，但那不是体验——从松手到文字出现之间还有
        // 排空音频、标点、重排、事件传递。少埋这一段，就会出现「span 树看着很快、
        // 人却觉得慢」的经典盲区。PTT 的预算是 300ms，管的是这个数字。
        let commit = Span::root("voice_release_to_commit", SpanContext::random())
            .with_property(|| ("mode", mode.as_str().to_owned()))
            .with_property(|| ("cancelled", "false".to_owned()));
        let released = Instant::now();
        {
            let _guard = commit.set_local_parent();
            let _ = tx.send(VoiceEvent::Recognizing);
            self.finish(mode, &mut ptt_buffer, &tail, &mut segmenter, tx);
        }
        commit.add_property(|| ("elapsed_ms", format!("{:.1}", released.elapsed().as_secs_f32() * 1000.0)));
        drop(commit);
        log::info!(
            "voice: release→commit {:.0}ms",
            released.elapsed().as_secs_f32() * 1000.0
        );

        log::info!("voice: session ended after {:.1}s", started.elapsed().as_secs_f32());
        let _ = tx.send(VoiceEvent::Stopped);
        self.active.store(false, Ordering::SeqCst);
        if quit {
            // Quit 命令在会话中到达：结束线程（Voice 正在 drop）
            self.recorder.stop().ok();
        }
    }

    /// 收尾：把剩余音频识别掉（PTT 是整段，常开是尾巴 + flush）。
    fn finish(
        &self,
        mode: Mode,
        ptt_buffer: &mut Vec<f32>,
        tail: &[f32],
        segmenter: &mut Segmenter,
        tx: &UnboundedSender<VoiceEvent>,
    ) {
        match mode {
            Mode::PushToTalk => {
                ptt_buffer.extend_from_slice(tail);
                let secs = samples_to_secs(ptt_buffer.len());
                if secs < self.config.min_seconds {
                    log::debug!(
                        "voice: ignoring {secs:.2}s tap (min {}s)",
                        self.config.min_seconds
                    );
                    let _ = tx.send(VoiceEvent::Empty);
                } else {
                    self.recognize(ptt_buffer, false, tx);
                }
            }
            Mode::Continuous => {
                for segment in segmenter.push(tail) {
                    self.recognize_segment(&segment, tx);
                }
                if let Some(segment) = segmenter.flush() {
                    self.recognize_segment(&segment, tx);
                }
                // 会话级 span：VAD 的判定质量只能在「一整段会话」的尺度上看
                // （切了几句、丢了几段、噪声底跑到哪去了），逐句 span 看不出来
                let stats = segmenter.stats();
                let session = Span::root("voice_session", SpanContext::random())
                    .with_property(|| ("segments", stats.segments.to_string()))
                    .with_property(|| ("dropped", stats.dropped.to_string()))
                    .with_property(|| ("frames", stats.frames.to_string()))
                    .with_property(|| ("speech_frames", stats.speech_frames.to_string()))
                    .with_property(|| ("noise_floor_db", format!("{:.1}", stats.noise_floor_db)));
                drop(session);
                log::info!(
                    "voice: continuous session done: {} segments, {} dropped, noise floor {:.1} dBFS",
                    stats.segments,
                    stats.dropped,
                    stats.noise_floor_db
                );
            }
        }
    }

    /// n-best 重排（委托给纯函数 [`rescore_nbest`]，便于单测）。
    fn rescore(&self, transcript: &mut cnt_asr::Transcript) {
        if let Some(scorer) = &self.scorer {
            rescore_nbest(transcript, scorer.as_ref());
        }
    }

    fn recognize_segment(&self, segment: &Segment, tx: &UnboundedSender<VoiceEvent>) {
        log::debug!(
            "voice: segment at {:.1}s, {:.2}s long{}",
            segment.start_secs,
            samples_to_secs(segment.samples.len()),
            if segment.forced { " (forced cut)" } else { "" }
        );
        self.recognize(&segment.samples, segment.forced, tx);
    }

    /// 识别一段音频并发事件。每句话一棵 root span。
    fn recognize(&self, samples: &[f32], forced: bool, tx: &UnboundedSender<VoiceEvent>) {
        let audio_secs = samples_to_secs(samples.len());
        let started = Instant::now();
        // 每句一个 span，挂在**外层 root** 下（PTT 是 voice_release_to_commit，
        // 常开的中途切句是 voice_segment）。这样一棵树里同时看得到
        // 「用户等了多久」和「时间花在哪一级」。
        // span 必须活到 RTF 算出来之后 —— 它是这棵树上最该有的那个数字。
        // 这里用 Span 而不是 LocalSpan：需要在 span 结束后补 RTF 属性（要先算完耗时），
        // 而 LocalSpan 的 add_property 是静态的、只作用于「当前」local span，做不到这点
        let root = Span::enter_with_local_parent("voice_utterance")
            .with_property(|| ("audio_secs", format!("{audio_secs:.2}")))
            .with_property(|| ("forced", forced.to_string()));
        let result = {
            let _guard = root.set_local_parent();
            let mut out = self.recognizer.transcribe(samples);
            // n-best 重排：声学分不确定时让语言模型说话（详见 rescore_nbest）
            if let Ok(t) = &mut out {
                LocalSpan::add_property(|| ("nbest", t.alternatives.len().to_string()));
                self.rescore(t);
            }
            // 标点恢复：**失败就用原文**（端口契约），一句话绝不能因为标点丢掉
            if let Ok(t) = &mut out
                && !t.text.is_empty()
            {
                match self.punctuator.restore(&t.text) {
                    Ok(punctuated) => t.text = punctuated,
                    Err(e) => log::warn!("voice: punctuation failed, using raw text: {e}"),
                }
            }
            if let Ok(t) = &out {
                LocalSpan::add_property(|| ("chars", t.text.chars().count().to_string()));
                LocalSpan::add_property(|| {
                    ("lang", t.language.clone().unwrap_or_else(|| "?".to_owned()))
                });
            }
            out
        };
        let elapsed = started.elapsed().as_secs_f32();
        // RTF（real-time factor）= 推理耗时 / 音频时长。这是语音链路唯一有意义的
        // 性能指标：<1 才可能跟得上说话速度，PTT 体验上要求 <0.3。
        let rtf = if audio_secs > 0.0 {
            elapsed / audio_secs
        } else {
            0.0
        };
        // RTF 挂到 root 上：它是语音链路唯一有意义的性能指标，
        // 必须能在 span 树里直接看到，而不是只在日志里
        root.add_property(|| ("rtf", format!("{rtf:.3}")));
        drop(root);
        match result {
            Ok(t) if t.is_empty() => {
                log::info!("voice: empty result ({audio_secs:.2}s audio, {elapsed:.2}s, rtf {rtf:.2})");
                let _ = tx.send(VoiceEvent::Empty);
            }
            Ok(t) => {
                log::info!(
                    "voice: {:?} ({audio_secs:.2}s audio, {elapsed:.2}s, rtf {rtf:.2}, {} tokens)",
                    t.text,
                    t.tokens.len()
                );
                let _ = tx.send(VoiceEvent::Text(t.text));
            }
            Err(e) => {
                log::error!("voice: recognition failed: {e}");
                let _ = tx.send(VoiceEvent::Error(e.to_string()));
            }
        }
    }
}

/// n-best 重排：`融合分 = 声学(log10) + λ × 语言模型(log10)`。
///
/// 语音识别的主要错误是**音对字错**（`瓶颈`→`平境`、`语音`→`原音`），
/// 声学模型对这类错误无能为力——它听到的音确实是那个音。而中文 n-gram
/// 恰好能分辨哪串字更像一句话。
///
/// 三条保守约束（与拼音侧的重排策略同源）：
///
/// 1. 只有 ≥2 条候选才重排（1-best 无从比较）；
/// 2. #1 领先 #2 超过 [`RESCORE_GAP`] 就不动 —— 声学已经确定，
///    此时插手只会把「说得不常见但确实说了」改成「常见但不是我说的」；
/// 3. 融合而非替代（λ = [`LM_WEIGHT`]），声学分始终是主。
///
/// 纯函数：不碰麦克风、不碰模型，可以直接单测。
pub fn rescore_nbest(transcript: &mut cnt_asr::Transcript, scorer: &dyn TextScorer) {
    if transcript.alternatives.len() < 2 {
        return;
    }
    let _span = LocalSpan::enter_with_local_parent("voice_rescore");

    // 声学分换成 log10，与语言模型同口径
    let acoustic: Vec<f32> = transcript
        .alternatives
        .iter()
        .map(|h| h.acoustic * LN_TO_LOG10)
        .collect();
    let gap = acoustic[0] - acoustic[1];
    if gap > RESCORE_GAP {
        log::debug!("voice: acoustic is confident (gap {gap:.2}), skipping rescore");
        return;
    }

    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, hyp) in transcript.alternatives.iter().enumerate() {
        let Some(lm) = scorer.logp10(&hyp.text) else {
            continue;
        };
        let fused = LM_WEIGHT.mul_add(lm, acoustic[i]);
        log::debug!(
            "voice: nbest[{i}] {:?} acoustic={a:.2} lm={lm:.2} fused={fused:.2}",
            hyp.text,
            a = acoustic[i]
        );
        if fused > best.1 {
            best = (i, fused);
        }
    }
    if best.0 != 0 {
        let picked = transcript.alternatives[best.0].text.clone();
        log::info!(
            "voice: rescored {:?} → {picked:?} (gap was {gap:.2})",
            transcript.text
        );
        LocalSpan::add_event(
            Event::new("rescored")
                .with_property(|| ("from", transcript.text.clone()))
                .with_property(|| ("to", picked.clone()))
                .with_property(|| ("model", scorer.name().to_owned())),
        );
        transcript.text = picked;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cnt_asr::{AsrError, Recognizer, Transcript};

    use super::{Mode, VoiceConfig, VoiceEvent};

    /// 假识别器：不碰麦克风、不碰模型，用来测编排契约。
    struct Fake;

    impl Recognizer for Fake {
        fn transcribe(&self, samples: &[f32]) -> Result<Transcript, AsrError> {
            let text = format!("{}样本", samples.len());
            Ok(Transcript {
                tokens: vec!["x".to_owned()],
                language: Some("zh".to_owned()),
                alternatives: vec![cnt_asr::Hypothesis {
                    text: text.clone(),
                    acoustic: -1.0,
                }],
                text,
            })
        }
        fn name(&self) -> &'static str {
            "fake"
        }
    }

    #[test]
    fn mode_names_are_stable() {
        assert_eq!(Mode::PushToTalk.as_str(), "push-to-talk");
        assert_eq!(Mode::Continuous.as_str(), "continuous");
    }

    #[test]
    fn default_config_is_sane() {
        let cfg = VoiceConfig::default();
        assert!(cfg.max_seconds > cfg.min_seconds);
        assert!(cfg.min_seconds > 0.0);
    }

    #[test]
    fn recognizer_port_is_object_safe() {
        // 端口必须能 dyn 化（编排层只持有 Arc<dyn Recognizer>）
        let r: Arc<dyn Recognizer> = Arc::new(Fake);
        assert_eq!(r.name(), "fake");
        assert_eq!(
            r.transcribe(&[0.0; 3]).expect("fake never fails").text,
            "3样本"
        );
    }

    /// 假打分器：把「正确答案」打高分，用来验证融合与触发条件。
    struct FakeScorer;

    impl cnt_asr::TextScorer for FakeScorer {
        fn logp10(&self, text: &str) -> Option<f32> {
            // 「瓶颈」是通顺的，「平境」不是
            Some(if text.contains('瓶') { -2.0 } else { -6.0 })
        }
        fn name(&self) -> &'static str {
            "fake"
        }
    }

    fn transcript(pairs: &[(&str, f32)]) -> Transcript {
        let alts: Vec<cnt_asr::Hypothesis> = pairs
            .iter()
            .map(|(t, a)| cnt_asr::Hypothesis {
                text: (*t).to_owned(),
                acoustic: *a,
            })
            .collect();
        Transcript {
            text: alts[0].text.clone(),
            tokens: Vec::new(),
            language: None,
            alternatives: alts,
        }
    }

    #[test]
    fn rescore_flips_homophone_when_acoustic_is_unsure() {
        // 声学分接近（gap 0.2 log10）→ 语言模型说话
        let mut t = transcript(&[("平境在哪里", -10.0), ("瓶颈在哪里", -10.5)]);
        super::rescore_nbest(&mut t, &FakeScorer);
        assert_eq!(t.text, "瓶颈在哪里");
    }

    #[test]
    fn rescore_keeps_acoustic_when_confident() {
        // 声学 #1 大幅领先（自然对数差 5 ≈ log10 差 2.17 > 1.5）→ 不许改
        let mut t = transcript(&[("平境在哪里", -10.0), ("瓶颈在哪里", -15.0)]);
        super::rescore_nbest(&mut t, &FakeScorer);
        assert_eq!(t.text, "平境在哪里", "声学确定时不该被语言模型翻掉");
    }

    #[test]
    fn rescore_needs_at_least_two_candidates() {
        let mut t = transcript(&[("平境", -10.0)]);
        super::rescore_nbest(&mut t, &FakeScorer);
        assert_eq!(t.text, "平境");
    }

    #[test]
    fn rescore_ignores_unscorable_candidates() {
        struct NoScore;
        impl cnt_asr::TextScorer for NoScore {
            fn logp10(&self, _text: &str) -> Option<f32> {
                None
            }
            fn name(&self) -> &'static str {
                "noscore"
            }
        }
        let mut t = transcript(&[("平境", -10.0), ("瓶颈", -10.1)]);
        super::rescore_nbest(&mut t, &NoScore);
        assert_eq!(t.text, "平境", "打不了分就保持声学顺序");
    }

    #[test]
    fn events_are_comparable() {
        // 引擎侧要按事件类型分派，事件需要可比较
        assert_eq!(
            VoiceEvent::Text("你好".to_owned()),
            VoiceEvent::Text("你好".to_owned())
        );
        assert_ne!(VoiceEvent::Empty, VoiceEvent::Stopped);
    }
}
