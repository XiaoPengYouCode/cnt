//! 语音活动检测（VAD）与切句：常开模式下把连续音频切成「一句话」。
//!
//! 用**自适应能量门限**（噪声底 + 相对余量），不用固定阈值：机械键盘、风扇、
//! 不同麦克风增益差十几 dB，固定阈值必然在某台机器上失效。
//!
//! ```text
//!   帧能量 dBFS ─► 噪声底跟踪（快落慢升）─► 门限 = 噪声底 + margin
//!                                            └─► 状态机（起音判定 / 尾静音挂起）
//! ```
//!
//! 两个必须有的细节：
//!
//! 1. **pre-roll**：起音判定要攒够几帧才确认，等确认时开头已经过去了，
//!    所以缓冲区要往前回溯一段（默认 300 ms），否则每句话头字都被吃掉；
//! 2. **尾静音挂起**：说话中的停顿不该断句，得连续静音超过 `trailing_silence`
//!    才认为一句结束（默认 700 ms）。
//!
//! 语音识别的准确率对切句非常敏感：切早了丢字，切晚了整段延迟上升。
//! 所以这两个量都是配置项，且 [`Segmenter::stats`] 把判定过程暴露出来可观测。

// VAD 全程在「样本数 ↔ 毫秒 ↔ dB」之间换算，这些转换是本质工作而非疏漏。
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]

use std::collections::VecDeque;

use cnt_asr::SAMPLE_RATE;

/// 一帧的时长（毫秒）。20 ms 是能量 VAD 的常规取值：短到能跟上起音，
/// 长到不会被单个爆破音带偏。
pub const FRAME_MS: usize = 20;
/// 一帧的样本数（16 kHz）。
pub const FRAME_SAMPLES: usize = SAMPLE_RATE as usize * FRAME_MS / 1000;
/// 收句时保留的尾静音（毫秒）：给声学模型一点收尾，多的剪掉。
const KEEP_TAIL_MS: usize = 200;

/// VAD / 切句配置。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadConfig {
    /// 判为语音所需的「高于噪声底」的余量（dB）。
    pub margin_db: f32,
    /// 绝对静音门限（dBFS）：低于此值一律算静音，防全静音环境下噪声底塌到 -90 后误触发。
    pub floor_db: f32,
    /// 起音确认所需的连续语音帧数。
    pub attack_frames: usize,
    /// 一句结束所需的连续静音时长（毫秒）。
    pub trailing_silence_ms: usize,
    /// 回溯缓冲时长（毫秒）：保住句首。
    pub preroll_ms: usize,
    /// 单句最长时长（秒）：超过强制切一刀，避免识别延迟无上限。
    pub max_segment_secs: f32,
    /// 有效句子的最短时长（毫秒）：短于此丢弃（咳嗽、敲键盘）。
    pub min_speech_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            margin_db: 10.0,
            floor_db: -55.0,
            attack_frames: 3,
            trailing_silence_ms: 700,
            preroll_ms: 300,
            max_segment_secs: 15.0,
            min_speech_ms: 250,
        }
    }
}

impl VadConfig {
    const fn trailing_silence_frames(&self) -> usize {
        let frames = self.trailing_silence_ms / FRAME_MS;
        if frames == 0 { 1 } else { frames }
    }

    const fn preroll_samples(&self) -> usize {
        self.preroll_ms / FRAME_MS * FRAME_SAMPLES
    }

    fn max_segment_samples(&self) -> usize {
        (self.max_segment_secs.max(1.0) * SAMPLE_RATE as f32) as usize
    }
}

/// 切出来的一句话。
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// 16 kHz 单声道样本。
    pub samples: Vec<f32>,
    /// 该句在整个会话中的起始时刻（秒，含 pre-roll）。
    pub start_secs: f32,
    /// 是否因为超长被强制切断（诊断用：频繁出现说明 `trailing_silence` 太长）。
    pub forced: bool,
}

/// 判定过程的可观测指标。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VadStats {
    /// 处理过的帧数。
    pub frames: usize,
    /// 判为语音的帧数。
    pub speech_frames: usize,
    /// 当前噪声底（dBFS）。
    pub noise_floor_db: f32,
    /// 最近一帧能量（dBFS）。
    pub last_db: f32,
    /// 切出的句子数。
    pub segments: usize,
    /// 被丢弃的过短片段数。
    pub dropped: usize,
}

/// 流式切句器：喂任意长度的样本，吐出完整的句子。
pub struct Segmenter {
    config: VadConfig,
    /// 未凑满一帧的余样本。
    pending: Vec<f32>,
    /// 静音期的回溯缓冲（pre-roll）。
    preroll: VecDeque<f32>,
    /// 正在累积的句子。
    current: Option<Vec<f32>>,
    /// 当前句子里真正判为语音的帧数（判断“够不够长”只能用这个：
    /// 段长包含 pre-roll 与尾静音，用它会把一下敲击声当成一秒的语音）。
    speech_frames: usize,
    /// 连续语音帧计数（静音态用于起音判定）。
    speech_run: usize,
    /// 连续静音帧计数（语音态用于结句判定）。
    silence_run: usize,
    /// 噪声底（dBFS，None = 尚未初始化）。
    noise_floor: Option<f32>,
    /// 会话已消费的样本数（算 `start_secs`）。
    consumed: usize,
    stats: VadStats,
}

impl Segmenter {
    /// 新建切句器。
    #[must_use]
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            pending: Vec::with_capacity(FRAME_SAMPLES * 2),
            preroll: VecDeque::new(),
            current: None,
            speech_frames: 0,
            speech_run: 0,
            silence_run: 0,
            noise_floor: None,
            consumed: 0,
            stats: VadStats::default(),
        }
    }

    /// 判定指标快照。
    #[must_use]
    pub const fn stats(&self) -> VadStats {
        self.stats
    }

    /// 当前是否处于「正在说话」状态（供 UI 显示）。
    #[must_use]
    pub const fn in_speech(&self) -> bool {
        self.current.is_some()
    }

    /// 喂一段样本，返回本次完成的句子（可能 0～多句）。
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        let _span = fastrace::local::LocalSpan::enter_with_local_parent("vad_push");
        let mut done = Vec::new();
        self.pending.extend_from_slice(samples);
        let full = self.pending.len() / FRAME_SAMPLES;
        let mut frame = [0.0f32; FRAME_SAMPLES];
        for i in 0..full {
            frame.copy_from_slice(&self.pending[i * FRAME_SAMPLES..][..FRAME_SAMPLES]);
            if let Some(seg) = self.push_frame(&frame) {
                done.push(seg);
            }
        }
        self.pending.drain(..full * FRAME_SAMPLES);
        done
    }

    /// 结束会话：把正在累积的句子交出来（够长的话）。
    pub fn flush(&mut self) -> Option<Segment> {
        // 余下不足一帧的样本也带上（尾字不能丢）
        let tail = std::mem::take(&mut self.pending);
        if let Some(cur) = self.current.as_mut() {
            cur.extend_from_slice(&tail);
        }
        self.speech_run = 0;
        self.silence_run = 0;
        self.finish_current(false)
    }

    /// 处理一帧，返回可能完成的句子。
    fn push_frame(&mut self, frame: &[f32; FRAME_SAMPLES]) -> Option<Segment> {
        let db = frame_db(frame);
        let is_speech = self.classify(db);
        self.stats.frames += 1;
        self.stats.last_db = db;
        if is_speech {
            self.stats.speech_frames += 1;
        }

        // ---- 语音态：累积，直到尾静音够长或超长 ----
        if let Some(cur) = self.current.as_mut() {
            {
                cur.extend_from_slice(frame);
                if is_speech {
                    self.silence_run = 0;
                    self.speech_frames += 1;
                } else {
                    self.silence_run += 1;
                }
                let ended = self.silence_run >= self.config.trailing_silence_frames();
                let too_long = self
                    .current
                    .as_ref()
                    .is_some_and(|c| c.len() >= self.config.max_segment_samples());
                if ended || too_long {
                    self.consumed += FRAME_SAMPLES;
                    let seg = self.finish_current(too_long && !ended);
                    self.silence_run = 0;
                    self.speech_run = 0;
                    return seg;
                }
            }
        } else {
            // ---- 静音态：攒 pre-roll，起音够久则开句 ----
            {
                self.preroll.extend(frame.iter().copied());
                let cap = self.config.preroll_samples();
                while self.preroll.len() > cap {
                    self.preroll.pop_front();
                }
                if is_speech {
                    self.speech_run += 1;
                } else {
                    self.speech_run = 0;
                }
                if self.speech_run >= self.config.attack_frames {
                    // 开句：把 pre-roll 一起带进来（否则头字被吃掉）
                    let started: Vec<f32> = self.preroll.drain(..).collect();
                    self.current = Some(started);
                    self.silence_run = 0;
                    self.speech_frames = self.speech_run;
                }
            }
        }
        self.consumed += FRAME_SAMPLES;
        None
    }

    /// 能量门限 + 噪声底跟踪（快落慢升）。
    fn classify(&mut self, db: f32) -> bool {
        let floor = self.noise_floor.unwrap_or(db);
        let updated = if db < floor {
            // 比噪声底还低：快速跟下去（0.3 的系数 ≈ 3 帧收敛）
            floor + 0.3 * (db - floor)
        } else {
            // 环境变吵：每帧只允许爬 0.05 dB（≈2.5 dB/s），
            // 保证一句长语音不会把噪声底抬到自己头上
            floor + 0.05
        };
        self.noise_floor = Some(updated);
        self.stats.noise_floor_db = updated;
        db > updated + self.config.margin_db && db > self.config.floor_db
    }

    /// 收句：过短丢弃，剪掉多余的尾静音，并把开始时刻算出来。
    fn finish_current(&mut self, forced: bool) -> Option<Segment> {
        let mut samples = self.current.take()?;
        let speech_ms = self.speech_frames * FRAME_MS;
        self.speech_frames = 0;
        if speech_ms < self.config.min_speech_ms {
            self.stats.dropped += 1;
            return None;
        }
        let start = self.consumed.saturating_sub(samples.len());
        // 尾静音只留 KEEP_TAIL_MS：剩下的那几百毫秒静音对识别没用，
        // 但会原样变成推理时长（帧数↑ → encoder 计算量↑）。
        let keep_frames = KEEP_TAIL_MS / FRAME_MS;
        let trim_frames = self.silence_run.saturating_sub(keep_frames);
        let trim = (trim_frames * FRAME_SAMPLES).min(samples.len());
        samples.truncate(samples.len() - trim);
        self.stats.segments += 1;
        Some(Segment {
            start_secs: start as f32 / SAMPLE_RATE as f32,
            samples,
            forced,
        })
    }
}

/// 一帧的 RMS 能量（dBFS）。
#[must_use]
pub fn frame_db(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return -120.0;
    }
    let sum: f64 = frame.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    let rms = (sum / frame.len() as f64).sqrt();
    (20.0 * (rms + 1e-12).log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::{frame_db, Segmenter, VadConfig, FRAME_SAMPLES};

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.2).sin() * amp)
            .collect()
    }

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn frame_db_matches_amplitude() {
        // 满幅正弦 RMS ≈ -3 dBFS，静音接近下限
        let db = frame_db(&tone(FRAME_SAMPLES, 1.0));
        assert!(db > -6.0 && db < 0.0, "unexpected {db}");
        assert!(frame_db(&silence(FRAME_SAMPLES)) < -100.0);
    }

    #[test]
    fn segments_speech_between_silences() {
        let mut seg = Segmenter::new(VadConfig::default());
        // 500 ms 静音 → 1 s 语音 → 1 s 静音
        let mut out = seg.push(&silence(8_000));
        out.extend(seg.push(&tone(16_000, 0.3)));
        out.extend(seg.push(&silence(16_000)));
        assert_eq!(out.len(), 1, "expected exactly one segment");
        let s = &out[0];
        assert!(!s.forced);
        // 语音 1 s + pre-roll 0.3 s + 尾静音 0.7 s ≈ 2 s
        let secs = s.samples.len() as f32 / 16_000.0;
        assert!(secs > 1.0 && secs < 2.5, "unexpected length {secs}s");
    }

    #[test]
    fn short_noise_is_dropped() {
        let mut seg = Segmenter::new(VadConfig::default());
        let mut out = seg.push(&silence(8_000));
        out.extend(seg.push(&tone(1_600, 0.5))); // 100 ms 敲击
        out.extend(seg.push(&silence(16_000)));
        assert!(out.is_empty(), "short burst should be dropped");
        assert_eq!(seg.stats().dropped, 1);
    }

    #[test]
    fn pause_does_not_split_sentence() {
        let mut seg = Segmenter::new(VadConfig::default());
        let mut out = seg.push(&silence(8_000));
        out.extend(seg.push(&tone(8_000, 0.3)));
        out.extend(seg.push(&silence(4_800))); // 300 ms 停顿 < 700 ms
        out.extend(seg.push(&tone(8_000, 0.3)));
        out.extend(seg.push(&silence(16_000)));
        assert_eq!(out.len(), 1, "pause must not split the sentence");
    }

    #[test]
    fn flush_returns_pending_speech() {
        let mut seg = Segmenter::new(VadConfig::default());
        seg.push(&silence(8_000));
        seg.push(&tone(16_000, 0.3));
        // 还没等到尾静音就结束会话（松手 / 关闭常开）
        let tail = seg.flush().expect("pending speech must be returned");
        assert!(tail.samples.len() >= 16_000);
        assert!(seg.flush().is_none(), "flush is idempotent");
    }

    #[test]
    fn long_speech_is_force_cut() {
        let cfg = VadConfig {
            max_segment_secs: 2.0,
            ..VadConfig::default()
        };
        let mut seg = Segmenter::new(cfg);
        seg.push(&silence(8_000));
        let out = seg.push(&tone(16_000 * 5, 0.3));
        assert!(out.len() >= 2, "5 s speech with 2 s cap must be cut");
        assert!(out[0].forced, "first cut is forced");
    }
}
