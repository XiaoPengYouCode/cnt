//! Kaldi 兼容的 fbank（log mel filterbank）特征提取。
//!
//! 为什么自己写：所有主流中文 ASR 模型（`FunASR`/`WeNet`/`Paraformer`/`SenseVoice`）
//! 的输入都是 kaldi 口径的 80 维 log-mel，训练侧用的是
//! `torchaudio.compliance.kaldi.fbank`。要么引一个 C++ 特征库（多一个外部依赖、
//! 交叉编译更麻烦），要么照口径实现一遍——这部分数学是固定的，且**必须逐位对齐**：
//! 窗函数、预加重、去直流的顺序错一步，识别率就会莫名下降十几个点。
//!
//! 口径（与 kaldi 默认一致）：
//!
//! | 项 | 值 |
//! |---|---|
//! | 帧长 / 帧移 | 25 ms / 10 ms（16 kHz → 400 / 160 样本） |
//! | 窗 | povey：`(0.5 - 0.5cos(2πi/(N-1)))^0.85` |
//! | 去直流 | 每帧减均值（在预加重之前） |
//! | 预加重 | 0.97 |
//! | FFT | 512（帧长向上取 2 的幂） |
//! | mel | 80 维，20 Hz ~ 奈奎斯特，`mel(f)=1127ln(1+f/700)` |
//! | 取对数 | `ln(max(e, f32::EPSILON))` |
//! | 幅度口径 | 样本按 int16 量级（×32768），与训练侧一致 |

// 信号处理全程在「样本索引 ↔ 频率」之间换算，f32/f64/usize 的转换是本质工作，
// 逐处加 allow 只会淹没代码；这里在模块级统一放开，精度损失是可接受且有意的。
// suboptimal_flops/imprecise_flops：本模块的公式必须与 kaldi 的实现逐行对照
// （谁改都要能一眼看出「这是预加重、这是 povey 窗、这是三角滤波器」），
// 改写成 mul_add / ln_1p 会让对照失效，而 fbank 的精度瓶颈在 f32 与窗函数本身。
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::imprecise_flops
)]

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// mel 尺度：`1127 * ln(1 + f/700)`。
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// fbank 参数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbankOptions {
    /// 采样率（Hz）。
    pub sample_rate: u32,
    /// 帧长（毫秒）。
    pub frame_length_ms: f32,
    /// 帧移（毫秒）。
    pub frame_shift_ms: f32,
    /// mel 滤波器组维数。
    pub num_bins: usize,
    /// mel 下限频率（Hz）。
    pub low_freq: f32,
    /// mel 上限频率（Hz）；`<=0` 表示奈奎斯特。
    pub high_freq: f32,
    /// 预加重系数。
    pub preemphasis: f32,
    /// 样本幅度缩放：模型训练侧用 int16 量级时为 32768。
    pub sample_scale: f32,
}

impl Default for FbankOptions {
    fn default() -> Self {
        Self {
            sample_rate: cnt_asr::SAMPLE_RATE,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            num_bins: 80,
            low_freq: 20.0,
            high_freq: 0.0,
            preemphasis: 0.97,
            sample_scale: 32768.0,
        }
    }
}

impl FbankOptions {
    /// 帧长（样本数）。
    #[must_use]
    pub fn frame_length(&self) -> usize {
        (self.frame_length_ms * self.sample_rate as f32 / 1000.0) as usize
    }

    /// 帧移（样本数）。
    #[must_use]
    pub fn frame_shift(&self) -> usize {
        (self.frame_shift_ms * self.sample_rate as f32 / 1000.0) as usize
    }
}

/// 一条 mel 滤波器：起始 FFT bin + 权重。
struct MelBank {
    offset: usize,
    weights: Vec<f32>,
}

/// fbank 提取器（窗、滤波器组、FFT 计划都在构造时算好，`compute` 只做算术）。
pub struct Fbank {
    opts: FbankOptions,
    window: Vec<f32>,
    banks: Vec<MelBank>,
    fft: Arc<dyn Fft<f32>>,
    fft_size: usize,
}

impl Fbank {
    /// 按参数建立提取器。
    #[must_use]
    pub fn new(opts: FbankOptions) -> Self {
        let frame_length = opts.frame_length();
        let fft_size = frame_length.next_power_of_two();
        let window = povey_window(frame_length);
        let banks = mel_banks(&opts, fft_size);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(fft_size);
        Self {
            opts,
            window,
            banks,
            fft,
            fft_size,
        }
    }

    /// 特征维数。
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.opts.num_bins
    }

    /// 给定样本数能切出多少帧（`snip_edges=true`：不足一帧的尾部丢弃）。
    #[must_use]
    pub fn num_frames(&self, num_samples: usize) -> usize {
        let (len, shift) = (self.opts.frame_length(), self.opts.frame_shift());
        if num_samples < len {
            0
        } else {
            1 + (num_samples - len) / shift
        }
    }

    /// 计算 log-mel 特征，返回 `frames × num_bins` 的行主序展开。
    #[must_use]
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let _span = fastrace::local::LocalSpan::enter_with_local_parent("fbank");
        let frame_length = self.opts.frame_length();
        let shift = self.opts.frame_shift();
        let frames = self.num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * self.opts.num_bins);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.fft_size];
        let mut work = vec![0.0f32; frame_length];

        for f in 0..frames {
            let src = &samples[f * shift..][..frame_length];
            // 1. 幅度换到训练口径
            for (dst, s) in work.iter_mut().zip(src) {
                *dst = *s * self.opts.sample_scale;
            }
            // 2. 去直流（逐帧减均值）
            let mean = work.iter().sum::<f32>() / frame_length as f32;
            for x in &mut work {
                *x -= mean;
            }
            // 3. 预加重（从后往前，避免用到已改写的值）
            let preemph = self.opts.preemphasis;
            if preemph != 0.0 {
                for i in (1..frame_length).rev() {
                    work[i] -= preemph * work[i - 1];
                }
                work[0] -= preemph * work[0];
            }
            // 4. 加窗 + 零填充到 FFT 长度
            for (slot, (x, w)) in buf.iter_mut().zip(work.iter().zip(&self.window)) {
                *slot = Complex32::new(x * w, 0.0);
            }
            for slot in &mut buf[frame_length..] {
                *slot = Complex32::new(0.0, 0.0);
            }
            // 5. FFT → 功率谱
            self.fft.process(&mut buf);
            // 6. mel 加权求和 + 取对数
            for bank in &self.banks {
                let mut energy = 0.0f32;
                for (i, w) in bank.weights.iter().enumerate() {
                    let c = buf[bank.offset + i];
                    energy += w * (c.re * c.re + c.im * c.im);
                }
                out.push(energy.max(f32::EPSILON).ln());
            }
        }
        out
    }
}

/// povey 窗：kaldi 的默认窗（汉宁窗的 0.85 次幂，两端更接近 0）。
fn povey_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let a = 2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0);
            (0.5 - 0.5 * a.cos()).powf(0.85)
        })
        .collect()
}

/// 三角 mel 滤波器组（在 mel 尺度上等间距）。
fn mel_banks(opts: &FbankOptions, fft_size: usize) -> Vec<MelBank> {
    let num_fft_bins = fft_size / 2;
    let nyquist = opts.sample_rate as f32 / 2.0;
    let high = if opts.high_freq <= 0.0 {
        nyquist
    } else {
        opts.high_freq.min(nyquist)
    };
    let bin_width = opts.sample_rate as f32 / fft_size as f32;
    let mel_low = hz_to_mel(opts.low_freq);
    let mel_high = hz_to_mel(high);
    let delta = (mel_high - mel_low) / (opts.num_bins as f32 + 1.0);

    (0..opts.num_bins)
        .map(|bin| {
            let left = mel_low + bin as f32 * delta;
            let center = left + delta;
            let right = left + 2.0 * delta;
            let mut offset = 0;
            let mut weights: Vec<f32> = Vec::new();
            for k in 0..num_fft_bins {
                let mel = hz_to_mel(bin_width * k as f32);
                let w = if mel > left && mel < right {
                    if mel <= center {
                        (mel - left) / delta
                    } else {
                        (right - mel) / delta
                    }
                } else {
                    0.0
                };
                if w > 0.0 {
                    if weights.is_empty() {
                        offset = k;
                    }
                    weights.push(w);
                } else if !weights.is_empty() {
                    break; // 三角形是连续区间，出了右边界即可停
                }
            }
            MelBank { offset, weights }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{hz_to_mel, mel_banks, povey_window, Fbank, FbankOptions};

    #[test]
    fn frame_count_follows_kaldi_snip_edges() {
        let fb = Fbank::new(FbankOptions::default());
        // 1 s @16k：(16000-400)/160 + 1 = 98 帧
        assert_eq!(fb.num_frames(16_000), 98);
        // 不足一帧
        assert_eq!(fb.num_frames(399), 0);
        assert_eq!(fb.num_frames(400), 1);
    }

    #[test]
    fn output_shape_is_frames_times_bins() {
        let fb = Fbank::new(FbankOptions::default());
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.05).sin() * 0.2)
            .collect();
        let feats = fb.compute(&samples);
        assert_eq!(feats.len(), 98 * 80);
        assert!(feats.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn window_is_zero_at_edges_and_one_at_center() {
        let w = povey_window(400);
        assert!(w[0].abs() < 1e-6);
        assert!(w[399].abs() < 1e-6);
        assert!((w[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn mel_scale_is_monotonic() {
        assert!(hz_to_mel(20.0) < hz_to_mel(1_000.0));
        assert!(hz_to_mel(1_000.0) < hz_to_mel(8_000.0));
    }

    #[test]
    fn mel_banks_cover_the_spectrum_without_gaps() {
        let opts = FbankOptions::default();
        let banks = mel_banks(&opts, 512);
        assert_eq!(banks.len(), 80);
        // 每条滤波器都非空，且中心频率随 bin 递增
        assert!(banks.iter().all(|b| !b.weights.is_empty()));
        let centers: Vec<usize> = banks
            .iter()
            .map(|b| {
                let (i, _) = b
                    .weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("non-empty");
                b.offset + i
            })
            .collect();
        assert!(centers.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn silence_gives_floor_values() {
        let fb = Fbank::new(FbankOptions::default());
        let feats = fb.compute(&vec![0.0; 4_000]);
        // 全静音 → 能量被 f32::EPSILON 兜住，ln 后是一个很小的常数
        assert!(feats.iter().all(|v| *v < -15.0), "silence should hit floor");
    }

    #[test]
    fn louder_input_gives_higher_energy() {
        let fb = Fbank::new(FbankOptions::default());
        let quiet: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.1).sin() * 0.01).collect();
        let loud: Vec<f32> = quiet.iter().map(|s| s * 10.0).collect();
        let a: f32 = fb.compute(&quiet).iter().sum();
        let b: f32 = fb.compute(&loud).iter().sum();
        assert!(b > a, "louder audio must yield higher log energy");
    }
}
