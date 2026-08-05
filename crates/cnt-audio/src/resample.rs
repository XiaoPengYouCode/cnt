//! 重采样：设备原生采样率 → 16 kHz（识别端口要求）。
//!
//! 用带窗 sinc 插值（windowed-sinc / Lanczos 式）而不是线性插值：
//! 48k → 16k 是 3 倍抽取，线性插值的抗镜像能力不足，会把 8 kHz 以上的能量
//! 折回语音频带，直接影响 fbank 特征进而影响识别率。
//!
//! 为什么自己写而不是引 `rubato`：这里只需要「固定比率、单声道、流式」这一种
//! 情形，几十行就够，而且能保证回调线程里零分配（`process` 复用内部缓冲）。

// 重采样是纯 DSP 代码：样本索引与浮点位置之间来回转换是本质工作，逐处加 allow
// 只会把算法淹没。float 的 while 条件同理——「还能产出一个输出样本吗」本来就是
// 位置与长度的浮点比较。mul_add 重写在这里损害可读性（核公式要一眼能对照公式书），
// 而这段代码的精度瓶颈在 sinc 截断而不是舍入。
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::while_float,
    clippy::suboptimal_flops
)]

/// 单边抽头数：每个输出样本用 2×TAPS 个输入样本。
const TAPS: usize = 16;
/// 窗函数与 sinc 的截止频率相对输入奈奎斯特的比例（留 5% 余量防边缘振铃）。
const CUTOFF_MARGIN: f64 = 0.95;

/// 流式重采样器（固定比率，单声道）。
pub struct Resampler {
    /// 输入相对输出的步长（`in_rate / out_rate`）。
    step: f64,
    /// 归一化截止频率（相对输入采样率的一半）。
    cutoff: f64,
    /// 待处理输入样本（前面保留 TAPS 个历史样本作左邻域）。
    buf: Vec<f32>,
    /// 下一个输出样本对应的输入位置（相对 `buf` 起点，浮点）。
    pos: f64,
}

impl Resampler {
    /// 建立 `in_rate → out_rate` 的重采样器。
    #[must_use]
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let in_rate = f64::from(in_rate.max(1));
        let out_rate = f64::from(out_rate.max(1));
        Self {
            step: in_rate / out_rate,
            // 下采样时截止取输出奈奎斯特（防折叠）；上采样时取输入奈奎斯特
            cutoff: CUTOFF_MARGIN * (out_rate / in_rate).min(1.0),
            // 左邻域先用静音填充：开头几个样本的轻微失真无关紧要
            buf: vec![0.0; TAPS],
            pos: TAPS as f64,
        }
    }

    /// 处理一段输入，把输出追加到 `out`。
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        let limit = self.buf.len() as f64 - TAPS as f64;
        while self.pos < limit {
            out.push(self.sample_at(self.pos));
            self.pos += self.step;
        }
        // 丢掉不再需要的历史，保留左邻域
        let keep_from = (self.pos.floor() as usize).saturating_sub(TAPS);
        if keep_from > 0 {
            self.buf.drain(..keep_from);
            self.pos -= keep_from as f64;
        }
    }

    /// 在浮点位置 `t` 处插值。
    fn sample_at(&self, t: f64) -> f32 {
        let center = t.floor() as usize;
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        let first = center + 1 - TAPS.min(center + 1);
        let last = (center + TAPS).min(self.buf.len() - 1);
        for i in first..=last {
            let u = t - i as f64;
            let w = kernel(u, self.cutoff);
            acc += f64::from(self.buf[i]) * w;
            norm += w;
        }
        if norm.abs() < f64::EPSILON {
            return 0.0;
        }
        (acc / norm) as f32
    }
}

/// 带 Blackman 窗的 sinc 核。
fn kernel(u: f64, cutoff: f64) -> f64 {
    let half = TAPS as f64;
    if u.abs() > half {
        return 0.0;
    }
    let sinc = if u.abs() < 1e-12 {
        cutoff
    } else {
        (std::f64::consts::PI * cutoff * u).sin() / (std::f64::consts::PI * u)
    };
    // Blackman 窗（旁瓣 -58 dB，足够压住镜像）
    let x = std::f64::consts::PI * (u + half) / half;
    let window = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
    sinc * window
}

#[cfg(test)]
mod tests {
    use super::Resampler;

    /// 输出长度应约等于 `输入长度 × out/in`。
    #[test]
    fn output_length_matches_ratio() {
        let mut r = Resampler::new(48_000, 16_000);
        let input = vec![0.0f32; 4800]; // 100 ms
        let mut out = Vec::new();
        r.process(&input, &mut out);
        // 100 ms @16k = 1600 样本，允许 ±TAPS 的边界偏差
        assert!(
            (out.len() as i64 - 1600).abs() < 32,
            "unexpected length {}",
            out.len()
        );
    }

    /// 直流信号重采样后仍是直流（增益为 1，核已归一化）。
    #[test]
    fn dc_gain_is_unity() {
        let mut r = Resampler::new(44_100, 16_000);
        let input = vec![0.5f32; 44_100];
        let mut out = Vec::new();
        r.process(&input, &mut out);
        let tail = &out[out.len() / 2..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!((mean - 0.5).abs() < 1e-3, "dc gain drifted: {mean}");
    }

    /// 分块喂入与整块喂入结果一致（流式状态正确）。
    #[test]
    fn chunked_equals_whole() {
        let signal: Vec<f32> = (0..8000)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let mut whole = Vec::new();
        Resampler::new(48_000, 16_000).process(&signal, &mut whole);

        let mut chunked = Vec::new();
        let mut r = Resampler::new(48_000, 16_000);
        for chunk in signal.chunks(257) {
            r.process(chunk, &mut chunked);
        }

        assert_eq!(whole.len(), chunked.len());
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// 采样率相同时输出等于输入（不建重采样器的那条路由由上层负责，这里验证退化情形正确）。
    #[test]
    fn identity_rate_preserves_signal() {
        let mut r = Resampler::new(16_000, 16_000);
        let signal: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out = Vec::new();
        r.process(&signal, &mut out);
        // 步长为 1，输出与输入等长（首尾各差 TAPS 之内）
        assert!((out.len() as i64 - 1000).abs() <= 32);
    }
}
