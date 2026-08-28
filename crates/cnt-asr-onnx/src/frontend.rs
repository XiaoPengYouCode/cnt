//! 声学前端的后两级：LFR 降帧率 + CMVN 归一。
//!
//! `SenseVoice`/`Paraformer` 一类 `FunASR` 模型不是直接吃 80 维 fbank 的：
//!
//! ```text
//!   fbank 80 维 @100 fps ─► LFR(m=7, n=6) ─► 560 维 @16.7 fps ─► CMVN ─► 模型
//! ```
//!
//! **LFR**（low frame rate）把连续 7 帧拼成一帧、每次前进 6 帧：上下文进了特征、
//! 序列长度降到 1/6，encoder 的计算量随之降到 1/6——这是这类模型能在 CPU 上
//! 跑到实时的关键一步，不能省。
//!
//! 两处容易错的边界（与 `FunASR` 的 `apply_lfr` 对齐，错了会整体偏移半个窗）：
//! - 左侧用**第一帧**重复 `(m-1)/2 = 3` 次做 padding；
//! - 右侧不足一窗时用**最后一帧**补齐。

#![allow(clippy::cast_precision_loss)]

/// LFR 参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lfr {
    /// 窗口大小（拼几帧）。
    pub window: usize,
    /// 窗口步长（每次前进几帧）。
    pub shift: usize,
}

impl Default for Lfr {
    fn default() -> Self {
        Self {
            window: 7,
            shift: 6,
        }
    }
}

impl Lfr {
    /// 应用 LFR：`frames × dim` → `out_frames × (dim × window)`。
    #[must_use]
    pub fn apply(&self, feats: &[f32], dim: usize) -> Vec<f32> {
        let _span = fastrace::local::LocalSpan::enter_with_local_parent("lfr");
        if dim == 0 || feats.is_empty() {
            return Vec::new();
        }
        let frames = feats.len() / dim;
        let pad = (self.window - 1) / 2;
        let total = frames + pad;
        let out_frames = total.div_ceil(self.shift);
        let mut out = Vec::with_capacity(out_frames * dim * self.window);

        // 取「填充后」的第 i 帧：前 pad 帧都是第一帧，其后是原始帧（越界取最后一帧）
        let frame_at = |i: usize| -> &[f32] {
            let idx = i.saturating_sub(pad).min(frames - 1);
            &feats[idx * dim..(idx + 1) * dim]
        };

        for t in 0..out_frames {
            for k in 0..self.window {
                out.extend_from_slice(frame_at(t * self.shift + k));
            }
        }
        out
    }

    /// LFR 后的特征维数。
    #[must_use]
    pub const fn out_dim(&self, dim: usize) -> usize {
        dim * self.window
    }
}

/// CMVN（倒谱均值方差归一）：`x = (x + neg_mean) * inv_stddev`。
///
/// 参数与模型一起分发（`SenseVoice` 直接存在 ONNX 的 metadata 里），
/// 用错或缺失会让特征分布整体偏移，输出变成乱码或空——所以缺失时必须显式告警。
#[derive(Debug, Clone, Default)]
pub struct Cmvn {
    /// 负均值（长度 = LFR 后维数）。
    pub neg_mean: Vec<f32>,
    /// 标准差的倒数。
    pub inv_stddev: Vec<f32>,
}

impl Cmvn {
    /// 是否可用（两个向量非空且等长）。
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        !self.neg_mean.is_empty() && self.neg_mean.len() == self.inv_stddev.len()
    }

    /// 从空格分隔的字符串解析（ONNX metadata 的存法）。
    #[must_use]
    pub fn parse(neg_mean: &str, inv_stddev: &str) -> Self {
        Self {
            neg_mean: parse_floats(neg_mean),
            inv_stddev: parse_floats(inv_stddev),
        }
    }

    /// 原地归一化（维数不匹配时按最短的来，不 panic）。
    pub fn apply(&self, feats: &mut [f32]) {
        let _span = fastrace::local::LocalSpan::enter_with_local_parent("cmvn");
        if !self.is_valid() {
            return;
        }
        let dim = self.neg_mean.len();
        for frame in feats.chunks_mut(dim) {
            for (i, x) in frame.iter_mut().enumerate() {
                *x = (*x + self.neg_mean[i]) * self.inv_stddev[i];
            }
        }
    }
}

fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Cmvn, Lfr};

    #[test]
    fn lfr_shapes_match_funasr() {
        let lfr = Lfr::default();
        let dim = 2;
        // 12 帧 → pad 3 → 15 帧 → ceil(15/6) = 3 个输出帧，每帧 2×7=14 维
        let feats: Vec<f32> = (0..12 * dim).map(|i| i as f32).collect();
        let out = lfr.apply(&feats, dim);
        assert_eq!(lfr.out_dim(dim), 14);
        assert_eq!(out.len(), 3 * 14);
    }

    #[test]
    fn lfr_left_pads_with_first_frame() {
        let lfr = Lfr::default();
        let dim = 1;
        let feats: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let out = lfr.apply(&feats, dim);
        // 第一个输出帧 = [f0, f0, f0, f0, f1, f2, f3]
        assert_eq!(&out[..7], &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn lfr_right_pads_with_last_frame() {
        let lfr = Lfr::default();
        let dim = 1;
        let feats: Vec<f32> = (0..5).map(|i| i as f32).collect(); // f0..f4
        let out = lfr.apply(&feats, dim);
        // pad 3 + 5 = 8 帧 → ceil(8/6)=2 个输出帧；第二帧从 idx6 开始，越界处补 f4
        assert_eq!(out.len(), 2 * 7);
        assert_eq!(&out[7..], &[3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn lfr_handles_single_frame_input() {
        let out = Lfr::default().apply(&[1.0, 2.0], 2);
        assert_eq!(out.len(), 14);
        assert!(out.chunks(2).all(|f| f == [1.0, 2.0]));
    }

    #[test]
    fn lfr_empty_input_is_empty() {
        assert!(Lfr::default().apply(&[], 80).is_empty());
    }

    #[test]
    fn cmvn_applies_per_dimension() {
        let cmvn = Cmvn::parse("-1.0 -2.0", "0.5 0.25");
        assert!(cmvn.is_valid());
        let mut feats = vec![1.0, 2.0, 3.0, 4.0];
        cmvn.apply(&mut feats);
        assert_eq!(feats, vec![0.0, 0.0, 1.0, 0.5]);
    }

    #[test]
    fn invalid_cmvn_is_a_noop() {
        let cmvn = Cmvn::parse("1.0 2.0", "");
        assert!(!cmvn.is_valid());
        let mut feats = vec![1.0, 2.0];
        cmvn.apply(&mut feats);
        assert_eq!(feats, vec![1.0, 2.0]);
    }
}
