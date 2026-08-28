//! 重排触发策略与分数融合（值对象）。
//!
//! 重排要花钱（延迟），所以**默认不做**，只在「基线自己也没把握」时做。
//! 判据全部来自基线分数本身，不需要额外模型：
//!
//! - 候选太少（<2 条）→ 排序无意义；
//! - 太短（单段）→ 神经模型没有上下文可利用，收益接近 0；
//! - `#1` 与 `#2` 分差大 → 基线已经很确定，重排只会带来抖动风险。

use crate::rescore::SentenceHyp;

/// 参与重排的候选数上限（推荐值）。
pub const DEFAULT_TOP_N: usize = 10;
/// 触发重排的最小片段数（推荐值：至少两段才有上下文）。
pub const DEFAULT_MIN_SEGMENTS: usize = 2;
/// `#1` 与 `#2` 分差阈值（log10）：小于此值说明基线不确定，值得重排。
pub const DEFAULT_GAP: f32 = 1.5;
/// 融合权重 λ（推荐值）：`final = base + λ × model`。
pub const DEFAULT_WEIGHT: f32 = 0.5;

/// 重排策略：何时触发 + 如何融合。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RescorePolicy {
    /// 只对前 `top_n` 条候选调用模型（模型代价与此线性相关）。
    pub top_n: usize,
    /// 少于这么多片段的候选句不重排。
    pub min_segments: usize,
    /// `#1` 与 `#2` 基线分差 ≥ 此值时跳过重排（基线足够确定）。
    pub gap: f32,
    /// 融合权重 λ：`final = base + λ × model`。
    ///
    /// 保留基线分是关键——用户调频、模糊音惩罚、用户词都编码在基线里，
    /// 完全交给模型会丢掉这些硬约束带来的确定性。
    pub weight: f32,
}

impl Default for RescorePolicy {
    fn default() -> Self {
        Self {
            top_n: DEFAULT_TOP_N,
            min_segments: DEFAULT_MIN_SEGMENTS,
            gap: DEFAULT_GAP,
            weight: DEFAULT_WEIGHT,
        }
    }
}

impl RescorePolicy {
    /// 是否值得对这批候选做重排（`hyps` 须按基线分降序）。
    #[must_use]
    pub fn should_rescore(&self, hyps: &[SentenceHyp<'_>]) -> bool {
        let (Some(first), Some(second)) = (hyps.first(), hyps.get(1)) else {
            return false; // 0/1 条候选：没有可排的顺序
        };
        if first.segments.len() < self.min_segments {
            return false; // 单段（单字/单词）：无上下文可利用
        }
        if !first.base_score.is_finite() || !second.base_score.is_finite() {
            return false; // 词候选（-inf）等非 LM 分，不进入重排空间
        }
        first.base_score - second.base_score < self.gap
    }

    /// 融合基线分与模型分。
    #[must_use]
    pub const fn fuse(&self, base: f32, model: f32) -> f32 {
        self.weight.mul_add(model, base)
    }

    /// 本次实际参与重排的候选数。
    #[must_use]
    pub const fn window(&self, len: usize) -> usize {
        if len < self.top_n { len } else { self.top_n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rescore::Segment;

    fn hyp<'a>(text: &'a str, segs: &'a [Segment<'a>], score: f32) -> SentenceHyp<'a> {
        SentenceHyp {
            text,
            segments: segs,
            base_score: score,
        }
    }

    #[test]
    fn skips_when_baseline_is_confident() {
        let segs = [
            Segment {
                pinyin: "xian",
                word: "现",
            },
            Segment {
                pinyin: "zai",
                word: "在",
            },
        ];
        let p = RescorePolicy::default();
        // 分差 3.0 ≥ gap(1.5)：基线确定，不重排
        assert!(!p.should_rescore(&[hyp("现在", &segs, -2.0), hyp("先在", &segs, -5.0)]));
        // 分差 0.3 < gap：值得重排
        assert!(p.should_rescore(&[hyp("现在", &segs, -2.0), hyp("先在", &segs, -2.3)]));
    }

    #[test]
    fn skips_single_segment_and_short_lists() {
        let one = [Segment {
            pinyin: "shi",
            word: "是",
        }];
        let p = RescorePolicy::default();
        assert!(!p.should_rescore(&[hyp("是", &one, -1.0), hyp("时", &one, -1.1)]));
        assert!(!p.should_rescore(&[hyp("是", &one, -1.0)]));
        assert!(!p.should_rescore(&[]));
    }

    #[test]
    fn fuse_keeps_baseline_and_scales_model() {
        let p = RescorePolicy {
            weight: 0.5,
            ..RescorePolicy::default()
        };
        assert!((p.fuse(-4.0, -2.0) - (-5.0)).abs() < 1e-6);
        assert_eq!(p.window(30), DEFAULT_TOP_N);
        assert_eq!(p.window(3), 3);
    }

    mod user {
        use super::super::user;

        #[test]
        fn zero_count_returns_pure_lm() {
            // 无证据（含未转正新词）→ 纯 LM，无任何 lift
            for lm in [-1.0, -3.0, -9.0] {
                assert!(
                    (user::mixture(lm, 0) - lm).abs() < 1e-6,
                    "count=0 应原样返回 LM"
                );
            }
        }

        #[test]
        fn lift_is_monotone_and_ceilinged() {
            // 核心性质（对齐 libime）：用户分是概率尺度，混合结果有绝对上限 ——
            // 用户证据再强，单段分数也不会超过 log P = 1（≈0），不会造出任意大的 gap。
            // 核心性质（对齐 libime）：用户分是概率尺度，混合结果有绝对上限 ——
            // 用户证据再强，单段分数也不会超过 log P = 1（≈0），不会造出任意大的 gap。
            let counts = [1, 2, 5, 10, 32, 100];
            let mut prev = f32::NEG_INFINITY;
            for c in counts {
                let m = user::mixture(-3.0, c);
                assert!(m > prev, "单调递增: c={c} m={m} prev={prev}");
                assert!(m <= 1e-3, "绝对上限 logP=1: c={c} m={m}");
                prev = m;
            }
            // LM 再差（-9 地板）也一样封顶
            for c in [1, 100] {
                assert!(user::mixture(-9.0, c) <= 1e-3);
            }
        }

        #[test]
        fn mixture_keeps_lm_ordering_when_user_weak() {
            // 用户证据弱时混合≈纯 LM：LM 决定量级（这是「拼接路径走 LM 打分」的
            // 机制基础 —— 单次/低次选择顶多把候选往上抬一小截，不翻转 LM 顺序）
            for c in [1, 2] {
                assert!(
                    user::mixture(-2.0, c) > user::mixture(-4.5, c),
                    "LM 差仍主导: c={c}"
                );
            }
            // 但同计数下 LM 差被压缩而不是抹平：好 LM 词始终 ≥ 差 LM 词
            for c in [1, 10, 100] {
                assert!(user::mixture(-2.0, c) >= user::mixture(-4.5, c) - 1e-3);
            }
        }

        #[test]
        fn voice_boost_unchanged_scale() {
            // 语音 n-best 的加法贡献保持原有有界量级
            assert!((user::boost(0) - 0.0).abs() < 1e-6);
            assert!((user::boost(10) - 2.0).abs() < 1e-6);
            assert!((user::boost(100) - 2.0).abs() < 1e-6, "封顶");
        }
    }
}

/// 用户调频加成的口径（**拼音与语音共用同一把尺**）。
///
/// 用户词是「这个用户自己确认过的证据」，两条链路都该按同一标准采信：
/// 拼音侧是候选排序里的加成，语音侧是 n-best 重排里的加成。
/// 分散成两个常量早晚会漂移，所以放在打分领域里做单一来源。
pub mod user {
    /// 每选一次的 log10 权重（**语音 n-best 用**：粗粒度有界加法贡献）。
    ///
    /// 拼音解码侧不走这里 —— 那是 log-linear 混合（`mixture`），逐步 lift 有界；
    /// 语音侧没有逐词 LM 分可用，只能对整句做加法，保留原来的有界增量。
    /// 两者共享同一个 `count`（未转正词计 0 → 无加成），是同一把尺。
    pub const BOOST_LOG: f32 = 0.2;
    /// 封顶次数：超过后不再增长。
    ///
    /// 防止「了/个」这类高频字无限刷分——32 次选择的加成不该压过整个分数空间。
    pub const BOOST_CAP: u32 = 10;
    /// log-linear 混合里用户模型的权重 w（libime 口径，初值；用 fuzzy-test 定案）。
    ///
    /// `WA = log10(1-w)`、`WB = log10(w)` 进 logsumexp，w=0.5 时两者各 -0.301：
    /// 用户分和 LM 分按同权重混合，谁也不自带优先权。
    pub const MIX_WEIGHT: f32 = 0.5;
    /// 用户计数 → 概率尺度映射的平滑常数 K（初值；`user_logp = log10(count/(count+K))`）。
    ///
    /// K 决定「一次选择给多大 lift」：K=100 时 count=1 → -2.0（≈0.01），
    /// count=32 → -0.6（≈0.25），count=100 → -0.3 —— 有界、单调、概率尺度。
    pub const MIX_SMOOTH_K: f32 = 100.0;

    /// 选择次数 → log10 加成（语音 n-best 加法路径，拼音侧不用）。
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // 计数 ≤ BOOST_CAP，u32→f32 无损
    pub fn boost(count: u32) -> f32 {
        count.min(BOOST_CAP) as f32 * BOOST_LOG
    }

    /// 用户计数 → log10 概率尺度（≤ 0，有界；count=0 返回 -∞）。
    ///
    /// 这是 libime `UserLanguageModel` 里 userScore 的口径：用户分是概率，
    /// 与 LM 分同一空间，混合后的 lift 有界 —— 不再出现
    /// 从(+2.0)+是(+2.0) 的线性叠加直接盖过 LM 的量级。
    #[must_use]
    fn user_logp(count: u32) -> f32 {
        if count == 0 {
            return f32::NEG_INFINITY;
        }
        #[allow(clippy::cast_precision_loss)] // 计数 ≤ MAX_COUNT(100)，u32→f32 无损
        let c = count as f32;
        (c / (c + MIX_SMOOTH_K)).log10()
    }

    /// libime 口径 log-linear 混合：`max(lm, logsumexp(lm + WA, user + WB))`。
    ///
    /// 用户分是概率尺度（`user_logp`），与 LM 分同一空间；`count == 0`（无证据，
    /// 含未转正新词）时直接返回纯 LM —— 未转正的词只可见不竞争。
    ///
    /// 语义对齐 `userlanguagemodel.cpp:139`：`std::max(score,
    /// sum_log_prob(score + wa, userScore + wb))`。逐步 lift 有界，拼接路径的
    /// 量级由 LM 决定（从是 vs 重试），单字计数顶多把候选往上抬一小截。
    #[must_use]
    pub fn mixture(lm: f32, count: u32) -> f32 {
        if count == 0 {
            return lm;
        }
        let wa = (1.0 - MIX_WEIGHT).log10();
        let wb = MIX_WEIGHT.log10();
        let a = lm + wa;
        let b = user_logp(count) + wb;
        let max = a.max(b);
        let sum = max + (1.0 + 10f32.powf(-(max - a.min(b)))).log10();
        lm.max(sum)
    }
}
