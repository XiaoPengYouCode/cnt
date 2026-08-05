//! 重排阶段（应用层编排）：把「基线候选」交给 [`Rescorer`] 端口，再按策略融合。
//!
//! 这一层只做编排，不含模型：
//! 1. 按 [`RescorePolicy`] 判断是否值得重排（默认多数按键直接跳过）；
//! 2. 取前 `top_n` 条构造 [`SentenceHyp`]；
//! 3. 调模型 → `fuse(base, model)` → 只对窗口内重排序（窗口外顺序不动）。
//!
//! 埋点：`rescore` span + `rescored` 事件（候选数）。默认关闭 trace 时为 noop；
//! 诊断时可直接在阶段聚合表里看到重排占按键延迟的比例，据此决定是否开启。

use cnt_input::Candidate;
use cnt_score::{RescorePolicy, Rescorer, Segment, SentenceHyp};
use fastrace::local::LocalSpan;
use fastrace::Event;

/// 对 `scored`（按分数降序）就地应用重排。返回是否真的重排了。
pub fn apply(
    rescorer: &dyn Rescorer,
    policy: &RescorePolicy,
    scored: &mut [(Candidate, f32)],
) -> bool {
    let n = policy.window(scored.len());
    if n < 2 {
        return false;
    }
    let _span = LocalSpan::enter_with_local_parent("rescore");

    // 借用期：构造 hyps（segments 先落地成 owned Vec，hyps 借用它）
    let segments: Vec<Vec<Segment<'_>>> = scored[..n]
        .iter()
        .map(|(c, _)| {
            c.learned
                .iter()
                .map(|l| Segment {
                    pinyin: l.pinyin.as_str(),
                    word: l.word.as_str(),
                })
                .collect()
        })
        .collect();
    let hyps: Vec<SentenceHyp<'_>> = scored[..n]
        .iter()
        .zip(&segments)
        .map(|((c, score), segs)| SentenceHyp {
            text: c.text.as_str(),
            segments: segs,
            base_score: *score,
        })
        .collect();

    if !policy.should_rescore(&hyps) {
        return false; // 基线足够确定 / 上下文太短：省掉模型开销
    }
    let Some(model_scores) = rescorer.score_sentences(&hyps) else {
        return false; // 模型放弃（未就绪/超时）：保持基线顺序
    };
    if model_scores.len() != n {
        log::warn!(
            "rescorer {} 返回 {} 个分数，期望 {n}，忽略本次重排",
            rescorer.name(),
            model_scores.len()
        );
        return false;
    }
    let fused: Vec<f32> = hyps
        .iter()
        .zip(&model_scores)
        .map(|(h, m)| policy.fuse(h.base_score, *m))
        .collect();
    drop(hyps);
    drop(segments);

    // 写回融合分并只重排窗口内（窗口外的候选与之未可比，顺序保持不变）
    for (slot, score) in scored[..n].iter_mut().zip(fused) {
        slot.1 = score;
    }
    scored[..n].sort_by(|a, b| b.1.total_cmp(&a.1));
    LocalSpan::add_event(
        Event::new("rescored")
            .with_property(|| ("count", n.to_string()))
            .with_property(|| ("model", rescorer.name())),
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use cnt_input::LearnedWord;

    /// 测试用重排器：偏好指定文本（给它 0 分，其余 -10 分）。
    struct Prefer(&'static str);

    impl Rescorer for Prefer {
        fn name(&self) -> &'static str {
            "test-prefer"
        }

        fn score_sentences(&self, hyps: &[SentenceHyp<'_>]) -> Option<Vec<f32>> {
            Some(
                hyps.iter()
                    .map(|h| if h.text == self.0 { 0.0 } else { -10.0 })
                    .collect(),
            )
        }
    }

    fn cand(text: &str, segs: usize) -> Candidate {
        Candidate::whole(
            text,
            (0..segs)
                .map(|i| LearnedWord::new(format!("p{i}"), text.to_string()))
                .collect(),
            segs * 2,
        )
    }

    #[test]
    fn rescoring_can_flip_close_candidates() {
        let mut scored = vec![(cand("先在", 2), -4.0), (cand("现在", 2), -4.2)];
        let policy = RescorePolicy::default();
        assert!(apply(&Prefer("现在"), &policy, &mut scored));
        assert_eq!(scored[0].0.text, "现在");
    }

    #[test]
    fn confident_baseline_is_left_alone() {
        // 分差 3.0 ≥ gap：不调用模型，顺序不变
        let mut scored = vec![(cand("现在", 2), -2.0), (cand("先在", 2), -5.0)];
        let policy = RescorePolicy::default();
        assert!(!apply(&Prefer("先在"), &policy, &mut scored));
        assert_eq!(scored[0].0.text, "现在");
    }

    #[test]
    fn window_limits_touched_candidates() {
        let policy = RescorePolicy {
            top_n: 2,
            ..RescorePolicy::default()
        };
        let mut scored = vec![
            (cand("先在", 2), -4.0),
            (cand("现在", 2), -4.2),
            (cand("鲜在", 2), -4.3),
        ];
        assert!(apply(&Prefer("鲜在"), &policy, &mut scored));
        // 窗口外的 鲜在 未参与，仍在最后；窗口内按模型分重排
        assert_eq!(scored[2].0.text, "鲜在");
    }
}
