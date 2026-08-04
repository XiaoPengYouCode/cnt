//! `cnt-score` 打分端口的 n-gram 实现（适配器）。
//!
//! 方向是「实现 → 端口」：解码器只认 [`NgramLm`]，本文件把 mmap 的 `.cntl`
//! 接上去。Rust 的孤儿规则也要求 impl 落在 `CntLm` 所属的 crate 里。

use cnt_score::{NgramLm, WordId};

use crate::mmap::CntLm;

impl NgramLm for CntLm {
    fn word_index(&self, word: &str) -> Option<WordId> {
        Self::word_index(self, word)
    }

    fn unigram_by_id(&self, id: WordId) -> Option<(f32, f32)> {
        self.unigram_by_idx(id)
    }

    fn bigram_by_id(&self, prev: WordId, cur: WordId) -> Option<f32> {
        self.bigram_by_idx(prev, cur)
    }

    fn unigram(&self, word: &str) -> Option<(f32, f32)> {
        Self::unigram(self, word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{build, Bigram, Unigram};

    /// 端口实现与具体类型的方法语义一致，且 Katz backoff 走端口默认实现。
    #[test]
    fn port_matches_inherent_methods() {
        let unigrams = vec![
            Unigram { word: "在".into(), logprob: -2.0, backoff: -0.5 },
            Unigram { word: "现".into(), logprob: -3.0, backoff: -0.4 },
            Unigram { word: "看".into(), logprob: -4.0, backoff: -0.3 },
        ];
        let bigrams = vec![Bigram { w1: "现".into(), w2: "在".into(), logprob: -0.1 }];
        let bytes = build(&unigrams, &bigrams).unwrap();
        let path = std::env::temp_dir().join(format!("cnt-lm-port-{}.cntl", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let lm = CntLm::open(&path).unwrap();

        let xian = NgramLm::word_index(&lm, "现").unwrap();
        let zai = NgramLm::word_index(&lm, "在").unwrap();
        let kan = NgramLm::word_index(&lm, "看").unwrap();
        assert_eq!(NgramLm::unigram(&lm, "在"), Some((-2.0, -0.5)));
        assert_eq!(lm.unigram_by_id(zai), Some((-2.0, -0.5)));

        // 登录的 bigram：直接用条件概率
        assert!((lm.conditional(Some(xian), Some(zai), -12.0) - (-0.1)).abs() < 1e-6);
        // 未登录：Katz backoff = logP(看) + backoff(现) = -4.0 + -0.4
        assert!((lm.conditional(Some(xian), Some(kan), -12.0) - (-4.4)).abs() < 1e-6);
        // 未登录词：按 UNK 处理
        assert!((lm.conditional(Some(xian), None, -12.0) - (-12.4)).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }
}
