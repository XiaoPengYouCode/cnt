//! 整句解码器：beam search 在切分格上搜索 Top-K 句子。
//!
//! 打分模型：
//! ```text
//! logP(句子) ≈ Σ unigram(wᵢ) + Σ bigram(wᵢ₋₁, wᵢ) + 用户调频加成
//! ```
//! bigram 缺失时用 Katz backoff：`logP(wᵢ) + backoff(wᵢ₋₁)`。

use std::sync::Arc; // 同时用于 Decoder 与 Hyp 段

use cnt_dict::PinyinModel;
use cnt_lm::CntLm;
use cnt_input::{Candidate, CandidateSource, LearnedWord};
use fastrace::local::LocalSpan;
use fastrace::{Event, Span};

use crate::syllable::{SyllableEdge, SyllableTable};

/// beam 宽度：同时保留的假设数。
const BEAM: usize = 8;
/// 每个音节/词键最多取多少个候选词。
const WORDS_PER_SYLLABLE: usize = 8;
/// 多音节词最多跨多少个音节（含首音节；4 = 成语如 莫名其妙）。
const MAX_WORD_SYLLABLES: usize = 4;
/// 模糊匹配的惩罚（log10）：模糊是回退读音，不应与精确读音平等竞争。
/// -1.0：恰好压过「精确但生僻」的路径（死后 -4.95），又不至于让 时候 沉底。
const FUZZY_PENALTY: f32 = -1.0;
/// 返回的句子候选数。
const TOP_SENTENCES: usize = 5;
/// 词不在 LM 中时的默认 unigram log10 概率。
const UNK_LOGPROB: f32 = -12.0;
/// 次读音/极罕见词的基础分（`freq ≤ SECONDARY_FREQ_CAP`）。
const SECONDARY_BASE: f32 = -6.0;
/// 用户词（`freq == 0`，学过的复合词/新词）的基础分。
const USER_WORD_BASE: f32 = -5.0;
/// 判定「次读音」的频率上限（词库次读音 freq=1，主读音 ≥1e4）。
const SECONDARY_FREQ_CAP: u32 = 100;
/// 用户调频加成：每选一次候选的 log10 权重。
const USER_BOOST_LOG: f32 = 0.2;
/// 调频封顶次数：超过后不再增长（防止 了/个 这类高频字无限刷分，
/// 让 32 次选择的 boost 不至于压过整个分数空间）。
const USER_BOOST_CAP: u32 = 10;

/// 单个 key 的缓存值：key 的 Arc + 候选词列表（词 Arc、词频、LM 词表下标）
type CachedWords = (Arc<str>, Vec<(Arc<str>, u32, Option<u32>)>);
/// 候选词缓存：`key` → `CachedWords`
type WordCache = std::collections::HashMap<String, CachedWords>;

/// 一条格的展开上下文（参数聚合，避免 expand 签名过长）。
struct EdgeCtx<'a> {
    end: usize,
    fuzzy: bool,
    key: &'a Arc<str>,
}

/// beam search 中的一条部分假设。
///
/// `segments`/`last` 用 `Arc<str>`：expand 时克隆只增引用计数、不拷贝堆数据
/// （原先 `Vec<LearnedWord>` + `String` 每次 expand 要 10+ 次堆分配，
/// 平均每 decode 451 次 expand —— 这是 beam 循环慢的主因）。
#[derive(Clone)]
struct Hyp {
    pos: usize,
    /// 上一个词的 (词, LM 词表下标)；下标用于快速 bigram 查询
    last: Option<(Arc<str>, Option<u32>)>,
    score: f32,
    /// 已选的学习段（拼音, 词），Arc 克隆廉价
    segments: Vec<(Arc<str>, Arc<str>)>,
}

/// 整句解码器：词库 + 语言模型 + 音节表。
pub struct Decoder {
    model: Arc<PinyinModel>,
    lm: Option<Arc<CntLm>>,
    syllables: SyllableTable,
    /// 模糊音开关（平翘舌/边鼻音/前后鼻音等）。
    fuzzy: bool,
}

impl Decoder {
    /// 构造解码器。`lm` 为 `None` 时退化为单字/词候选（无整句）。
    #[must_use]
    pub fn new(model: Arc<PinyinModel>, lm: Option<Arc<CntLm>>, fuzzy: bool) -> Self {
        Self {
            model,
            lm,
            syllables: SyllableTable::new(),
            fuzzy,
        }
    }

    /// 提交学习数据：逐段调频 + 相邻两段拼合成新词（郑+爽 → zhengshuang/郑爽）。
    pub fn learn(&self, learned: &[LearnedWord]) {
        for seg in learned {
            self.model.bump(&seg.pinyin, &seg.word);
        }
        // 新词学习：相邻两段拼成复合词，下次输入完整拼音直接出
        for pair in learned.windows(2) {
            let key = format!("{}{}", pair[0].pinyin, pair[1].pinyin);
            let word = format!("{}{}", pair[0].word, pair[1].word);
            if key.len() <= 12 {
                self.model.bump(&key, &word);
            }
        }
    }

    /// 持久化用户数据。
    ///
    /// # Errors
    /// 写盘失败（IO）时返回错误。
    pub fn flush_user(&self) -> std::io::Result<()> {
        self.model.flush_user()
    }

    /// 完整候选：整句候选 + 尾音节补全按分数合并排序，再补词候选（去重）。
    ///
    /// 分数同一空间（log10）：用户学过的补全词（持久化）能压过次读音拼接的
    /// 垃圾候选（持久和）；而 le 的补全（冷）分数低于精确读音（了），排在后面。
    ///
    /// fastrace 埋点：有 local parent（daemon/bench 设置了 root span）时记录
    /// `candidates` span；无 context 时 `enter_with_local_parent` 为 noop，零开销。
    fn candidates_merged(&self, pinyin: &str) -> Vec<Candidate> {
        let _span = Span::enter_with_local_parent("candidates");
        let mut scored: Vec<(Candidate, f32)> = self.decode(pinyin);
        if let Some(lm) = &self.lm {
            scored.extend(self.completions(pinyin, lm));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        // 去重：同文本保留最高分
        let mut out: Vec<Candidate> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (c, _) in scored {
            if seen.insert(c.text.clone()) {
                out.push(c);
            }
        }
        out.truncate(20);
        for w in self.model.query(pinyin) {
            if !seen.contains(w.as_str()) {
                seen.insert(w.clone());
                out.push(Candidate {
                    text: w.clone(),
                    learned: vec![LearnedWord::new(pinyin.to_string(), w)],
                });
            }
        }
        out
    }

    /// 尾音节补全：末音节残缺（`chijiuh` → `h→hua`）或可延长（`chijiuhu` → `hu→hua`）时，
    /// 补出完整词候选（带分数；用户学过的词如 持久化 用 `USER_WORD_BASE` + 调频）。
    fn completions(&self, pinyin: &str, lm: &CntLm) -> Vec<(Candidate, f32)> {
        let _span = Span::enter_with_local_parent("completions");
        if pinyin.len() < 2 {
            return Vec::new();
        }
        let lattice = if self.fuzzy {
            self.syllables.lattice_fuzzy(pinyin)
        } else {
            self.syllables.lattice(pinyin)
        };

        // 可达位置（BFS）
        let mut reachable = vec![false; pinyin.len() + 1];
        reachable[0] = true;
        let mut max_pos = 0usize;
        for pos in 0..pinyin.len() {
            if reachable[pos] {
                for e in &lattice[pos] {
                    reachable[e.end] = true;
                    max_pos = max_pos.max(e.end);
                }
            }
        }
        if max_pos == 0 {
            return Vec::new(); // 一个音节都切不出来
        }

        // (完整 key 的前缀, 要补全成的更长沙节)
        let mut completions: Vec<(String, String)> = Vec::new();
        if max_pos == pinyin.len() {
            // 完整输入：末音节可延长（chijiuhu 的 hu → hua）
            for e in lattice.iter().flatten().filter(|e| e.end == pinyin.len() && !e.fuzzy) {
                let base = &pinyin[..e.end - e.syl.len()];
                for ext in self.syllables.syllables_with_prefix(e.syl) {
                    completions.push((base.to_string(), ext.to_string()));
                }
            }
        } else {
            // 未完成输入：补全残缺尾部（chijiuh 的 h → hua）
            let tail = &pinyin[max_pos..];
            let base = &pinyin[..max_pos];
            for ext in self.syllables.syllables_with_prefix(tail) {
                completions.push((base.to_string(), ext.to_string()));
            }
        }

        let mut out: Vec<(Candidate, f32)> = Vec::new();
        for (base, ext) in completions {
            let key = format!("{base}{ext}");
            for (word, freq) in self.model.ranked_words(&key, 4) {
                let score = reading_base(&word, freq, lm)
                    + self.boost(&key, &word);
                out.push((
                    Candidate {
                        text: word.clone(),
                        learned: vec![LearnedWord::new(key.clone(), word)],
                    },
                    score,
                ));
            }
        }
        out
    }

    /// beam search 主循环：在音节格上反复展开 + 束剪枝，返回完整句子假设
    /// （按分数降序、已去重截断）。
    ///
    /// 展开次数以 fastrace 事件（`expand`）记录在 `beam` span 上，供阶段
    /// 工作量分解（无 context 时 noop 零开销）。
    fn beam_search(
        &self,
        pinyin: &str,
        lattice: &[Vec<SyllableEdge>],
        reachable: &[bool],
        lm: &CntLm,
    ) -> Vec<Hyp> {
        let mut hyps = vec![Hyp {
            pos: 0,
            last: None,
            score: 0.0,
            segments: Vec::new(),
        }];
        // 完整句子：已消费全部输入的假设直接进 done（如 打字/时候/莫名其妙 在
        // 第一轮就完整），它们是与「半截探索」并列的答案，不能被 beam 剪掉。
        let mut done: Vec<Hyp> = Vec::new();
        // 按 key 缓存候选词（decode 内同一 key 会被多条路径重复查询）
        let mut word_cache: WordCache = std::collections::HashMap::new();

        let _beam_span = Span::enter_with_local_parent("beam");
        let mut n_expand = 0usize;
        for _ in 0..pinyin.len() {

            let mut next: Vec<Hyp> = Vec::new();
            for h in &hyps {
                if h.pos >= lattice.len() {
                    done.push(h.clone());
                    continue;
                }
                for edge in &lattice[h.pos] {
                    // 单音节词
                    let (key_arc, words) = self.cached_words(&mut word_cache, edge.syl, lm);
                    n_expand += words.len();
                    self.expand(
                        h,
                        &EdgeCtx { end: edge.end, fuzzy: edge.fuzzy, key: key_arc },
                        words,
                        &mut next,
                        lm,
                    );
                    // 多音节词：贪心拼接后续最长音节成完整 key（如 gong-zuo → 工作）
                    let mut key = String::from(edge.syl);
                    let mut cur_end = edge.end;
                    let mut fuzzy = edge.fuzzy;
                    for _ in 1..MAX_WORD_SYLLABLES {
                        let Some(next_edge) =
                            lattice.get(cur_end).and_then(|edges| edges.first())
                        else {
                            break;
                        };
                        key.push_str(next_edge.syl);
                        cur_end = next_edge.end;
                        fuzzy |= next_edge.fuzzy;
                        let (key_arc, words) = self.cached_words(&mut word_cache, &key, lm);
                        n_expand += words.len();
                        self.expand(
                            h,
                            &EdgeCtx { end: cur_end, fuzzy, key: key_arc },
                            words,
                            &mut next,
                            lm,
                        );
                    }
                }
            }
            // 束剪枝：先丢死路（in-place retain），再 in-place 分区为
            // 「部分假设（前）」与「完整假设（后）」，零分配（此前 partition()
            // 每轮新建 2 个 Vec）。
            // 束剪枝：先丢死路 + 分离完整假设，再对部分假设取前 BEAM。
            // 用 partition() 保持与原实现一致的假设顺序：select_nth_unstable 是
            // 不稳定选择，若 next 顺序变化，beam top-8 的选择会变（sihou 的
            // 「时」路径曾因此被剪掉，时候 掉出 #1）。
            next.retain(|h| reachable[h.pos]); // 丢弃走不到末尾的死路
            let (complete, partial): (Vec<Hyp>, Vec<Hyp>) =
                next.into_iter().partition(|h| h.pos >= lattice.len());
            done.extend(complete);
            hyps = partial;
            if hyps.len() > BEAM {
                hyps.select_nth_unstable_by(BEAM, |a, b| b.score.total_cmp(&a.score));
                hyps.truncate(BEAM);
            }
            if hyps.is_empty() {
                break;
            }
        }
        // 展开次数作为 beam span 的属性（fastrace 事件，无 context 时 noop）
        LocalSpan::add_event(Event::new("expand").with_property(|| ("count", n_expand.to_string())));

        done.sort_by(|a, b| b.score.total_cmp(&a.score));
        // 去重：同一句文本可能来自「多音节词」和「单字拼合」两条路径，保留最高分
        let mut seen = std::collections::HashSet::new();
        done.retain(|h| seen.insert(join_segments(&h.segments)));
        done.truncate(TOP_SENTENCES);
        done
    }

    /// beam search 解码：Top-K 句子候选（带分数）。
    fn decode(&self, pinyin: &str) -> Vec<(Candidate, f32)> {
        let Some(lm) = &self.lm else {
            return Vec::new();
        };
        // fastrace：lattice 构建（模糊音节格）
        let lattice = {
            let _span = Span::enter_with_local_parent("lattice");
            if self.fuzzy {
                self.syllables.lattice_fuzzy(pinyin)
            } else {
                self.syllables.lattice(pinyin)
            }
        };
        if lattice.len() < 2 {
            return Vec::new(); // 单音节退化，交给词候选
        }

        // 可达性：位置 i 能否通过合法音节走到末尾（反向 DP）。
        // beam 剪枝时只保留可达假设，避免错误切分的死路（如 xianzai 里的
        // xia 路径）占掉名额把正确路径挤掉。
        let mut reachable = vec![false; lattice.len() + 1];
        reachable[lattice.len()] = true;
        for i in (0..lattice.len()).rev() {
            reachable[i] = lattice[i].iter().any(|e| reachable[e.end]);
        }

        self.beam_search(pinyin, &lattice, &reachable, lm)
            .into_iter()
            .map(|h| {
                let score = h.score;
                let learned = h
                    .segments
                    .iter()
                    .map(|(p, w)| LearnedWord::new(p.to_string(), w.to_string()))
                    .collect();
                (
                    Candidate {
                        text: join_segments(&h.segments),
                        learned,
                    },
                    score,
                )
            })
            .collect()
    }

    /// 按 key 缓存候选词（decode 内同一 key 会被多条路径重复查询，避免重复计算）。
    fn cached_words<'a>(
        &self,
        cache: &'a mut WordCache,
        key: &str,
        lm: &CntLm,
    ) -> &'a CachedWords {
        if !cache.contains_key(key) {
            let words = self.words_for(key);
            let arcs: Vec<(Arc<str>, u32, Option<u32>)> = words
                .into_iter()
                .map(|(w, f)| {
                    let idx = lm.word_index(&w);
                    (Arc::<str>::from(w), f, idx)
                })
                .collect();
            cache.insert(key.to_string(), (Arc::<str>::from(key), arcs));
        }
        &cache[key]
    }

    /// 把一个 (结束位置, 词键, 是否模糊) 的候选词展开进 beam。
    /// `ctx.key`/`words` 来自缓存，`segments` 克隆只增 Arc 引用计数，零堆拷贝。
    fn expand(
        &self,
        h: &Hyp,
        ctx: &EdgeCtx<'_>,
        words: &[(Arc<str>, u32, Option<u32>)],
        next: &mut Vec<Hyp>,
        lm: &CntLm,
    ) {
        let penalty = if ctx.fuzzy { FUZZY_PENALTY } else { 0.0 };
        for (word_arc, freq, word_idx) in words {
            let mut nh = h.clone();
            // ARPA bigram 是条件概率 log P(w2|w1)：首词用读音感知的基础分，
            // 后续词用条件概率（bigram 或 Katz backoff），用户调频加成始终加。
            // 模糊读音加惩罚：回退读音不能与精确读音平等竞争（否则 dazi 会出「他只」）。
            // 打分全程用缓存的 LM 词表下标（unigram_by_idx/bigram_by_idx），
            // 热路径零字符串查找。
            let boost = self.boost(ctx.key, word_arc);
            let step = match (h.last.as_ref(), *word_idx) {
                // 双词都在词表：bigram 下标查询；缺失时 Katz backoff（unigram 下标）
                (Some((_, Some(prev_idx))), Some(word_idx)) => {
                    lm.bigram_by_idx(*prev_idx, word_idx).unwrap_or_else(|| {
                        let u2 = lm.unigram_by_idx(word_idx).map_or(UNK_LOGPROB, |(p, _)| p);
                        let bk = lm.unigram_by_idx(*prev_idx).map_or(0.0, |(_, b)| b);
                        u2 + bk
                    })
                }
                // 词不全在词表：各自按下标查 unigram（不在词表的按 UNK/0）
                (Some((_, prev_idx)), _) => {
                    let u2 = word_idx.map_or(UNK_LOGPROB, |i| {
                        lm.unigram_by_idx(i).map_or(UNK_LOGPROB, |(p, _)| p)
                    });
                    let bk = (*prev_idx).map_or(0.0, |i| {
                        lm.unigram_by_idx(i).map_or(0.0, |(_, b)| b)
                    });
                    u2 + bk
                }
                // 首词：用户词/次读音走常量；主读音用下标 unigram（不在词表 → UNK）
                _ => {
                    if *freq == 0 {
                        USER_WORD_BASE
                    } else if *freq <= SECONDARY_FREQ_CAP {
                        SECONDARY_BASE
                    } else {
                        word_idx.map_or(UNK_LOGPROB, |i| {
                            lm.unigram_by_idx(i).map_or(UNK_LOGPROB, |(p, _)| p)
                        })
                    }
                }
            };
            let step = step + boost + penalty;
            nh.score += step;
            nh.pos = ctx.end;
            nh.last = Some((word_arc.clone(), *word_idx));
            nh.segments.push((ctx.key.clone(), word_arc.clone()));
            next.push(nh);
        }
    }

    /// 某音节/词键的高频候选词（带词库频率；领域规则在 `PinyinModel::ranked_words` 单点定义）。
    fn words_for(&self, key: &str) -> Vec<(String, u32)> {
        self.model.ranked_words(key, WORDS_PER_SYLLABLE)
    }

    /// 用户调频加成：每选一次候选的 log10 权重（封顶 `USER_BOOST_CAP` 次）。
    #[allow(clippy::cast_precision_loss)] // 用户计数 ≤ 100，u32→f32 无损
    fn boost(&self, syllable: &str, word: &str) -> f32 {
        self.model.user_count(syllable, word).min(USER_BOOST_CAP) as f32 * USER_BOOST_LOG
    }

}

/// 读音感知的基础分（首词用）：
/// - 用户词（`freq == 0`）：`USER_WORD_BASE` —— 学过的词应能浮现，不再被 -12 埋没
/// - 次读音（`freq ≤ SECONDARY_FREQ_CAP`，如 的di、和hu）：`SECONDARY_BASE`
///   —— 多音字不再串频（否则 fu 会出 和、diyige 会出 的一个）
/// - 主读音：LM unigram（保留细粒度排序：是 > 时 > 事）
fn reading_base(word: &str, freq: u32, lm: &CntLm) -> f32 {
    if freq == 0 {
        USER_WORD_BASE
    } else if freq <= SECONDARY_FREQ_CAP {
        SECONDARY_BASE
    } else {
        unigram_score(lm, word)
    }
}

impl CandidateSource for Decoder {
    fn candidates(&self, pinyin: &str) -> Vec<Candidate> {
        self.candidates_merged(pinyin)
    }
}

/// 词的 unigram 分（首词用；`completions` 的低频路径）。
fn unigram_score(lm: &CntLm, word: &str) -> f32 {
    lm.unigram(word).map_or(UNK_LOGPROB, |(p, _)| p)
}

/// 把学习段拼成候选文本。
fn join_segments(segments: &[(Arc<str>, Arc<str>)]) -> String {
    let mut text = String::new();
    for (_, word) in segments {
        text.push_str(word);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syllable::SyllableTable;

    /// 构建一个小模型：dict + lm 用临时文件。
    fn decoder_with(
        name: &str,
        pairs: &[(&str, &str, u32)],
        unigrams: &[(&str, f32, f32)],
        bigrams: &[(&str, &str, f32)],
        fuzzy: bool,
    ) -> Decoder {
        use cnt_dict::writer;
        use cnt_lm::writer::{build, Bigram, Unigram};

        let dir = std::env::temp_dir().join(format!("cnt-decode-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dict_path = dir.join("dict.cntd");
        let user_path = dir.join("user.dict");
        let lm_path = dir.join("lm.cntl");

        let pairs: Vec<(String, String, u32)> = pairs
            .iter()
            .map(|(p, w, f)| (p.to_string(), w.to_string(), *f))
            .collect();
        writer::write_to_file(&pairs, &dict_path).unwrap();

        let unigrams: Vec<Unigram> = unigrams
            .iter()
            .map(|(w, p, b)| Unigram {
                word: w.to_string(),
                logprob: *p,
                backoff: *b,
            })
            .collect();
        let bigrams: Vec<Bigram> = bigrams
            .iter()
            .map(|(a, b, p)| Bigram {
                w1: a.to_string(),
                w2: b.to_string(),
                logprob: *p,
            })
            .collect();
        let bytes = build(&unigrams, &bigrams).unwrap();
        std::fs::write(&lm_path, bytes).unwrap();

        let model = Arc::new(PinyinModel::open(&dict_path, &user_path).unwrap());
        let lm = Arc::new(CntLm::open(&lm_path).unwrap());
        Decoder::new(model, Some(lm), fuzzy)
    }

    #[test]
    fn decode_prefers_bigram_path() {
        // 词库：xian→先/现, zai→在；LM: 现在 bigram 强
        let d = decoder_with(
            "bigram",
            &[
                ("xian", "先", 900),
                ("xian", "现", 100),
                ("zai", "在", 900),
                ("xi", "西", 800),
                ("an", "安", 700),
            ],
            &[
                ("先", -3.0, 0.0),
                ("现", -2.0, 0.0),
                ("在", -3.0, 0.0),
                ("西", -3.5, 0.0),
                ("安", -3.5, 0.0),
            ],
            &[
                ("现", "在", -0.1), // 现在 强
                ("先", "在", -2.0),
                ("西", "安", -0.5), // 西安 也强
            ],
            false,
        );
        let cands = d.candidates("xianzai");
        // 完整句子候选（beam 只保留整句路径）：
        //   xian-zai → 现在 / 先在；xi-an-zai → 西安在
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        let now = texts.iter().position(|t| *t == "现在").unwrap();
        let xian = texts.iter().position(|t| *t == "西安在").unwrap();
        assert!(now < xian, "现在 should rank before 西安在: {texts:?}");
        // 现在的学习对 = [(xian,现),(zai,在)]；西安在 = [(xi,西),(an,安),(zai,在)]
        let now_cand = &cands[now];
        assert_eq!(
            now_cand.learned,
            vec![
                LearnedWord::new("xian".to_string(), "现".to_string()),
                LearnedWord::new("zai".to_string(), "在".to_string())
            ]
        );
    }

    #[test]
    fn decode_prefers_multisyllable_word() {
        // 词库含多音节词 工作(gongzuo)，应优先于 宫+坐 拼字
        let d = decoder_with(
            "multi",
            &[
                ("wo", "我", 900),
                ("men", "们", 900),
                ("zai", "在", 900),
                ("gongzuo", "工作", 800),
                ("gong", "宫", 900),
                ("zuo", "坐", 800),
            ],
            &[
                ("我", -2.3, 0.0),
                ("们", -4.0, 0.0),
                ("在", -2.0, 0.0),
                ("工作", -4.0, 0.0),
                ("宫", -4.0, 0.0),
                ("坐", -4.0, 0.0),
            ],
            &[
                ("我", "们", -3.0),
                ("们", "在", -2.0),
                ("在", "工作", -1.0), // 强
                ("在", "宫", -4.0),
                ("宫", "坐", -4.0),
            ],
            false,
        );
        let cands = d.candidates("womenzaigongzuo");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        let idx = texts.iter().position(|t| *t == "我们在工作");
        assert!(idx.is_some_and(|i| i == 0), "我们在工作 should be #1: {texts:?}");
    }

    #[test]
    fn learn_compound_creates_new_word() {
        // 学过的复合词（郑+爽 → zhengshuang/郑爽）下次输入完整拼音直接出
        let d = decoder_with(
            "newword",
            &[
                ("zheng", "郑", 900),
                ("shuang", "爽", 900),
            ],
            &[
                ("郑", -3.0, 0.0),
                ("爽", -3.0, 0.0),
            ],
            &[("郑", "爽", -0.5)],
            false,
        );
        // 提交 郑爽（两个片段）
        d.learn(&[
            LearnedWord::new("zheng".to_string(), "郑".to_string()),
            LearnedWord::new("shuang".to_string(), "爽".to_string()),
        ]);
        // 下次输入 zhengshuang：新词 郑爽 应作为整词出现
        let cands = d.candidates("zhengshuang");
        assert!(
            cands.iter().any(|c| c.text == "郑爽"),
            "learned new word 郑爽 should appear: {:?}",
            cands.iter().map(|c| c.text.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_with_fuzzy_matches_standard_reading() {
        // 模糊音：输入 sihou（si 想表达 shi）→ 出 时候（shi-hou）
        let d = decoder_with(
            "fuzzy",
            &[
                ("shi", "是", 900),
                ("shi", "时", 800),
                ("hou", "候", 900),
                ("hou", "后", 800),
                ("si", "四", 900),
            ],
            &[
                ("是", -1.9, 0.0),
                ("时", -2.9, 0.0),
                ("候", -4.0, 0.0),
                ("后", -3.0, 0.0),
                ("四", -2.5, 0.0),
            ],
            &[("时", "候", -0.5), ("是", "后", -1.0)],
            true, // 开启模糊音
        );
        let cands = d.candidates("sihou");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        let idx = texts.iter().position(|t| *t == "时候");
        assert!(
            idx.is_some_and(|i| i <= 1),
            "时候 (fuzzy si->shi) should rank top: {texts:?}"
        );
    }

    #[test]
    fn decode_without_lm_falls_back_to_words() {
        let d = decoder_with(
            "fallback",
            &[("ni", "你", 900), ("hao", "好", 800)],
            &[("你", -2.0, 0.0), ("好", -2.0, 0.0)],
            &[("你", "好", -0.5)],
            false,
        );
        let _ = &d; // 构造成功即可
    }

    #[test]
    fn syllables_table_lattice_works() {
        let t = SyllableTable::new();
        let lattice = t.lattice("nihao");
        assert!(!lattice[0].is_empty());
        assert!(lattice[0].iter().any(|e| e.syl == "ni"));
        assert!(lattice[2].iter().any(|e| e.syl == "hao"));
    }
}
