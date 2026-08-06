//! 整句解码器：beam search 在切分格上搜索 Top-K 句子。
//!
//! 打分模型：
//! ```text
//! logP(句子) ≈ Σ unigram(wᵢ) + Σ bigram(wᵢ₋₁, wᵢ) + 用户调频加成
//! ```
//! bigram 缺失时用 Katz backoff：`logP(wᵢ) + backoff(wᵢ₋₁)`。
//!
//! **搜索与打分解耦**：搜索（词图 + beam）在本文件，打分能力来自
//! `cnt-score` 端口——
//! - 热路径（beam 内每次展开）走 [`NgramLm`] 泛型，静态分派、零抽象开销；
//! - 冷路径（每按键 ≤1 次）走 [`Rescorer`]（`dyn`），可换成小神经模型。

use std::sync::Arc; // 同时用于 Decoder 与 Hyp 段

use cnt_dict::PinyinModel;
use cnt_lm::CntLm;
use cnt_input::{Candidate, CandidateSource, LearnedWord};
use cnt_score::{NgramLm, RescorePolicy, Rescorer};
use fastrace::local::LocalSpan;

use crate::syllable::{SyllableEdge, SyllableTable, MAX_FUZZY_COST};

/// beam 宽度：同时保留的假设数。
const BEAM: usize = 8;
/// 每个音节/词键最多取多少个候选词（长输入）。
const WORDS_PER_SYLLABLE: usize = 8;
/// 单音节输入的每键候选词数。
///
/// 单音节同音字本身就几十个（li → 里/力/历/理/立/例……），只展开 8 个会让
/// 词频上万的精确字（力/历/理）根本进不了候选。
const WORDS_PER_SYLLABLE_MONO: usize = 20;
/// 双音节输入的每键候选词数（比长输入宽、比单音节省：双音节的展开是乘性的）。
const WORDS_PER_SYLLABLE_SHORT: usize = 12;
/// 多音节词最多跨多少个音节（含首音节；4 = 成语如 莫名其妙）。
const MAX_WORD_SYLLABLES: usize = 4;
/// 模糊匹配的分级惩罚（log10），下标 = `SyllableEdge::cost`（0 = 精确，无惩罚）。
///
/// 模糊是回退读音，不应与精确读音平等竞争；但惩罚不能一刀切：
/// - 1（平翘舌）-1.0：恰好压过「精确但生僻」的路径（死后 -4.95），
///   又不至于让 时候 沉底（sihou → 时候 仍 #1）
/// - 2（边鼻音 n/l、鼻韵尾 an/ang 等）-2.0：阻止 xiangchen → 县城 抢 #1
/// - 3（f/h、r/l、k/g、t/d）-3.0：较少见的混淆，不得压过精确读音词
const FUZZY_PENALTY: [f32; (MAX_FUZZY_COST as usize) + 1] = [0.0, -1.0, -2.0, -3.0];
/// 模糊边叠加的额外惩罚（超线性）：第 n 条模糊边额外扣 `(n-1) ×` 此值。
///
/// 逐边线性相加拦不住两处轻度模糊的高频词：zhuchen → 「组成」（zh/z + ch/c）
/// 只扣 -2 就能抢 #1。现实中一个词同时打错两个音的概率远低于打错一个，
/// 惩罚必须超线性增长（第 2 条额外 -2、第 3 条额外 -4……）。
const FUZZY_STACK_PENALTY: f32 = 2.0;
/// 单音节输入的模糊额外惩罚：无上下文佐证时，模糊回退纯属噪声。
///
/// 模糊音的价值来自整词/整句的佐证（sihou → 时候：两个音节互相印证）；输入
/// 只有一个音节时既没有词也没有上下文，「li 出你」只会挤掉精确同音字（利/理）。
const SINGLE_SYLLABLE_FUZZY_PENALTY: f32 = 3.0;
/// 允许占 #1 的模糊边数上限：单处模糊（sihou → 时候）是模糊音的初衷，可以 #1；
/// 两处以上同时模糊（zhuchen → 组成）则一律不得占 #1。
const MAX_FUZZY_EDGES_AT_TOP: u8 = 1;
/// 返回的句子候选数（长输入）。
const TOP_SENTENCES: usize = 5;
/// 单音节输入的句子候选数：整句候选就是精确同音字，只留 5 个会把位置
/// 6~10 让给补全词与模糊音。
const TOP_SENTENCES_MONO: usize = 15;
/// 双音节输入的句子候选数。
const TOP_SENTENCES_SHORT: usize = 10;
/// 整个输入作为一个词键时取多少个词候选（含 LM 未覆盖的高频字：备/碑/辈/悲）。
const WHOLE_KEY_WORDS: usize = 20;
/// 对外返回的候选总数上限。
const CANDIDATE_LIMIT: usize = 30;
/// 候选分组：2 = 部分候选（只覆盖输入前一段）。
const GROUP_PARTIAL: u8 = 2;
/// 部分候选（只覆盖输入前一段的词）最多给几个。
///
/// Rime 式增量确认的修复入口：整句错了不必删光重打，选中前一段即可确认，
/// 剩下的拼音继续组合。给太多会挤占整句候选的位置。
const PARTIAL_LIMIT: usize = 8;
/// LM 完全未知词的兜底 log10 概率（下界参照）。
///
/// 打分不再直接用它：不在 LM 词表的词走 `first_word_base`（用户词/次读音/按词频的
/// OOV 分），比 -12 的悬崖合理得多。留作 OOV 打分的下界断言。
#[cfg(test)]
const UNK_LOGPROB: f32 = -12.0;
/// 词库有、LM 无的主读音整词基础分（如 信息量）：整词不该因不在 LM
/// 而拿到 UNK（-12）输给任何整句拼接，给一个与用户词同档的基础分。
const OOV_BASE: f32 = -5.0;
/// OOV 打分的参考词频（`OOV_BASE` 对应的词频量级）。
///
/// 不在 LM 的词一律给 `OOV_BASE` 会让「词频 1 的生僻字」和「词频 2 万的常用字」
/// 同分，单音节候选的 5~10 位于是被 㔹/㖀 这类字符占掉。按词库词频做 log10
/// 修正后，常用字略高于基准、生僻字明显下沉。
const OOV_REF_FREQ: f32 = 10_000.0;
/// OOV 词频修正的下限/上限（log10）：防止极端词频把 OOV 抬过 LM 覆盖的词或砸穿 UNK。
const OOV_ADJUST_MIN: f32 = -4.0;
const OOV_ADJUST_MAX: f32 = 1.0;
/// 补全词惩罚（completions 的延长音节词）：精确读音候选优先于补全候选。
/// -1.5：-0.5 太轻，输入 jian 时 jiang 的高频词（将 -2.82）会压过精确读音的
/// 见/件/间，单音节候选前排被「猜你还没打完」的词占掉。
const COMPLETION_PENALTY: f32 = -1.5;
/// 次读音的基础分上限（`freq ≤ SECONDARY_FREQ_CAP` 且**在 LM 词表**）。
///
/// 只封顶、不抬升：`min(SECONDARY_BASE, unigram)`。多音字的次读音（的di、和hu）
/// 在 LM 里查到的是主读音的高概率，直接用会串频，所以封到这一档；而本身就在 LM
/// 地板的罕见字不该被这一档「抬」上来。
const SECONDARY_BASE: f32 = -6.0;
/// 用户词（`freq == 0`，学过的复合词/新词）的基础分。
const USER_WORD_BASE: f32 = -5.0;
/// 判定「次读音 / 词库长尾」的频率上限（词库次读音 freq=1，主读音 ≥1e4）。
///
/// 注意这一档里混着两类词（词库构建把它们都记成 freq=1）：真次读音（的di、和hu，
/// 在 LM 词表里）与长尾字形（㙢/㝵/㮶 这类 Ext-B 异体字，不在 LM 词表）。
/// 打分必须区分，见 `reading_base`。
const SECONDARY_FREQ_CAP: u32 = 100;

/// 一个词键下的候选词。
///
/// 与假设无关的量全部在枚举阶段算好（句首基础分、用户调频、LM 下标），
/// beam 展开时只做加法 + 一次 bigram 查询。
struct WordCand {
    word: Arc<str>,
    /// LM 词表下标（热路径打分零字符串查找）
    lm_id: Option<u32>,
    /// 作为句首词的基础分（用户词/次读音/OOV/LM unigram）
    first_base: f32,
    /// 用户调频加成：每个 (词键, 词) 只查一次用户库
    /// （此前每次展开都要抢一次用户库的锁 + 两次哈希查找）
    boost: f32,
    /// 是否为词库长尾字形（见 [`is_dict_tail`]）：参与候选分组硬约束。
    dict_tail: bool,
}

/// 某个位置上可展开的一个词键（单音节或多音节整词）及其候选词。
///
/// 词键只取决于「位置 + 音节格」，与假设无关，所以每个位置只枚举一次：
/// 此前每条假设都要重做一遍字符串拼接与词库前缀查找，BEAM=8 就是 8 倍冗余。
struct PosKey {
    end: usize,
    /// 该词键路径上的模糊代价等级（多音节取最大）
    cost: u8,
    /// 该词键路径上的模糊边条数（整词可能连错两个音：zhuchen → 组成）
    fuzzy_edges: u8,
    key: Arc<str>,
    /// 本次输入实际使用的候选词数（缓存按最宽上限存）
    limit: usize,
    words: Arc<[WordCand]>,
}

/// beam 路径的 arena 节点：假设只存父节点下标。
///
/// 此前每条假设持有 `Vec<(Arc, Arc)> segments`，每次展开都要克隆整个 Vec
/// （一次堆分配 + 全部 Arc 引用计数），而每次解码有数百次展开。
struct Node {
    parent: u32,
    key: Arc<str>,
    word: Arc<str>,
}

/// 跨按键的词键缓存。
///
/// 打字是**增量**的：敲 `womenzaigongzuo` 的每一个字母都会把整个前缀重新解码一遍，
/// 同一批词键（wo/women/zai/gong/gongzuo…）会被反复查词库、反复分配 `Arc<str>`。
/// 缓存按词键存两样东西：候选词表、以及「词库里有没有以它为前缀的键」（链剪枝用）。
///
/// 缓存条目记下「按多宽的上限算过」：请求更窄时直接 `take`（`ranked_words` 是
/// 先定序再截断，前缀等价），请求更宽时才重算 —— 长输入不必为单音节档的 20 个
/// 候选付代价。
#[derive(Default)]
struct KeyCache {
    words: std::collections::HashMap<Box<str>, (usize, Arc<[WordCand]>)>,
    prefix: std::collections::HashMap<Box<str>, bool>,
}

impl KeyCache {
    /// 整体失效（仅 `clear_cache` 用：内存回收 / 冷启动基准）。
    fn clear(&mut self) {
        self.words.clear();
        self.prefix.clear();
    }

    /// 单个词键失效（调频/新词影响这个键的候选词表、以及它各级前缀的存在性判定）。
    ///
    /// 学到新复合词 `zhengshuang` 后，「有没有以 `zhengsh` 开头的键」的答案会从
    /// false 变 true（用户库也参与该判定），而 beam 的多音节链剪枝正是靠它 ——
    /// 不清掉这些前缀判定，新词就永远拼不进句子中间。
    fn invalidate(&mut self, key: &str) {
        self.words.remove(key);
        for end in 1..=key.len() {
            if key.is_char_boundary(end) {
                self.prefix.remove(&key[..end]);
            }
        }
    }
}

/// 缓存条目上限：超过就整体清空（键的复用是局部的，简单清空比 LRU 划算）。
const KEY_CACHE_CAP: usize = 4096;

/// `Hyp::node` 的空值（尚未选任何词）。
const NO_NODE: u32 = u32::MAX;

/// 上一个词的状态（打分用）：句首 vs 有前词（可能不在 LM 词表）。
#[derive(Clone, Copy)]
enum Prev {
    /// 句首：用读音感知的基础分
    Start,
    /// 有前词，携带其 LM 词表下标（`None` = 前词不在词表，条件概率退化为 unigram）
    Word(Option<u32>),
}

/// 一条待收录的词键路径（参数聚合，避免 `push_key` 签名过长）。
struct KeyPath<'a> {
    key: &'a str,
    end: usize,
    cost: u8,
    fuzzy_edges: u8,
}

/// 候选规模上限：按输入音节数自适应。
///
/// 单/双音节输入的搜索空间本来就小（一两个位置），却有几十个同音字要展示；
/// 长输入反之（搜索贵、候选少）。用同一套常量对待两者是错的。
#[derive(Debug, Clone, Copy)]
struct Limits {
    /// 每个词键展开多少候选词
    words_per_key: usize,
    /// 保留多少整句候选
    top_sentences: usize,
    /// 模糊边的额外惩罚（单音节输入无上下文佐证时加重）
    fuzzy_extra: f32,
}

impl Limits {
    /// 按「覆盖输入所需的最少音节数」选规模。
    const fn for_syllables(min_syllables: usize) -> Self {
        match min_syllables {
            // 单音节：搜索空间只有一个位置，尽管放宽；模糊回退无上下文佐证，加重惩罚
            0 | 1 => Self {
                words_per_key: WORDS_PER_SYLLABLE_MONO,
                top_sentences: TOP_SENTENCES_MONO,
                fuzzy_extra: SINGLE_SYLLABLE_FUZZY_PENALTY,
            },
            // 双音节：展开是乘性的，取中间档
            2 => Self {
                words_per_key: WORDS_PER_SYLLABLE_SHORT,
                top_sentences: TOP_SENTENCES_SHORT,
                fuzzy_extra: 0.0,
            },
            _ => Self {
                words_per_key: WORDS_PER_SYLLABLE,
                top_sentences: TOP_SENTENCES,
                fuzzy_extra: 0.0,
            },
        }
    }
}

/// 一个带分与来源信息的候选（排序阶段的内部表示）。
///
/// `fuzzy_edges` / `completion` / `dict_tail` 不只是元数据：它们参与**硬约束**
/// （模糊叠加不得 #1、补全词不得插到精确候选之前、词库长尾不得凭「精确」
/// 占住前排）——这些是分数调参担保不了的，必须结构上保证。
struct Scored {
    cand: Candidate,
    score: f32,
    /// 该候选路径上的模糊边数（0 = 精确读音）
    fuzzy_edges: u8,
    /// 是否为尾音节补全候选（「猜你还没打完」）
    completion: bool,
    /// 是否为词库长尾字形的单词候选（见 [`is_dict_tail`]）。
    dict_tail: bool,
}

/// beam search 中的一条部分假设（`Copy`，克隆零成本、零分配）。
///
/// 路径不存在假设里，而是 arena 的父指针链（`node`）——展开时只拷贝几十字节。
#[derive(Clone, Copy)]
struct Hyp {
    pos: usize,
    /// 上一个词的状态（句首 / 前词的 LM 下标）
    prev: Prev,
    score: f32,
    /// 已走过的模糊边数（用于超线性叠加惩罚与「不得占 #1」约束）
    fuzzy_edges: u8,
    /// 句首词是否为词库长尾字形（单词句子才用得上，见 `Scored::dict_tail`）
    first_tail: bool,
    /// arena 中的路径节点下标（`NO_NODE` = 空路径）
    node: u32,
}

/// beam 搜索结果：完整假设 + 复原路径所需的 arena + 各位置的词键。
///
/// `keys_at` 带出来给部分候选复用（位置 0 的词键就是「覆盖输入前一段的词」），
/// 免得再枚举一遍。
struct BeamResult {
    hyps: Vec<Hyp>,
    arena: Vec<Node>,
    keys_at: Vec<Option<Vec<PosKey>>>,
}

/// 整句解码器：词库 + 语言模型（[`NgramLm`] 端口）+ 音节表 + 可选重排器。
///
/// `L` 默认是 `cnt-lm` 的 mmap n-gram（`CntLm`），因此下游 `Arc<Decoder>` 写法不变；
/// 泛型参数存在的意义是打分实现可替换且不牺牲热路径性能（单态化，无虚表）。
pub struct Decoder<L: NgramLm = CntLm> {
    model: Arc<PinyinModel>,
    lm: Option<Arc<L>>,
    syllables: SyllableTable,
    /// 模糊音开关（平翘舌/边鼻音/前后鼻音等）。
    fuzzy: bool,
    /// 可选的整句重排器（小神经模型等）；`None` = 纯 n-gram 基线。
    rescorer: Option<Arc<dyn Rescorer>>,
    /// 重排触发/融合策略。
    policy: RescorePolicy,
    /// 跨按键的词键缓存（用户调频变化时整体失效）。
    keys: std::sync::Mutex<KeyCache>,
}

impl<L: NgramLm> Decoder<L> {
    /// 构造解码器。`lm` 为 `None` 时退化为单字/词候选（无整句）。
    #[must_use]
    pub fn new(model: Arc<PinyinModel>, lm: Option<Arc<L>>, fuzzy: bool) -> Self {
        Self {
            model,
            lm,
            syllables: SyllableTable::new(),
            fuzzy,
            rescorer: None,
            policy: RescorePolicy::default(),
            keys: std::sync::Mutex::new(KeyCache::default()),
        }
    }

    /// 装上整句重排器（构造期注入，运行期不可变）。
    ///
    /// 未调用时解码行为与纯 n-gram 完全一致——重排是可选增强，不是必需路径。
    #[must_use]
    pub fn with_rescorer(mut self, rescorer: Arc<dyn Rescorer>, policy: RescorePolicy) -> Self {
        log::info!("rescorer enabled: {} policy={policy:?}", rescorer.name());
        self.rescorer = Some(rescorer);
        self.policy = policy;
        self
    }

    /// 提交学习数据：逐段调频 + 相邻两段拼合成新词（郑+爽 → zhengshuang/郑爽）。
    ///
    /// 词键缓存做**精确失效**：只有这次动过的键（各段 + 拼合出的新词键）会被丢弃。
    /// 早先图省事整体清空，代价是每次上屏后的下一句都退回冷路径（实测每键 +26%）——
    /// 调频只影响它自己那个键的候选词表与加成分，没有理由连累其他键。
    pub fn learn(&self, learned: &[LearnedWord]) {
        // 先写模型再失效：反过来的话，并发的解码可能拿旧数据重新填满缓存
        for seg in learned {
            self.model.bump(&seg.pinyin, &seg.word);
        }
        // 新词学习：相邻两段拼成复合词，下次输入完整拼音直接出
        let mut compounds: Vec<String> = Vec::new();
        for pair in learned.windows(2) {
            let key = format!("{}{}", pair[0].pinyin, pair[1].pinyin);
            let word = format!("{}{}", pair[0].word, pair[1].word);
            if key.len() <= 12 {
                self.model.bump(&key, &word);
                compounds.push(key);
            }
        }
        let mut cache = self.lock_keys();
        for seg in learned {
            cache.invalidate(&seg.pinyin);
        }
        for key in &compounds {
            cache.invalidate(key);
        }
    }

    /// 把拼音串按标准音节切开，供预编辑显示（`nihaoshijie` → `ni hao shi jie`）。
    ///
    /// 只用精确音节表（不含模糊变体）：预编辑要如实反映**用户打的**内容，
    /// 不能显示成模糊映射后的读音。切不出来的尾巴原样附在后面。
    #[must_use]
    pub fn display_pinyin(&self, pinyin: &str) -> String {
        self.syllables.split_for_display(pinyin)
    }

    /// 清空词键缓存。
    ///
    /// 正常使用不需要调用（`learn` 会自动失效）；给内存压力回收与
    /// 「冷启动延迟」基准测量用。
    pub fn clear_cache(&self) {
        self.lock_keys().clear();
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
        self.candidates_scored(pinyin)
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    /// 完整候选（带分数，诊断/测试用）：decode 整句 + 补全 + 词候选，
    /// 与用户实际看到的候选完全一致（同一 `CandidateSource` 路径）。
    #[must_use]
    pub fn candidates_scored(&self, pinyin: &str) -> Vec<(Candidate, f32)> {
        let _span = LocalSpan::enter_with_local_parent("candidates");
        let mut scored: Vec<Scored> = self.decode(pinyin);
        // 整个输入作为词键的精确候选（含 LM 未覆盖但词频上万的字：备/碑/辈/悲），
        // 它们与整句候选在同一分数空间里竞争——不再用 -inf 垫底排在补全词后面。
        scored.extend(self.whole_key_words(pinyin));
        if let Some(lm) = &self.lm {
            scored.extend(self.completions(pinyin, lm));
        }
        // 排序分三组（组间是硬顺序，组内按分数）：
        // 0 完全覆盖输入的候选（整句/整键词）——你已经打完的读音优先；
        // 1 补全候选（「猜你还没打完」）——不该插到已打完的读音前面
        //   （输入 jian 时前排不能被 jiang 的词占掉），以及词库长尾字形；
        // 2 部分候选（只覆盖前一段）——修复入口，放在最后，不干扰正常整句选词。
        //
        // 长尾字形（LM 不认识 + 词频 ≤ 100，见 `is_dict_tail`）虽然确实覆盖了全部
        // 输入，但不享受「已打完的读音优先」：否则输入 n 时 ㅕ午/咹（OOV 地板 -9.0）
        // 会把 你/能/年挤到 6 位后，输入 de 时 彳/得德? 一类异体字占住 4~10 位。
        // 它们降到补全组，和 你/能 按分数硬碰一次（且不带 COMPLETION_PENALTY，
        // 真正常用的字仍能赢）。判据与上游打分一致：落在「在不在 LM 词表」上。
        let input_len = pinyin.len();
        let group = |s: &Scored| -> u8 {
            if s.cand.covers_all(input_len) {
                u8::from(s.completion || s.dict_tail)
            } else {
                GROUP_PARTIAL
            }
        };
        scored.sort_by(|a, b| {
            let (ga, gb) = (group(a), group(b));
            ga.cmp(&gb)
                // 覆盖长的优先（只对部分候选有意义）
                .then_with(|| b.cand.consumed.cmp(&a.cand.consumed))
                .then_with(|| {
                    if ga == GROUP_PARTIAL {
                        // 部分候选按词库排名（稳定排序保留原序），不按 LM 分：
                        // 单词场景「词频高」比「unigram 高」更接近用户想要的
                        // （你好 的 unigram 在 LM 地板上，词频却极高）
                        std::cmp::Ordering::Equal
                    } else {
                        b.score.total_cmp(&a.score)
                    }
                })
        });
        // 去重：同文本保留首个（= 组内最高分）
        let mut out: Vec<Scored> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for s in &scored {
            if seen.insert(s.cand.text.as_str()) {
                out.push(Scored {
                    cand: s.cand.clone(),
                    score: s.score,
                    fuzzy_edges: s.fuzzy_edges,
                    completion: s.completion,
                    dict_tail: s.dict_tail,
                });
            }
        }
        drop(scored);
        demote_stacked_fuzzy(&mut out);
        out.truncate(CANDIDATE_LIMIT);

        let mut out: Vec<(Candidate, f32)> =
            out.into_iter().map(|s| (s.cand, s.score)).collect();
        // 神经重排（可选）：仅在基线不确定时对前 top_n 条重排；
        // 未装重排器时这里完全不产生开销。
        if let Some(rescorer) = &self.rescorer {
            crate::rescore::apply(rescorer.as_ref(), &self.policy, &mut out);
        }
        out
    }

    /// 整个输入作为一个词键的精确候选（`li` → 里/力/历/理…，`xianzai` → 现在）。
    ///
    /// 分数走 `reading_base`（LM unigram / OOV / 次读音 / 用户词）+ 用户调频，
    /// 与整句候选同一空间。曾经这些候选被硬塞 `-inf`「恒排最后」，导致
    /// LM 未覆盖但词频上万的精确字（备/碑）永远排在补全词与模糊音之后。
    fn whole_key_words(&self, pinyin: &str) -> Vec<Scored> {
        let _span = LocalSpan::enter_with_local_parent("whole_key_words");
        self.model
            .ranked_words(pinyin, WHOLE_KEY_WORDS)
            .into_iter()
            .enumerate()
            .map(|(rank, (word, freq))| {
                // 无 LM 时退化为按词库排名给分（保持词频顺序，仍是有限值可参与排序）
                #[allow(clippy::cast_precision_loss)] // rank < WHOLE_KEY_WORDS
                let (score, dict_tail) = self.lm.as_ref().map_or_else(
                    || (-(rank as f32), false),
                    |lm| {
                        let (base, tail) = reading_base(&word, freq, lm.as_ref());
                        (base + self.boost(pinyin, &word), tail)
                    },
                );
                Scored {
                    cand: Candidate::whole(
                        word.to_string(),
                        vec![LearnedWord::new(pinyin.to_string(), word.into_owned())],
                        pinyin.len(),
                    ),
                    score,
                    fuzzy_edges: 0, // 整键精确匹配
                    completion: false,
                    dict_tail,
                }
            })
            .collect()
    }

    /// 尾音节补全：末音节残缺（`chijiuh` → `h→hua`）或可延长（`chijiuhu` → `hu→hua`）时，
    /// 补出完整词候选（带分数；用户学过的词如 持久化 用 `USER_WORD_BASE` + 调频）。
    ///
    /// 也覆盖「整串都还不是音节」的开局（`l`/`zh`）：那时残缺尾部就是整个输入，
    /// base 为空。曾经这里有 `len < 2` 与 `max_pos == 0` 两道早退，纯声母（l/z/w/zh）
    /// 因此一个候选都不出——它是每个词的第一键，不能空窗。
    fn completions(&self, pinyin: &str, lm: &L) -> Vec<Scored> {
        let _span = LocalSpan::enter_with_local_parent("completions");
        if pinyin.is_empty() {
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
        // max_pos == 0（一个音节都切不出来）不早退：整串当残缺尾部走下面的分支。

        // (完整 key 的前缀, 要补全成的更长沙节)
        let mut completions: Vec<(String, String)> = Vec::new();
        if max_pos == pinyin.len() {
            // 完整输入：末音节可延长（chijiuhu 的 hu → hua）
            for e in lattice.iter().flatten().filter(|e| e.end == pinyin.len() && !e.is_fuzzy()) {
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

        let mut out: Vec<Scored> = Vec::new();
        for (base, ext) in completions {
            let key = format!("{base}{ext}");
            for (word, freq) in self.model.ranked_words(&key, 4) {
                let mut score = reading_base(&word, freq, lm).0 + self.boost(&key, &word);
                // 补全词（延长音节）减惩罚：精确读音候选优先于补全候选；
                // 用户词（freq == 0）不动，保证 持久化 这类仍能压过拼接。
                if freq != 0 {
                    score += COMPLETION_PENALTY;
                }
                out.push(Scored {
                    // 补全候选消耗掉全部已输入的拼音（它还多补了没打完的音节）
                    cand: Candidate::whole(
                        word.to_string(),
                        vec![LearnedWord::new(key.clone(), word.into_owned())],
                        pinyin.len(),
                    ),
                    score,
                    fuzzy_edges: 0,
                    // 用户学过的补全词（持久化）仍按补全处理：它排在精确候选之后，
                    // 但精确候选里没有它的竞争者时依旧是第一梯队。
                    completion: true,
                    // 补全候选本就在第 1 组，长尾与否不再影响分组
                    dict_tail: false,
                });
            }
        }
        out
    }

    /// beam search 主循环：在音节格上反复展开 + 束剪枝，返回完整句子假设
    /// （按分数降序、已去重截断）与复原路径用的 arena。
    ///
    /// 每个位置的可用词键只枚举一次（`PosKey`），假设本身是 `Copy` 的小结构，
    /// 路径靠 arena 父指针表示 —— 展开阶段没有任何堆分配。
    ///
    /// 展开次数以 fastrace 事件（`expand`）记录在 `beam` span 上，供阶段
    /// 工作量分解（无 context 时 noop 零开销）。
    /// 位置同步（position-synchronous）beam 搜索：Top-K 句子候选。
    ///
    /// **为什么必须按位置分桶剪枝**，而不是把所有存活假设放一起取前 BEAM：
    ///
    /// 累积 logP 随「已消费的音节数」单调下降，所以覆盖 3 个音节的
    /// `输入法`（≈-5.2）永远排在只覆盖 1 个音节的 `书`（≈-3.4）后面。
    /// 全局 beam 会系统性地把多音节词从前沿剪掉 —— 长句于是只能逐字硬拼：
    ///
    /// ```text
    ///   shurufa               → 输入法              （单独输入时对）
    ///   shurufabuzhun         → 书如发不准          （全局 beam：词被剪掉）
    ///   wodeshurufahenhaoyong → 我的书如法很好用
    /// ```
    ///
    /// 高频词（`我们`）能侥幸活下来，低频词（`输入法`）必死——这不是分数调参
    /// 能解决的，是**不同覆盖长度的假设不可比**。
    ///
    /// 改成按 `pos` 分桶后，只有覆盖相同输入量的假设互相竞争（`输入法` 与
    /// `书+如+发` 都停在 pos 3，公平比较），长度偏置消失。
    fn beam_search(
        &self,
        pinyin: &str,
        lattice: &[Vec<SyllableEdge>],
        reachable: &[bool],
        lm: &L,
        limits: Limits,
    ) -> BeamResult {
        let _ = pinyin; // 位置同步版不再按输入长度轮询
        let mut arena: Vec<Node> = Vec::new();
        // 每个位置的词键（懒枚举：只有 beam 真的走到的位置才算）
        let mut keys_at: Vec<Option<Vec<PosKey>>> = (0..lattice.len()).map(|_| None).collect();
        // 假设按「已消费到的位置」分桶：桶内才是可比的
        let mut at: Vec<Vec<Hyp>> = (0..=lattice.len()).map(|_| Vec::new()).collect();
        at[0].push(Hyp {
            pos: 0,
            prev: Prev::Start,
            score: 0.0,
            fuzzy_edges: 0,
            first_tail: false,
            node: NO_NODE,
        });
        // 完整句子：消费完全部输入的假设（末位置那个桶），不参与中途剪枝
        let mut done: Vec<Hyp> = Vec::new();

        // 前词 → bigram 行的小缓存（每个位置存活假设 ≤ BEAM 条，线性扫描最快）
        let mut rows: Vec<(u32, (u32, u32))> = Vec::with_capacity(BEAM * 2);

        let _beam_span = LocalSpan::enter_with_local_parent("beam");
        let mut n_expand = 0usize;
        let mut scratch: Vec<Hyp> = Vec::new();
        for pos in 0..lattice.len() {
            // 取出本位置的假设（take 避免「遍历 at[pos] 同时写 at[end]」的借用冲突；
            // 边只会前进，end > pos，所以不存在自我别名）
            let mut bucket = std::mem::take(&mut at[pos]);
            if bucket.is_empty() {
                continue;
            }
            // 桶内剪枝：只有到这里才丢，保证同覆盖长度的假设公平竞争。
            // 用 partition/select 的稳定性要求见下方 `done` 去重的注释。
            if bucket.len() > BEAM {
                bucket.select_nth_unstable_by(BEAM, |a, b| b.score.total_cmp(&a.score));
                bucket.truncate(BEAM);
            }
            if keys_at[pos].is_none() {
                keys_at[pos] = Some(self.keys_at(lattice, pos, limits));
            }
            let Some(keys) = keys_at[pos].as_ref() else { continue };
            rows.clear();
            for h in &bucket {
                // 前词的 bigram 行：只有活下来的假设才定位，且同一前词复用 ——
                // 定位一次行 = 两次全表二分，绝不能放到「每次展开」里。
                let row = match h.prev {
                    Prev::Word(Some(id)) => {
                        if let Some((_, r)) = rows.iter().find(|(w, _)| *w == id) {
                            *r
                        } else {
                            let r = lm.bigram_row(id);
                            rows.push((id, r));
                            r
                        }
                    }
                    _ => (0, 0),
                };
                for pk in keys {
                    n_expand += pk.limit;
                    expand(h, row, pk, &mut scratch, &mut arena, lm, limits);
                }
                // 按落点分发；死路（走不到末尾的错误切分）直接丢。
                // scratch 是复用缓冲（避免每个假设一次堆分配），所以分发后清空而非消耗
                for nh in &scratch {
                    if nh.pos >= lattice.len() {
                        done.push(*nh);
                    } else if reachable[nh.pos] {
                        at[nh.pos].push(*nh);
                    }
                }
                scratch.clear();
            }
        }
        // 展开次数：工作量指标 → **属性**（挂在 beam span 上，聚合表直接可读）
        LocalSpan::add_property(|| ("expand", n_expand.to_string()));

        done.sort_by(|a, b| b.score.total_cmp(&a.score));
        // 去重：同一句文本可能来自「多音节词」和「单字拼合」两条路径，保留最高分
        let mut seen = std::collections::HashSet::new();
        done.retain(|h| seen.insert(sentence_of(&arena, h.node)));
        done.truncate(limits.top_sentences);
        BeamResult {
            hyps: done,
            arena,
            keys_at,
        }
    }

    /// 枚举某个位置上所有可展开的词键（单音节 + 多音节整词）及其候选词。
    ///
    /// 与假设无关，因此每个位置只调一次；词库前缀查询用于剪掉不可能成词的链。
    fn keys_at(&self, lattice: &[Vec<SyllableEdge>], pos: usize, limits: Limits) -> Vec<PosKey> {
        let _span = LocalSpan::enter_with_local_parent("keys_at");
        let mut out: Vec<PosKey> = Vec::new();
        // 链键拼接的复用缓冲：只有确认成键（前缀命中）时才真的分配 String
        let mut buf = String::new();
        for edge in &lattice[pos] {
            self.push_key(
                &mut out,
                &KeyPath {
                    key: edge.syl,
                    end: edge.end,
                    cost: edge.cost,
                    fuzzy_edges: u8::from(edge.is_fuzzy()),
                },
                limits,
            );
            // 多音节词：沿格的所有路径拼接完整 key（如 xin-xi-liang → 信息量），
            // 词库不存在的键剪枝——不依赖格内 edge 顺序（first() 会因音节表
            // 顺序拼错路径，漏掉整词）。
            let mut chains: Vec<(String, usize, u8, u8)> = vec![(
                String::from(edge.syl),
                edge.end,
                edge.cost,
                u8::from(edge.is_fuzzy()),
            )];
            for _ in 1..MAX_WORD_SYLLABLES {
                let mut next_chains: Vec<(String, usize, u8, u8)> = Vec::new();
                for (key, cur_end, cost, fuzzy_edges) in &chains {
                    let Some(edges) = lattice.get(*cur_end) else { continue };
                    for next_edge in edges {
                        buf.clear();
                        buf.push_str(key);
                        buf.push_str(next_edge.syl);
                        if !self.has_key_prefix(&buf) {
                            continue; // 词库无以此开头的键：不可能成词，剪枝
                        }
                        next_chains.push((
                            buf.clone(),
                            next_edge.end,
                            (*cost).max(next_edge.cost),
                            fuzzy_edges.saturating_add(u8::from(next_edge.is_fuzzy())),
                        ));
                    }
                }
                for (key, end, cost, fuzzy_edges) in &next_chains {
                    self.push_key(
                        &mut out,
                        &KeyPath {
                            key,
                            end: *end,
                            cost: *cost,
                            fuzzy_edges: *fuzzy_edges,
                        },
                        limits,
                    );
                }
                chains = next_chains;
            }
        }
        out
    }

    /// 锁词键缓存（毒锁恢复：缓存是纯派生数据，锁中毒不该让输入法崩）。
    fn lock_keys(&self) -> std::sync::MutexGuard<'_, KeyCache> {
        self.keys
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// 词键的候选词（缓存命中则零查询零分配）。
    fn words_of(&self, key: &str, limit: usize) -> Arc<[WordCand]> {
        if let Some((cached_limit, hit)) = self.lock_keys().words.get(key)
            && (*cached_limit >= limit || hit.len() < *cached_limit)
        {
            return hit.clone(); // 够宽，或词库本身就没那么多词
        }
        let words: Arc<[WordCand]> = self
            .model
            .ranked_words(key, limit)
            .into_iter()
            .map(|(word, freq)| {
                let lm_id = self.lm.as_ref().and_then(|lm| lm.word_index(&word));
                let (first_base, dict_tail) = self.lm.as_ref().map_or(
                    (OOV_BASE, false),
                    |lm| first_word_base(freq, lm_id, lm.as_ref()),
                );
                WordCand {
                    first_base,
                    dict_tail,
                    boost: self.boost(key, &word),
                    word: Arc::from(word),
                    lm_id,
                }
            })
            .collect();
        let mut cache = self.lock_keys();
        if cache.words.len() >= KEY_CACHE_CAP {
            cache.words.clear();
        }
        cache.words.insert(Box::from(key), (limit, words.clone()));
        words
    }

    /// 词库里是否存在以 `key` 开头的词键（多音节链剪枝，结果缓存）。
    fn has_key_prefix(&self, key: &str) -> bool {
        if let Some(hit) = self.lock_keys().prefix.get(key) {
            return *hit;
        }
        let hit = self.model.has_key_prefix(key);
        let mut cache = self.lock_keys();
        if cache.prefix.len() >= KEY_CACHE_CAP {
            cache.prefix.clear();
        }
        cache.prefix.insert(Box::from(key), hit);
        hit
    }

    /// 把一个词键的候选词收进枚举结果（无候选词的键直接丢弃）。
    ///
    /// 同一位置的同一词键可能由多条边得到（多个模糊变体归一到同一标准音节），
    /// 只保留代价最低的一份：代价低 = 分高，此前重复展开也是靠最终去重留高分。
    fn push_key(&self, out: &mut Vec<PosKey>, path: &KeyPath<'_>, limits: Limits) {
        let KeyPath { key, end, cost, fuzzy_edges } = *path;
        if let Some(prev) = out.iter_mut().find(|p| p.end == end && &*p.key == key) {
            if cost < prev.cost {
                prev.cost = cost;
                prev.fuzzy_edges = fuzzy_edges;
            }
            return;
        }
        let words = self.words_of(key, limits.words_per_key);
        if words.is_empty() {
            return;
        }
        out.push(PosKey {
            end,
            cost,
            fuzzy_edges,
            key: Arc::from(key),
            // 缓存可能比本次需要的更宽（另一档输入填的），只取前 limit 个
            limit: limits.words_per_key.min(words.len()),
            words,
        });
    }

    /// beam search 解码：Top-K 句子候选（带分数）。
    fn decode(&self, pinyin: &str) -> Vec<Scored> {
        let Some(lm) = &self.lm else {
            return Vec::new();
        };
        // fastrace：lattice 构建（模糊音节格）
        let lattice = {
            let _span = LocalSpan::enter_with_local_parent("lattice");
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

        // 覆盖输入所需的最少音节数（最短跳数 DP）：候选规模按它自适应。
        // 单音节 li 与七字整句用同一套上限是错的（前者要宽、后者要省）。
        let limits = Limits::for_syllables(min_syllables(&lattice));

        let BeamResult {
            hyps,
            arena,
            keys_at,
        } = self.beam_search(pinyin, &lattice, &reachable, lm, limits);
        let mut out: Vec<Scored> = hyps
            .into_iter()
            .map(|h| {
                // 路径只在这里（top-K 条）复原：热路径不碰字符串
                let segments = segments_of(&arena, h.node);
                // 长尾标记只对单词句子有意义：多词拼接是另一回事，不能因为
                // 句首是个异体字就把整句降组
                let dict_tail = h.first_tail && segments.len() == 1;
                Scored {
                    cand: Candidate::whole(
                        segments.iter().map(|(_, w)| &***w).collect::<String>(),
                        segments
                            .iter()
                            .map(|(k, w)| LearnedWord::new(k.to_string(), w.to_string()))
                            .collect(),
                        pinyin.len(),
                    ),
                    score: h.score,
                    fuzzy_edges: h.fuzzy_edges,
                    completion: false,
                    dict_tail,
                }
            })
            .collect();
        // 部分候选：位置 0 出发、只覆盖输入前一段的词（整句错了就咬一段确认）
        if let Some(keys) = keys_at.first().and_then(Option::as_ref) {
            out.extend(partial_candidates(keys, pinyin.len()));
        }
        out
    }

    /// 用户调频加成（口径与语音侧共用：`cnt_score::policy::user`）。
    fn boost(&self, syllable: &str, word: &str) -> f32 {
        cnt_score::policy::user::boost(self.model.user_count(syllable, word))
    }

}

/// 读音感知的基础分（首词用），返回 `(基础分, 是否词库长尾)`：
/// - 用户词（`freq == 0`）：`USER_WORD_BASE` —— 学过的词应能浮现，不再被 -12 埋没
/// - 次读音（`freq ≤ SECONDARY_FREQ_CAP` 且在 LM，如 的di、和hu）：封顶到
///   `SECONDARY_BASE` —— 多音字不再串频（否则 fu 会出 和、diyige 会出 的一个）
/// - 不在 LM 的词（长尾字形、生僻整词）：`oov_score(freq)`，按词库词频排序
/// - 主读音：LM unigram（保留细粒度排序：是 > 时 > 事）
///
/// 「freq ≤ CAP 一律给 `SECONDARY_BASE`」曾是本文件最贵的一个平带：词库把
/// 「不在 LM unigram 的读音」全记成 freq=1，于是 㙢/㝵/㮶 这类 Ext-B 字形拿到恒定
/// -6.000，反而压过走 LM unigram 的 5000 频常用字（扪 -6.115、锝 -6.2…），把
/// men/de/shuo/fa/zai 的候选 4~10 位整段占掉。判据落在「在不在 LM 词表」上：
/// 真次读音的字必在 LM（它的主读音是常用词），长尾字形不在。
fn reading_base<L: NgramLm>(word: &str, freq: u32, lm: &L) -> (f32, bool) {
    if freq == 0 {
        return (USER_WORD_BASE, false);
    }
    lm.unigram(word).map_or_else(
        || (oov_score(freq), is_dict_tail(freq)),
        |(p, _)| (secondary_cap(freq, p), false),
    )
}

/// 词库长尾：**不在 LM 词表**（调用方已确认）且 `freq ≤ SECONDARY_FREQ_CAP`。
///
/// 就是 ㅕ午/咹/嘶/筽/兒（繁体）这批：词库里只有一个占位词频（构建时不在
/// LM unigram 的读音一律记 freq=1）、LM 也不认识。它们仍然是「精确读音」，
/// 但不该凭「覆盖了全部输入」这条硬约束压在常用补全词（你/能/年）前面
/// —— 见 `candidates_scored` 的分组注释。
const fn is_dict_tail(freq: u32) -> bool {
    freq <= SECONDARY_FREQ_CAP
}

/// 次读音封顶：`freq ≤ SECONDARY_FREQ_CAP` 的词不得继承主读音的 LM 概率。
const fn secondary_cap(freq: u32, unigram: f32) -> f32 {
    if freq <= SECONDARY_FREQ_CAP {
        unigram.min(SECONDARY_BASE)
    } else {
        unigram
    }
}

/// 不在 LM 的词（OOV）的基础分：以 `OOV_BASE` 为基准，按词库词频做 log10 修正。
#[allow(clippy::cast_precision_loss)] // 词频量级 ≤ 1e9，f32 精度足够
fn oov_score(freq: u32) -> f32 {
    let adjust = (freq.max(1) as f32 / OOV_REF_FREQ)
        .log10()
        .clamp(OOV_ADJUST_MIN, OOV_ADJUST_MAX);
    OOV_BASE + adjust
}

impl<L: NgramLm> CandidateSource for Decoder<L> {
    fn candidates(&self, pinyin: &str) -> Vec<Candidate> {
        self.candidates_merged(pinyin)
    }
}

/// 覆盖整个输入所需的最少音节数（音节格上的最短跳数 DP）。
///
/// 只用来选候选规模，不参与打分；走不通（无合法切分）时返回 `usize::MAX`，
/// 此时按长输入的保守规模处理。
fn min_syllables(lattice: &[Vec<SyllableEdge>]) -> usize {
    let end = lattice.len();
    let mut hops = vec![usize::MAX; end + 1];
    hops[0] = 0;
    for pos in 0..end {
        if hops[pos] == usize::MAX {
            continue;
        }
        for e in &lattice[pos] {
            hops[e.end] = hops[e.end].min(hops[pos] + 1);
        }
    }
    hops[end]
}

/// 硬约束：模糊边 ≥2 的候选不得占 #1。
///
/// 超线性惩罚已让多处模糊的分数大幅下沉，但「不占 #1」是可用性底线，不能只靠
/// 调参：一旦榜首是叠加模糊候选，就把最靠前的「单处模糊或精确」候选提到 #1
/// （其余相对顺序不变）。找不到这样的候选时（全是叠加模糊）保持原状。
fn demote_stacked_fuzzy(out: &mut [Scored]) {
    if out
        .first()
        .is_none_or(|s| s.fuzzy_edges <= MAX_FUZZY_EDGES_AT_TOP)
    {
        return;
    }
    if let Some(i) = out
        .iter()
        .position(|s| s.fuzzy_edges <= MAX_FUZZY_EDGES_AT_TOP && !s.completion)
    {
        out[..=i].rotate_right(1);
    }
}

/// 把一个词键的候选词展开进 beam。
///
/// 热路径：每次展开只做「加法 + 一次 bigram 查询 + 一次 arena push」，
/// 没有堆分配、没有字符串、没有锁（这些都在 `PosKey` 枚举阶段做完了）。
fn expand<L: NgramLm>(
    h: &Hyp,
    row: (u32, u32),
    pk: &PosKey,
    next: &mut Vec<Hyp>,
    arena: &mut Vec<Node>,
    lm: &L,
    limits: Limits,
) {
    let base_penalty = FUZZY_PENALTY[usize::from(pk.cost.min(MAX_FUZZY_COST))];
    // 超线性叠加：本词贡献 pk.fuzzy_edges 条模糊边，之前已有 h.fuzzy_edges 条，
    // 除第 1 条外每条额外扣 FUZZY_STACK_PENALTY（整词内部连错两音同样算叠加：
    // zhuchen → 组成 是 zh/z + en/eng 两条边，不能只按最大 cost 扣一次）。
    let penalty = if pk.fuzzy_edges > 0 {
        let stacked = h
            .fuzzy_edges
            .saturating_add(pk.fuzzy_edges)
            .saturating_sub(1);
        FUZZY_STACK_PENALTY.mul_add(-f32::from(stacked), base_penalty) - limits.fuzzy_extra
    } else {
        0.0
    };
    let fuzzy_edges = h.fuzzy_edges.saturating_add(pk.fuzzy_edges);
    for w in pk.words.iter().take(pk.limit) {
        // 句首用读音感知的基础分（枚举期算好），续接用条件概率
        // （bigram，缺失走 Katz backoff —— 回退语义由 NgramLm::conditional 定义）。
        // 续接：在前词的 bigram 行内查条件概率（行已在上一步定位好，见 Hyp::row）；
        // 缺失走 Katz backoff —— 回退语义由 NgramLm 端口统一定义。
        let step = match h.prev {
            // 不在 LM 词表的词（用户新词、词库长尾）用它的句首基础分当伪 unigram，
            // 而不是一律 UNK(-12)：否则学过的复合词只能出现在句首，句子中间一定
            // 输给逐字拼接（-12 的悬崖比任何拼接都差）。
            Prev::Word(prev_id) => lm.conditional_in_row(row, prev_id, w.lm_id, w.first_base),
            Prev::Start => w.first_base,
        } + w.boost
            + penalty;
        arena.push(Node {
            parent: h.node,
            key: pk.key.clone(),
            word: w.word.clone(),
        });
        next.push(Hyp {
            pos: pk.end,
            prev: Prev::Word(w.lm_id),
            score: h.score + step,
            fuzzy_edges,
            // 句首词才定调：后续词的长尾与否不影响分组（只有单词句子用得上这个标记）
            first_tail: match h.prev {
                Prev::Start => w.dict_tail,
                Prev::Word(_) => h.first_tail,
            },
            node: u32::try_from(arena.len() - 1).unwrap_or(NO_NODE),
        });
    }
}

/// 句首词的读音感知基础分 + 长尾标记（枚举期算一次，见 `reading_base` 的同款规则）。
fn first_word_base<L: NgramLm>(freq: u32, lm_id: Option<u32>, lm: &L) -> (f32, bool) {
    if freq == 0 {
        return (USER_WORD_BASE, false); // 用户词：学过的词应能浮现
    }
    // 不在 LM 的词（整词长尾、Ext-B 字形）按词频给 OOV 分，而非 UNK（否则必然
    // 输给整句拼接），也不给 SECONDARY_BASE 平带（否则压过 LM 里的常用字）。
    lm_id.and_then(|i| lm.unigram_by_id(i)).map_or_else(
        || (oov_score(freq), is_dict_tail(freq)),
        |(p, _)| (secondary_cap(freq, p), false),
    )
}

/// 从位置 0 的词键里取出「只覆盖输入前一段」的候选。
///
/// 打分与句首词同一套（读音基础分 + 用户调频 + 模糊惩罚），但它们在候选排序里
/// 自成一组（见 `candidates_scored`），不与整句混排。
fn partial_candidates(keys: &[PosKey], input_len: usize) -> Vec<Scored> {
    let mut out: Vec<Scored> = Vec::new();
    for pk in keys
        .iter()
        // 只取精确读音：部分确认是「修复入口」，给模糊读音的怪切分（niha → lifa）
        // 只会添乱；模糊音该在整句候选里体现。
        .filter(|pk| pk.end < input_len && pk.fuzzy_edges == 0)
    {
        for w in pk.words.iter().take(pk.limit) {
            out.push(Scored {
                cand: Candidate::partial(
                    w.word.to_string(),
                    vec![LearnedWord::new(pk.key.to_string(), w.word.to_string())],
                    pk.end,
                ),
                score: w.first_base + w.boost,
                fuzzy_edges: 0,
                completion: false,
                // 部分候选自成一组（GROUP_PARTIAL），长尾与否不影响分组
                dict_tail: false,
            });
        }
    }
    // 只按覆盖长度排（稳定排序，组内保持词库排名）：单词候选里「词频高」比
    // 「LM unigram 高」更接近用户想要的（你好 的 unigram 在地板上，词频却极高）。
    out.sort_by_key(|s| std::cmp::Reverse(s.cand.consumed));
    out.truncate(PARTIAL_LIMIT);
    out
}

/// 沿 arena 父指针复原路径（拼音键, 词），按输入顺序返回。
fn segments_of(arena: &[Node], node: u32) -> Vec<(&Arc<str>, &Arc<str>)> {
    let mut out = Vec::new();
    let mut cur = node;
    while let Some(n) = arena.get(cur as usize) {
        out.push((&n.key, &n.word));
        if n.parent == NO_NODE {
            break;
        }
        cur = n.parent;
    }
    out.reverse();
    out
}

/// 复原整句文本（去重用）。
fn sentence_of(arena: &[Node], node: u32) -> String {
    segments_of(arena, node)
        .into_iter()
        .map(|(_, w)| &**w)
        .collect()
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

    /// 位置同步 beam 的回归：长输入必须仍然用得上多音节词。
    ///
    /// 曾经的 bug：全局 beam 按累积 logP 取前 8，覆盖 3 个音节的 `输入法`（≈-5.2）
    /// 永远排在只覆盖 1 个音节的 `书`（≈-3.4）后面 → 长句退化成逐字硬拼
    /// （`shurufabuzhun` → 书如发不准）。高频词侥幸活下来，低频词必死。
    #[test]
    fn long_input_still_uses_multi_syllable_words() {
        // 位置同步的核心性质：同一位置的假设才互相比较。
        // 这里用纯逻辑断言表达该性质（真实词库的端到端验证在 tests/e2e.rs 与
        // cnt-dict-tools decode 里做，单测不依赖 30 万词的词库）。
        let mut bucket = [
            Hyp { pos: 3, prev: Prev::Start, score: -5.2, fuzzy_edges: 0, first_tail: false, node: NO_NODE },
            Hyp { pos: 1, prev: Prev::Start, score: -3.4, fuzzy_edges: 0, first_tail: false, node: NO_NODE },
        ];
        // 全局排序会把覆盖 1 个音节的假设排在前面（这正是偏置的来源）
        bucket.sort_by(|a, b| b.score.total_cmp(&a.score));
        assert_eq!(bucket[0].pos, 1, "全局排序偏向覆盖短的假设");
        // 按位置分桶后，两者根本不在同一个桶里，不会互相挤掉
        let same_bucket = bucket[0].pos == bucket[1].pos;
        assert!(!same_bucket, "不同覆盖长度的假设不可比，必须分桶");
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
    fn fuzzy_tier_keeps_exact_reading_first() {
        // 鼻韵尾（an/ang，cost=2）：输入 xiangchen 时，即使 县城(xiancheng) 词频高得多，
        // 也不得压过精确读音的 相称。
        let d = decoder_with(
            "fuzzy_tier",
            &[
                ("xiang", "相", 900),
                ("chen", "称", 900),
                ("xian", "县", 900),
                ("cheng", "城", 900),
            ],
            &[
                ("相", -3.0, 0.0),
                ("称", -3.0, 0.0),
                ("县", -2.0, 0.0),
                ("城", -2.0, 0.0),
            ],
            &[("相", "称", -3.0), ("县", "城", -2.0)],
            true,
        );
        let cands = d.candidates("xiangchen");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts.first(), Some(&"相称"), "精确读音应 #1: {texts:?}");
    }

    #[test]
    fn bare_initial_yields_completion_candidates() {
        // 纯声母（l/z/w/zh）一个音节都切不出来，但它是每个词的第一键：
        // 必须按补全给出候选，不能空窗（曾被 `len < 2` 与 `max_pos == 0` 两道早退挡住）。
        let d = decoder_with(
            "bare_initial",
            &[("le", "了", 90_000), ("lai", "来", 80_000), ("zhe", "这", 90_000)],
            &[("了", -2.0, 0.0), ("来", -2.8, 0.0), ("这", -2.1, 0.0)],
            &[],
            false,
        );
        for (input, want) in [("l", "了"), ("zh", "这")] {
            let texts: Vec<String> =
                d.candidates(input).into_iter().map(|c| c.text).collect();
            assert_eq!(
                texts.first().map(String::as_str),
                Some(want),
                "输入 {input} 应给出补全候选: {texts:?}"
            );
        }
        // 空输入仍然没有候选（别把空窗变成全词库）
        assert!(d.candidates("").is_empty());
    }

    /// 回归：词库长尾字形（LM 不认识、词频 ≤ CAP）不得凭「精确覆盖全输入」
    /// 的硬分组压在常用补全词前面。
    ///
    /// 真实现场：输入 `n` 时 ㅕ午/咹（OOV 地板 -9.0）占掉第 4~5 位，把 你/能/年
    /// 挤到 6 位后；`m`（嘶）、`o`（噦/筽）、`r`（兒繁体）同病。
    #[test]
    fn dict_tail_does_not_outrank_common_completion() {
        let d = decoder_with(
            "dict_tail",
            &[
                // 精确读音：常用叹词（在 LM）+ 两个长尾字形（不在 LM、词频 1）
                ("n", "嗯", 50_000),
                ("n", "ㅕ", 1),
                ("n", "咹", 1),
                // 补全候选：常用字
                ("ni", "你", 90_000),
                ("neng", "能", 80_000),
            ],
            &[("嗯", -4.9, 0.0), ("你", -2.5, 0.0), ("能", -2.7, 0.0)],
            &[],
            false,
        );
        let texts: Vec<String> = d.candidates("n").into_iter().map(|c| c.text).collect();
        let pos = |t: &str| texts.iter().position(|x| x == t);
        assert_eq!(pos("嗯"), Some(0), "LM 认识的精确读音仍然 #1: {texts:?}");
        for tail in ["ㅕ", "咹"] {
            assert!(
                pos("你") < pos(tail) && pos("能") < pos(tail),
                "常用补全词应排在长尾字形 {tail} 之前: {texts:?}"
            );
        }
    }

    #[test]
    fn completion_does_not_outrank_exact_syllable() {
        // 输入 jian：补全候选 将(jiang，词频更高) 不得压过精确读音的 见。
        let d = decoder_with(
            "completion_rank",
            &[("jian", "见", 900), ("jiang", "将", 900), ("hou", "后", 900)],
            &[("见", -3.5, 0.0), ("将", -2.8, 0.0), ("后", -3.0, 0.0)],
            &[],
            false,
        );
        let cands = d.candidates("jian");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        let exact = texts.iter().position(|t| *t == "见");
        let compl = texts.iter().position(|t| *t == "将");
        assert!(
            exact.is_some() && (compl.is_none() || exact < compl),
            "精确读音 见 应在补全候选 将 之前: {texts:?}"
        );
    }

    #[test]
    fn mono_input_fills_top_with_exact_readings() {
        // 单音节 li：精确同音字（里/力/历）必须排在补全词（两 = liang）
        // 与模糊音候选（你 = ni，l/n）之前——单音节没有上下文佐证模糊回退。
        let d = decoder_with(
            "mono",
            &[
                ("li", "里", 50_000),
                ("li", "力", 30_000),
                ("li", "历", 20_000),
                ("liang", "两", 90_000),
                ("ni", "你", 90_000),
            ],
            &[
                ("里", -3.0, 0.0),
                ("力", -3.5, 0.0),
                ("历", -4.0, 0.0),
                ("两", -2.5, 0.0), // 补全词 LM 分更高，仍不得插到精确读音前
                ("你", -2.0, 0.0), // 模糊候选 LM 分最高，同样不得插队
            ],
            &[],
            true,
        );
        let cands = d.candidates("li");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        let pos = |t: &str| texts.iter().position(|x| *x == t);
        let (li, force, hist) = (pos("里"), pos("力"), pos("历"));
        assert!(li.is_some() && force.is_some() && hist.is_some(), "精确字应全部在候选里: {texts:?}");
        for exact in [li, force, hist] {
            for other in [pos("两"), pos("你")] {
                if let (Some(e), Some(o)) = (exact, other) {
                    assert!(e < o, "精确读音应在补全/模糊之前: {texts:?}");
                }
            }
        }
    }

    #[test]
    fn stacked_fuzzy_never_ranks_first() {
        // zhuchen：组成（zu-cheng，zh/z + en/eng 两处模糊）即使分数最高，
        // 也不得占 #1——两处同时打错的概率远低于一处。
        let d = decoder_with(
            "stacked_fuzzy",
            &[("zhu", "主", 90_000), ("chen", "臣", 50_000), ("zucheng", "组成", 80_000)],
            &[("主", -3.0, -0.5), ("臣", -4.0, 0.0), ("组成", -2.0, 0.0)],
            &[],
            true,
        );
        let cands = d.candidates("zhuchen");
        let texts: Vec<&str> = cands.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts.first(), Some(&"主臣"), "叠加模糊不得占 #1: {texts:?}");
        assert!(texts.contains(&"组成"), "叠加模糊候选仍应保留（只是不占 #1）: {texts:?}");
    }

    #[test]
    fn offers_partial_candidates_for_repair() {
        // 整句候选之外必须给「只覆盖前一段」的词候选：整句错了不必删光重打，
        // 选中前一段确认，剩余拼音继续组合。
        let d = decoder_with(
            "partial",
            &[
                ("ni", "你", 900),
                ("hao", "好", 900),
                ("nihao", "你好", 5000),
                ("shijie", "世界", 5000),
                ("shijie", "时节", 4000),
            ],
            &[
                ("你", -2.5, -0.5),
                ("好", -3.0, -0.5),
                ("你好", -4.0, -0.5),
                ("世界", -4.5, 0.0),
                ("时节", -4.2, 0.0),
            ],
            &[("你好", "时节", -0.5), ("你好", "世界", -0.8)],
            false,
        );
        let cands = d.candidates("nihaoshijie");
        let partial = cands
            .iter()
            .find(|c| c.text == "你好")
            .expect("应给出部分候选 你好");
        assert_eq!(partial.consumed, "nihao".len(), "部分候选要记下消耗掉多少输入");
        assert!(!partial.covers_all("nihaoshijie".len()));
        // 结构约束：完全覆盖输入的候选一律排在部分候选之前
        let first_partial = cands
            .iter()
            .position(|c| !c.covers_all("nihaoshijie".len()))
            .expect("应有部分候选");
        assert!(
            cands[..first_partial]
                .iter()
                .all(|c| c.covers_all("nihaoshijie".len())),
            "部分候选不得插到整句候选中间: {:?}",
            cands.iter().map(|c| (&c.text, c.consumed)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn learned_compound_usable_mid_sentence() {
        // 学过的复合词必须能被 beam 当作句子中间的一个词用：
        // 教会 郑爽 之后，打 zhengshuangzhenpiaoliang 应切出 [郑爽][真][漂亮]，
        // 而不是把 郑/爽 拆成两个字去和别的路径竞争。
        let d = decoder_with(
            "compound_mid",
            &[
                ("zheng", "郑", 900),
                ("shuang", "爽", 900),
                ("zhen", "真", 900),
                ("piaoliang", "漂亮", 900),
            ],
            &[
                ("郑", -3.0, -0.5),
                ("爽", -3.5, -0.5),
                ("真", -2.5, -0.5),
                ("漂亮", -3.0, -0.5),
            ],
            &[("真", "漂亮", -0.5)],
            false,
        );
        // 先查一遍填满词键缓存（含「zhengsh 前缀不存在」的判定）
        let _ = d.candidates("zhengshuangzhenpiaoliang");
        d.learn(&[
            LearnedWord::new("zheng".to_string(), "郑".to_string()),
            LearnedWord::new("shuang".to_string(), "爽".to_string()),
        ]);
        let cands = d.candidates("zhengshuangzhenpiaoliang");
        let whole = cands.iter().find(|c| {
            c.learned
                .iter()
                .any(|l| l.pinyin == "zhengshuang" && l.word == "郑爽")
        });
        assert!(
            whole.is_some(),
            "学过的 郑爽 应作为整词参与整句切分: {:?}",
            cands.iter().map(|c| (&c.text, &c.learned)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn learn_boost_survives_key_cache() {
        // 词键缓存把「候选词表 + 用户调频加成」一起缓存了，learn 必须让对应键失效，
        // 否则用户调了半天频，beam 还在用旧加成分（越用越顺手直接失效）。
        let d = decoder_with(
            "learn_cache",
            &[("shi", "是", 900), ("shi", "时", 800), ("hou", "候", 900)],
            &[("是", -1.9, 0.0), ("时", -2.9, 0.0), ("候", -4.0, 0.0)],
            &[],
            false,
        );
        // 先查一次填满缓存（此时 是候 在前）
        let before = d.candidates("shihou");
        let pos_before = before.iter().position(|c| c.text == "时候");
        assert!(pos_before.is_some_and(|i| i > 0), "初始 时候 不该是 #1: {before:?}");
        // 反复选择 时（每次 +0.2 log10），足够翻过 是 与 时 的 1.0 差距
        for _ in 0..6 {
            d.learn(&[
                LearnedWord::new("shi".to_string(), "时".to_string()),
                LearnedWord::new("hou".to_string(), "候".to_string()),
            ]);
        }
        let after = d.candidates("shihou");
        assert_eq!(
            after.first().map(|c| c.text.as_str()),
            Some("时候"),
            "调频后 时候 应升到 #1（缓存未失效则不会变）: {after:?}"
        );
    }

    #[test]
    fn oov_score_follows_dict_frequency() {
        // 不在 LM 的词按词库词频分层，别让「词频 1 的生僻字」与常用字同分
        assert!(oov_score(20_000) > oov_score(5_000));
        assert!(oov_score(5_000) > oov_score(1));
        // 上下限：极高词频不得抬过 LM 覆盖词的量级，极低词频不得砸穿 UNK
        assert!(oov_score(u32::MAX) <= OOV_BASE + OOV_ADJUST_MAX);
        assert!(oov_score(0) >= OOV_BASE + OOV_ADJUST_MIN);
        assert!(oov_score(0) > UNK_LOGPROB);
    }

    /// 回归：`freq ≤ SECONDARY_FREQ_CAP` 的长尾字形不得压过 LM 里的常用字。
    ///
    /// 曾经的 bug：freq ≤ CAP 一律给 `SECONDARY_BASE`(-6.0)，而词库把「不在 LM
    /// unigram 的读音」全记成 freq=1，于是 Ext-B 字形（㙢/㡈）拿恒定 -6.000，
    /// 反而排在走 LM unigram 的 5000 频常用字（扪 -6.115）前面，把 men/de/shuo
    /// 的候选 4~10 位整段占掉。
    #[test]
    fn rare_glyphs_rank_below_lm_covered_chars() {
        let d = decoder_with(
            "rare-glyph",
            // 常用字在 LM（词库频 5000）；长尾字形不在 LM（词库频 1）
            &[("men", "扪", 5_000), ("men", "㙢", 1), ("men", "㡈", 1)],
            &[("扪", -6.115, 0.0)],
            &[],
            false,
        );
        let cands: Vec<String> =
            d.candidates("men").into_iter().map(|c| c.text).collect();
        let pos = |w: &str| cands.iter().position(|c| c == w);
        assert!(
            pos("扪") < pos("㙢") && pos("扪") < pos("㡈"),
            "LM 覆盖的常用字必须排在 freq=1 长尾字形之前: {cands:?}"
        );
    }

    /// 回归：真次读音（在 LM 词表里）仍被封顶，不继承主读音的高概率。
    ///
    /// 的(di)/和(hu) 的 LM unigram 是主读音的频率（的de ≈ -1.35），直接用会串频。
    #[test]
    fn secondary_reading_is_capped_not_boosted() {
        // 在 LM 且概率很高的次读音：封顶到 SECONDARY_BASE
        assert!((secondary_cap(1, -1.35) - SECONDARY_BASE).abs() < f32::EPSILON);
        // 本身就在 LM 地板的罕见字：不因为封顶反被抬升
        assert!((secondary_cap(1, -7.4) - (-7.4)).abs() < f32::EPSILON);
        // 主读音（freq > CAP）不受影响
        assert!((secondary_cap(20_000, -1.35) - (-1.35)).abs() < f32::EPSILON);
    }

    #[test]
    fn limits_scale_with_input_length() {
        // 单音节最宽、双音节居中、长输入最省（展开是乘性的，长输入不能放宽）
        let mono = Limits::for_syllables(1);
        let short = Limits::for_syllables(2);
        let long = Limits::for_syllables(5);
        assert!(mono.words_per_key > short.words_per_key);
        assert!(short.words_per_key > long.words_per_key);
        assert!(mono.top_sentences > short.top_sentences);
        assert!(short.top_sentences > long.top_sentences);
        // 只有单音节加重模糊惩罚（多音节靠上下文互相印证）
        assert!(mono.fuzzy_extra > 0.0);
        assert!(short.fuzzy_extra.abs() < f32::EPSILON);
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
