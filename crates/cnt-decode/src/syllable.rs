//! 标准拼音音节表 + 切分（含模糊音）。
//!
//! 音节表是《汉语拼音方案》的无调音节全集（约 410 个），硬编码为标准数据，
//! 不依赖任何外部库。切分用全切分（动态规划），生成音节格（lattice）。
//! 支持音节分隔符 `'`（Rime 惯例）：`xi'an` 强制切为 xi+an，不被自动切分
//! 当成 xian/现 —— 见 [`strip_separators`] 与 [`SyllableTable::lattice_strict`]。
//!
//! **模糊音**：`FUZZY_RULES` 定义双向替换规则（平翘舌/边鼻音/前后鼻音等），
//! 每个标准音节生成其模糊变体；用户输入的变体匹配回标准音节后查词。
//! 例：用户打 `si` → 变体匹配到标准 `shi` → 查 是/时 等词。

use std::collections::HashMap;

/// 标准无调拼音音节表（《汉语拼音方案》音节全集）。
const SYLLABLES: &[&str] = &[
    "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng",
    "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao",
    "ce", "cen", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng", "chi",
    "chong", "chou", "chu", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong",
    "cou", "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "dei",
    "den", "deng", "di", "dia", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du",
    "duan", "dui", "dun", "duo", "e", "ei", "en", "eng", "er", "fa", "fan", "fang", "fei",
    "fen", "feng", "fo", "fou", "fu", "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen",
    "geng", "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha",
    "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua",
    "huai", "huan", "huang", "hui", "hun", "huo", "ji", "jia", "jian", "jiang", "jiao", "jie",
    "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang",
    "kao", "ke", "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui",
    "kun", "kuo", "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li", "lia",
    "lian", "liang", "liao", "lie", "lin", "ling", "liu", "lo", "long", "lou", "lu", "luan",
    "lun", "luo", "lv", "lve", "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng",
    "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu", "na", "nai", "nan",
    "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin",
    "ning", "niu", "nong", "nou", "nu", "nuan", "nuo", "nv", "nve", "o", "ou", "pa", "pai",
    "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping",
    "po", "pou", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong",
    "qiu", "qu", "quan", "que", "qun", "ran", "rang", "rao", "re", "ren", "reng", "ri",
    "rong", "rou", "ru", "rua", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang",
    "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei", "shen",
    "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo",
    "si", "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai", "tan", "tang", "tao",
    "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui",
    "tun", "tuo", "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia",
    "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun",
    "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan",
    "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai",
    "zhan", "zhang", "zhao", "zhe", "zhei", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu",
    "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou", "zu",
    "zuan", "zui", "zun", "zuo",
];

/// 模糊音规则（双向）+ 代价等级：等级越高，越不像「用户本意读音」。
///
/// 分级依据是方音混淆的真实频度——一视同仁的单一惩罚会让高频的模糊音词
/// 压过精确读音词（xiangchen → 县城 挤掉 相称），而把惩罚整体加重又会
/// 打死真正常见的混淆（sihou → 时候）。故按类别分档：
/// - 1 = 平翘舌 zh/z、ch/c、sh/s：南方口音最普遍，惩罚最轻
/// - 2 = 边鼻音 n/l 与前后鼻音韵尾 an/ang…：常见但已足以改变词形
/// - 3 = 其他声母混淆 f/h、r/l、k/g、t/d：实际较少见，代价最高
const FUZZY_RULES: &[(&str, &str, u8)] = &[
    ("zh", "z", 1),
    ("ch", "c", 1),
    ("sh", "s", 1),
    ("n", "l", 2),
    ("f", "h", 3),
    ("r", "l", 3),
    ("k", "g", 3),
    ("t", "d", 3),
    ("an", "ang", 2),
    ("en", "eng", 2),
    ("in", "ing", 2),
    ("ian", "iang", 2),
    ("uan", "uang", 2),
];

/// 模糊代价的最大等级（`FUZZY_RULES` 中的最高档）。
pub const MAX_FUZZY_COST: u8 = 3;

/// 一条切分边：结束位置 + 标准音节 + 模糊代价等级（0 = 精确读音）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyllableEdge {
    pub end: usize,
    pub syl: &'static str,
    pub cost: u8,
}

impl SyllableEdge {
    /// 是否为模糊读音（非精确匹配）。
    #[must_use]
    pub const fn is_fuzzy(&self) -> bool {
        self.cost > 0
    }
}

/// 音节表：按首字母索引，查询 `input[pos..]` 处的合法音节。
pub struct SyllableTable {
    by_first: HashMap<char, Vec<&'static str>>,
    fuzzy: FuzzyTable,
}

/// 模糊变体表：`map[变体] = [(标准音节, 代价等级)]`
/// （变体本身是标准音节时以代价 0 包含自己）。
struct FuzzyTable {
    map: HashMap<&'static str, Vec<(&'static str, u8)>>,
    by_first: HashMap<char, Vec<&'static str>>,
}

impl Default for SyllableTable {
    fn default() -> Self {
        Self::new()
    }
}

/// 音节分隔符：组合中输入 `'` 强制切分（`xi'an` → xi+an）。
pub const SYLLABLE_SEPARATOR: char = '\'';

/// 剥离音节分隔符：返回 `(clean, pos_map, boundaries)`。
///
/// - `clean`：去掉所有 `'` 后的输入（切分/查询都在它上面做）；
/// - `pos_map[i]`：clean 第 i 个字符在**原始输入**中的字节下标（把候选的
///   consumed 从 clean 坐标映射回原串坐标用——`consumed = pos_map[c-1] + 1`）；
/// - `boundaries`：强制切分边界（clean 字节坐标）。连续分隔符折叠、
///   首尾分隔符忽略（`'xian` 或 `xian'` 的边界不产生效果）。
#[must_use]
pub fn strip_separators(input: &str) -> (String, Vec<usize>, Vec<usize>) {
    let mut clean = String::with_capacity(input.len());
    let mut pos_map = Vec::with_capacity(input.len());
    let mut boundaries = Vec::new();
    for (i, ch) in input.char_indices() {
        if ch == SYLLABLE_SEPARATOR {
            boundaries.push(clean.len());
        } else {
            clean.push(ch);
            pos_map.push(i);
        }
    }
    boundaries.dedup();
    boundaries.retain(|&b| b > 0 && b < clean.len());
    (clean, pos_map, boundaries)
}

impl SyllableTable {
    /// 用标准音节表构造（按首字母分桶，供 `syllables_at` 快速过滤）。
    #[must_use]
    pub fn new() -> Self {
        let by_first = bucket(SYLLABLES);
        Self {
            by_first,
            fuzzy: FuzzyTable::new(),
        }
    }

    /// 从 `input[pos..]` 能精确匹配到的所有音节。
    #[must_use]
    pub fn syllables_at(&self, input: &str, pos: usize) -> Vec<SyllableEdge> {
        syllables_in_bucket(&self.by_first, input, pos)
    }

    /// 从 `input[pos..]` 能匹配到的音节（含模糊变体；`cost > 0` 表示变体读音）。
    ///
    /// 例：输入 `sang` → 标准 `sang`（精确，cost=0）与 `shang`（模糊 s↔sh，cost=1）。
    #[must_use]
    pub fn fuzzy_syllables_at(&self, input: &str, pos: usize) -> Vec<SyllableEdge> {
        let mut out: Vec<SyllableEdge> = Vec::new();
        if let Some(first) = input[pos..].chars().next()
            && let Some(bucket) = self.fuzzy.by_first.get(&first)
        {
            let rest = &input[pos..];
            for variant in bucket {
                if !rest.starts_with(*variant) {
                    continue;
                }
                if let Some(canonicals) = self.fuzzy.map.get(*variant) {
                    for (syl, cost) in canonicals {
                        let edge = SyllableEdge {
                            end: pos + variant.len(),
                            syl,
                            cost: *cost,
                        };
                        if !out.contains(&edge) {
                            out.push(edge);
                        }
                    }
                }
            }
        }
        out
    }

    /// 所有以 `prefix` 开头的标准音节（不含 `prefix` 本身，即「可延长的更长沙节」）。
    #[must_use]
    pub fn syllables_with_prefix(&self, prefix: &str) -> Vec<&'static str> {
        let Some(first) = prefix.chars().next() else {
            return Vec::new();
        };
        let Some(bucket) = self.by_first.get(&first) else {
            return Vec::new();
        };
        bucket
            .iter()
            .filter(|s| s.starts_with(prefix) && **s != prefix)
            .copied()
            .collect()
    }

    /// 按标准音节切分并用空格连接，供预编辑显示（`nihaoshijie` → `ni hao shi jie`）。
    ///
    /// 取「音节数最少」的切分（与解码器的规模判定同一口径）；某个位置切不下去时，
    /// 把剩下的字母原样附上 —— 用户还没打完的尾巴（`shij` 的 `j`）也要看得见。
    /// 输入含音节分隔符 `'` 时先剥离再切（`xi'an` → `xi an`，边界强制）。
    #[must_use]
    pub fn split_for_display(&self, input: &str) -> String {
        let (clean, _, boundaries) = if input.contains(SYLLABLE_SEPARATOR) {
            strip_separators(input)
        } else {
            (input.to_string(), Vec::new(), Vec::new())
        };
        let n = clean.len();
        let lat = self.lattice_strict(&clean, &boundaries, false);
        // 1) 正向可达：能切出音节的最远位置（还没打完的尾巴切不出来，如 shij 的 j）
        let mut reachable = vec![false; n + 1];
        reachable[0] = true;
        let mut max_pos = 0usize;
        for pos in 0..n {
            if !reachable[pos] {
                continue;
            }
            for e in &lat[pos] {
                if e.is_fuzzy() {
                    continue; // 显示层只认精确读音：预编辑要如实反映用户打的内容
                }
                reachable[e.end] = true;
                max_pos = max_pos.max(e.end);
            }
        }
        // 2) 反向 DP 到 max_pos：音节数最少的切分（与解码器的规模判定同一口径）
        let mut hops = vec![usize::MAX; n + 1];
        let mut choice: Vec<Option<&'static str>> = vec![None; n + 1];
        hops[max_pos] = 0;
        for pos in (0..max_pos).rev() {
            for e in &lat[pos] {
                if e.is_fuzzy() || e.end > max_pos {
                    continue;
                }
                if hops[e.end] != usize::MAX && hops[e.end] + 1 < hops[pos] {
                    hops[pos] = hops[e.end] + 1;
                    choice[pos] = Some(e.syl);
                }
            }
        }
        // 3) 拼出「音节 音节 … 尾巴」
        let mut out = String::with_capacity(n + n / 2);
        let mut pos = 0usize;
        while pos < max_pos {
            let Some(syl) = choice[pos] else { break };
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&clean[pos..pos + syl.len()]);
            pos += syl.len();
        }
        if pos < n {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&clean[pos..]);
        }
        out
    }

    /// 全切分：`lattice[pos]` = 从 pos 开始的所有切分边。
    #[must_use]
    pub fn lattice(&self, input: &str) -> Vec<Vec<SyllableEdge>> {
        let mut out = Vec::with_capacity(input.len());
        for pos in 0..input.len() {
            out.push(self.syllables_at(input, pos));
        }
        out
    }

    /// 全切分（含模糊音）。
    #[must_use]
    pub fn lattice_fuzzy(&self, input: &str) -> Vec<Vec<SyllableEdge>> {
        let mut out = Vec::with_capacity(input.len());
        for pos in 0..input.len() {
            out.push(self.fuzzy_syllables_at(input, pos));
        }
        out
    }

    /// 全切分（含模糊音），并剪掉跨越强制边界的边（音节分隔符 `'`）。
    ///
    /// 剪边让假设在边界处天然断开：`xi'an` 的边界在 clean 坐标 2，
    /// `xian`（0..4）跨边界被剪，只剩 xi(0..2) + an(2..4)——
    /// `keys_at`/beam 完全不用感知分隔符。
    #[must_use]
    pub fn lattice_strict(
        &self,
        input: &str,
        boundaries: &[usize],
        fuzzy: bool,
    ) -> Vec<Vec<SyllableEdge>> {
        let mut lat = if fuzzy {
            self.lattice_fuzzy(input)
        } else {
            self.lattice(input)
        };
        for &b in boundaries {
            for pos in 0..b {
                lat[pos].retain(|e| e.end <= b);
            }
        }
        lat
    }
}

/// 把音节列表按首字母分桶（桶内长度降序，长音节优先匹配）。
fn bucket<'a>(syllables: &[&'a str]) -> HashMap<char, Vec<&'a str>> {
    let mut by_first: HashMap<char, Vec<&'a str>> = HashMap::new();
    for s in syllables {
        if let Some(c) = s.chars().next() {
            by_first.entry(c).or_default().push(s);
        }
    }
    for v in by_first.values_mut() {
        v.sort_by_key(|s| std::cmp::Reverse(s.len()));
    }
    by_first
}

/// 在桶中查找 `input[pos..]` 的精确匹配。
fn syllables_in_bucket(
    by_first: &HashMap<char, Vec<&'static str>>,
    input: &str,
    pos: usize,
) -> Vec<SyllableEdge> {
    let Some(first) = input[pos..].chars().next() else {
        return Vec::new();
    };
    let Some(bucket) = by_first.get(&first) else {
        return Vec::new();
    };
    let rest = &input[pos..];
    bucket
        .iter()
        .filter(|s| rest.starts_with(**s))
        .map(|s| SyllableEdge {
            end: pos + s.len(),
            syl: s,
            cost: 0,
        })
        .collect()
}

impl FuzzyTable {
    fn new() -> Self {
        // 变体用 'static 生命周期：进程常驻，Leak 一次无妨。
        let mut map: HashMap<&'static str, Vec<(&'static str, u8)>> = HashMap::new();
        let mut by_first: HashMap<char, Vec<&'static str>> = HashMap::new();
        // 第一遍：精确映射先入表（保证 `first()` 优先拿到标准读音，
        // 否则 f↔h 等规则会让 `fou` 遮蔽 `hou`，把 时候 拼成 是否）。
        for syl in SYLLABLES {
            let leaked: &'static str = Box::leak((*syl).into());
            map.entry(leaked).or_default().push((syl, 0));
            by_first.entry(leaked.chars().next().unwrap_or('a')).or_default().push(leaked);
        }
        // 第二遍：模糊变体补入（同一 变体→标准 由多条规则得到时保留最低代价）
        for syl in SYLLABLES {
            for (v, cost) in fuzzy_variants(syl) {
                let leaked: &'static str = Box::leak(v.into_boxed_str());
                let entry = map.entry(leaked).or_default();
                match entry.iter_mut().find(|(s, _)| s == syl) {
                    Some((_, c)) => *c = (*c).min(cost),
                    None => entry.push((syl, cost)),
                }
                if let Some(c) = leaked.chars().next() {
                    by_first.entry(c).or_default().push(leaked);
                }
            }
        }
        for v in by_first.values_mut() {
            v.sort_by_key(|s| std::cmp::Reverse(s.len()));
        }
        Self { map, by_first }
    }
}

/// 生成一个音节的所有模糊变体（每条规则最多一次替换），带规则代价等级。
fn fuzzy_variants(syl: &str) -> Vec<(String, u8)> {
    let mut out = Vec::new();
    for (a, b, cost) in FUZZY_RULES {
        // 方向 a→b：若 b 以 a 开头（前缀重叠，如 sh/s），跳过已被 b 占据的位置
        if let Some(pos) = syl.find(a)
            && !(a.len() < b.len() && syl[pos..].starts_with(b))
        {
            out.push((substitute(syl, pos, a.len(), b), *cost));
        }
        // 方向 b→a
        if let Some(pos) = syl.find(b)
            && !(b.len() < a.len() && syl[pos..].starts_with(a))
        {
            out.push((substitute(syl, pos, b.len(), a), *cost));
        }
    }
    out
}

fn substitute(syl: &str, pos: usize, len: usize, repl: &str) -> String {
    let mut out = String::with_capacity(syl.len() + repl.len());
    out.push_str(&syl[..pos]);
    out.push_str(repl);
    out.push_str(&syl[pos + len..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_table_size() {
        let t = SyllableTable::new();
        // 标准无调音节约 410 个
        let n = t.by_first.values().map(Vec::len).sum::<usize>();
        assert!(n > 400);
        assert!(n < 430);
    }

    #[test]
    fn segments_xianzai() {
        let t = SyllableTable::new();
        let lattice = t.lattice("xianzai");
        let starts: Vec<&str> = lattice[0].iter().map(|e| e.syl).collect();
        assert!(starts.contains(&"xian"));
        assert!(starts.contains(&"xi"));
        let reachable = reach(&lattice, "xianzai");
        assert!(reachable.iter().any(|s| s == "xian zai"));
        assert!(reachable.iter().any(|s| s == "xi an zai"));
    }

    #[test]
    fn segments_nihao() {
        let t = SyllableTable::new();
        let lattice = t.lattice("nihao");
        let reachable = reach(&lattice, "nihao");
        assert!(reachable.iter().any(|s| s == "ni hao"));
    }

    #[test]
    fn strip_separators_folds_and_maps() {
        let (clean, pos_map, boundaries) = strip_separators("xi'an");
        assert_eq!(clean, "xian");
        assert_eq!(pos_map, vec![0, 1, 3, 4]);
        assert_eq!(boundaries, vec![2]);
        // 连续分隔符折叠、首尾忽略
        let (clean, _, boundaries) = strip_separators("'xi''an'");
        assert_eq!(clean, "xian");
        assert_eq!(boundaries, vec![2]);
        // 无分隔符：边界空、映射恒等
        let (clean, pos_map, boundaries) = strip_separators("xian");
        assert_eq!(clean, "xian");
        assert_eq!(pos_map, vec![0, 1, 2, 3]);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn lattice_strict_forces_boundary() {
        let t = SyllableTable::new();
        // xian 本来能整切；加上边界 2 后只能 xi+an
        let strict = t.lattice_strict("xian", &[2], false);
        let plain = t.lattice("xian");
        assert!(plain[0].iter().any(|e| e.syl == "xian"), "无边界时可切 xian");
        assert!(
            strict[0].iter().all(|e| e.end <= 2),
            "跨边界边被剪: {:?}",
            strict[0]
        );
        assert!(
            strict[0].iter().any(|e| e.syl == "xi"),
            "xi 保留: {:?}",
            strict[0]
        );
        let reachable = reach(&strict, "xian");
        assert!(
            reachable.iter().any(|s| s == "xi an"),
            "强制切 xi an: {reachable:?}"
        );
        assert!(
            reachable.iter().all(|s| s != "xian"),
            "整切被剪掉: {reachable:?}"
        );
        // 模糊格同样剪边
        let strict_fuzzy = t.lattice_strict("xian", &[2], true);
        assert!(strict_fuzzy[0].iter().all(|e| e.end <= 2));
    }

    #[test]
    fn split_for_display_respects_separator() {
        let t = SyllableTable::new();
        assert_eq!(t.split_for_display("xi'an"), "xi an");
        assert_eq!(t.split_for_display("xian"), "xian");
        assert_eq!(t.split_for_display("nihaoshijie"), "ni hao shi jie");
    }

    #[test]
    fn fuzzy_maps_si_to_shi() {
        let t = SyllableTable::new();
        let edges = t.fuzzy_syllables_at("si", 0);
        assert!(
            edges.iter().any(|e| e.syl == "shi"),
            "si 应模糊匹配到 shi: {edges:?}"
        );
    }

    #[test]
    fn fuzzy_lattice_covers_fuzzy_input() {
        let t = SyllableTable::new();
        // 用户打 si(想输 是/时)，模糊格应能切出 shi 路径
        let lattice = t.lattice_fuzzy("sihou");
        let reachable = reach(&lattice, "sihou");
        assert!(
            reachable.iter().any(|s| s == "shi hou"),
            "sihou 应能切出 shi hou: {reachable:?}"
        );
    }

    #[test]
    fn split_for_display_segments_pinyin() {
        let t = SyllableTable::new();
        // 预编辑要看得见切分
        assert_eq!(t.split_for_display("nihaoshijie"), "ni hao shi jie");
        assert_eq!(t.split_for_display("womenzaigongzuo"), "wo men zai gong zuo");
        // 没打完的尾巴原样附上（shij 的 j）
        assert_eq!(t.split_for_display("nihaoshij"), "ni hao shi j");
        // 单音节 / 空串
        assert_eq!(t.split_for_display("ni"), "ni");
        assert_eq!(t.split_for_display(""), "");
        // 整串都切不出音节：原样返回
        assert_eq!(t.split_for_display("zzz"), "zzz");
    }

    #[test]
    fn fuzzy_cost_is_tiered() {
        let t = SyllableTable::new();
        let cost_of = |input: &str, syl: &str| {
            t.fuzzy_syllables_at(input, 0)
                .into_iter()
                .find(|e| e.syl == syl)
                .map(|e| e.cost)
        };
        // 精确匹配 cost=0
        assert_eq!(cost_of("shang", "shang"), Some(0));
        // 平翘舌（sh/s）= 1
        assert_eq!(cost_of("sang", "shang"), Some(1));
        // 前后鼻音韵尾（an/ang）= 2；边鼻音（n/l）= 2
        assert_eq!(cost_of("san", "sang"), Some(2));
        assert_eq!(cost_of("lan", "nan"), Some(2));
        // 其他声母混淆（k/g、f/h、r/l）= 3
        assert_eq!(cost_of("gou", "kou"), Some(3));
        assert_eq!(cost_of("fou", "hou"), Some(3));
    }

    #[test]
    fn fuzzy_does_not_break_exact() {
        let t = SyllableTable::new();
        let exact = t.lattice("nihao");
        let fuzzy = t.lattice_fuzzy("nihao");
        assert!(fuzzy.len() >= exact.len());
        assert!(
            reach(&fuzzy, "nihao").iter().any(|s| s == "ni hao")
        );
    }

    /// 枚举所有能到结尾的切分路径（测试用，字符串拼接）。
    fn reach(lattice: &[Vec<SyllableEdge>], input: &str) -> Vec<String> {
        let mut results = Vec::new();
        let mut stack: Vec<(usize, Vec<&str>)> = vec![(0, Vec::new())];
        while let Some((pos, path)) = stack.pop() {
            if pos == input.len() {
                results.push(path.join(" "));
                continue;
            }
            for edge in &lattice[pos] {
                let mut p = path.clone();
                p.push(edge.syl);
                stack.push((edge.end, p));
            }
        }
        results
    }
}
