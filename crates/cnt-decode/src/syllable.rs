//! 标准拼音音节表 + 切分（含模糊音）。
//!
//! 音节表是《汉语拼音方案》的无调音节全集（约 410 个），硬编码为标准数据，
//! 不依赖任何外部库。切分用全切分（动态规划），生成音节格（lattice）。
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

/// 模糊音规则（双向）：平翘舌 / 边鼻音 / 前后鼻音等。
const FUZZY_RULES: &[(&str, &str)] = &[
    ("zh", "z"),
    ("ch", "c"),
    ("sh", "s"),
    ("n", "l"),
    ("f", "h"),
    ("r", "l"),
    ("k", "g"),
    ("t", "d"),
    ("an", "ang"),
    ("en", "eng"),
    ("in", "ing"),
    ("ian", "iang"),
    ("uan", "uang"),
];

/// 一条切分边：结束位置 + 标准音节 + 是否模糊匹配（变体 → 标准）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyllableEdge {
    pub end: usize,
    pub syl: &'static str,
    pub fuzzy: bool,
}

/// 音节表：按首字母索引，查询 `input[pos..]` 处的合法音节。
pub struct SyllableTable {
    by_first: HashMap<char, Vec<&'static str>>,
    fuzzy: FuzzyTable,
}

/// 模糊变体表：`map[变体] = 可能的多个标准音节`（变体本身是标准音节时包含自己）。
struct FuzzyTable {
    map: HashMap<&'static str, Vec<&'static str>>,
    by_first: HashMap<char, Vec<&'static str>>,
}

impl Default for SyllableTable {
    fn default() -> Self {
        Self::new()
    }
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

    /// 从 `input[pos..]` 能匹配到的音节（含模糊变体；`fuzzy=true` 表示变体读音）。
    ///
    /// 例：输入 `sang` → 标准 `sang`（精确）与 `shang`（模糊，s↔sh）。
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
                    for syl in canonicals {
                        let edge = SyllableEdge {
                            end: pos + variant.len(),
                            syl,
                            fuzzy: variant != syl, // 变体≠标准 → 模糊
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
            fuzzy: false,
        })
        .collect()
}

impl FuzzyTable {
    fn new() -> Self {
        // 变体用 'static 生命周期：进程常驻，Leak 一次无妨。
        let mut map: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        let mut by_first: HashMap<char, Vec<&'static str>> = HashMap::new();
        // 第一遍：精确映射先入表（保证 `first()` 优先拿到标准读音，
        // 否则 f↔h 等规则会让 `fou` 遮蔽 `hou`，把 时候 拼成 是否）。
        for syl in SYLLABLES {
            let leaked: &'static str = Box::leak((*syl).into());
            map.entry(leaked).or_default().push(syl);
            by_first.entry(leaked.chars().next().unwrap_or('a')).or_default().push(leaked);
        }
        // 第二遍：模糊变体补入
        for syl in SYLLABLES {
            for v in fuzzy_variants(syl) {
                let leaked: &'static str = Box::leak(v.into_boxed_str());
                if !map.get(leaked).is_some_and(|l| l.contains(syl)) {
                    map.entry(leaked).or_default().push(syl);
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

/// 生成一个音节的所有模糊变体（每条规则最多一次替换）。
fn fuzzy_variants(syl: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (a, b) in FUZZY_RULES {
        // 方向 a→b：若 b 以 a 开头（前缀重叠，如 sh/s），跳过已被 b 占据的位置
        if let Some(pos) = syl.find(a)
            && !(a.len() < b.len() && syl[pos..].starts_with(b))
        {
            out.push(substitute(syl, pos, a.len(), b));
        }
        // 方向 b→a
        if let Some(pos) = syl.find(b)
            && !(b.len() < a.len() && syl[pos..].starts_with(a))
        {
            out.push(substitute(syl, pos, b.len(), a));
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
