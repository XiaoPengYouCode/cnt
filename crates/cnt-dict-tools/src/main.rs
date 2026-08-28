//! 词库工具 CLI。
//!
//! ```text
//! cnt-dict-tools build <in.tsv> <out.cntd>                编译词表为二进制
//! cnt-dict-tools import-rime-table <table.txt> <out.tsv>  Rime 词表 → 标准词表
//! cnt-dict-tools import-libime <dict.txt> <out.tsv>      libime 词表 → 标准词表
//! cnt-dict-tools build-lm <lm.arpa> <out.cntl>           ARPA 语言模型 → 二进制
//! cnt-dict-tools decode <dict.cntd> <lm.cntl> <pinyin>.. 整句解码（开发调试）
//!   decode 选项：--user <user.dict> 用真实用户库；--top <n> 候选条数；
//!   --tsv 机器可读输出（`拼音<TAB>排名<TAB>候选<TAB>分数<TAB>拼音键`），
//!   拼音键是解码器实际走的读音路径，供 fuzzy 测试判定精确/模糊/补全
//! cnt-dict-tools info  <dict.cntd>                       打印词库统计
//! cnt-dict-tools query <dict.cntd> <pinyin>..            查询候选
//! ```
//!
//! 标准词表格式（tsv，`#` 开头为注释，频率可省略默认 1000）：
//! ```text
//! nihao<TAB>你好<TAB>1000
//! ```

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use cnt_dict::{writer, MmapDict};

/// 各子命令的统一返回（CLI 错误聚合）。
type CliResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// `f64` → `u32` 频率：四舍五入并夹紧到 `[1, u32::MAX]`。
/// 转换器专用：权重/对数概率转整数频率，截断与符号丢失在此是预期语义。
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn freq_u32(v: f64) -> u32 {
    let v = v.clamp(1.0, f64::from(u32::MAX));
    v.round() as u32
}

/// libime 顶层词标记：dict 权重精确为 0（文本里就是写的 `0`，用精确比较是正确语义）。
#[allow(clippy::float_cmp)]
fn is_libime_top_tier(w: f64) -> bool {
    w == 0.0
}

/// 有 LM 路径的频率换算常量（log10 概率 → 频率的缩放）。
const LM_SCALE: f64 = 10_000.0;
/// 无 LM 路径（dict 权重）的频率换算常量。
const WEIGHT_SCALE: f64 = 100_000.0;
/// 「次读音」判定阈值：libime 主读音权重 ≈ 0（-0.0001 ~ 0），
/// 次读音显著更负（-2.7 ~ -5.3）。次读音用按读音权重，避免多音字串频
/// （的(de) vs 的(di)、和(he) vs 和(hu)）。
const SECONDARY_READING_THRESHOLD: f64 = -0.1;
/// LM 地板词的“常用词带”高度（0.5 个 log10 单位）。
const FLOOR_BOOST: u32 = 5_000;

/// 有 LM 时：词在 unigram 里 → 对数概率换算频率；否则长尾 `freq = 1`。
/// libime 顶层词（dict 权重 == 0）若是 LM 地板词（如 你好，LM 按 你+好
/// 二元组建模所以 unigram 概率在地板），抬到“常用词带”，避免排到生僻词后。
fn lm_freq(map: &std::collections::HashMap<String, f64>, p_min: f64, word: &str, dict_weight: Option<&str>) -> u32 {
    map.get(word).map_or(1, |p| {
        let raw = freq_u32((p - p_min) * LM_SCALE);
        let top_tier = dict_weight
            .and_then(|w| w.trim().parse::<f64>().ok())
            .is_some_and(is_libime_top_tier);
        if top_tier && raw < FLOOR_BOOST {
            FLOOR_BOOST
        } else {
            raw
        }
    })
}

/// 混合频率（读音感知）：
/// - **次读音**（权重 < `SECONDARY_READING_THRESHOLD`，如 的di=-3.5、和hu=-2.8）
///   用 libime 按读音权重 —— 多音字的次读音频率显著低于主读音，不再串频；
/// - **主读音**（权重 ≈ 0，如 的de、是/时/事）用 LM unigram —— 保留细粒度排序
///   （是 与 时 不并列）。
fn blend_freq(
    map: &std::collections::HashMap<String, f64>,
    p_min: f64,
    word: &str,
    dict_weight: Option<&str>,
    _min_w: f64,
) -> u32 {
    match dict_weight.and_then(|w| w.trim().parse::<f64>().ok()) {
        // 次读音（如 的di、和hu）：freq=1，与主读音的 LM 频率（≥1e4 量级）天然区分
        Some(w) if w < SECONDARY_READING_THRESHOLD => 1,
        _ => lm_freq(map, p_min, word, dict_weight),
    }
}

/// 读取已编译的 `.cntl` 语言模型（替代解析 ARPA 文本）。
fn read_cntl_unigrams(path: &str) -> CliResult<(std::collections::HashMap<String, f64>, f64)> {
    use cnt_lm::CntLm;
    let lm = CntLm::open(path)?;
    let mut map = std::collections::HashMap::new();
    let mut p_min = f64::MAX;
    for (word, logprob, _backoff) in lm.unigrams() {
        map.insert(word.to_string(), f64::from(logprob));
        p_min = p_min.min(f64::from(logprob));
    }
    Ok((map, p_min))
}

/// 无 LM 时：dict 权重 → 频率（多音字正权重按 0 封顶）；无权重长尾 `freq = 1`。
fn weight_freq(dict_weight: Option<&str>, min_w: f64) -> u32 {
    dict_weight
        .and_then(|w| w.trim().parse::<f64>().ok())
        .map_or(1, |w| freq_u32((w.min(0.0) - min_w) * WEIGHT_SCALE))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        std::process::exit(2);
    }
    // 除 bench（自带聚合 reporter）外，其余命令用 ConsoleReporter：
    // 诊断命令没有 reporter 等于埋了点也看不见（trace 关闭时这行是 noop）
    if args[1] != "bench" {
        fastrace::set_reporter(
            fastrace::collector::ConsoleReporter,
            fastrace::collector::Config::default(),
        );
    }
    let result = match args[1].as_str() {
        "build" => {
            if args.len() != 4 {
                usage();
                std::process::exit(2);
            }
            cmd_build(&args[2], &args[3])
        }
        "import-rime-table" => {
            if args.len() != 4 {
                usage();
                std::process::exit(2);
            }
            cmd_import_rime_table(&args[2], &args[3])
        }
        "import-libime" => {
            // import-libime <dict.txt> <out.tsv> [--lm <lm.arpa>]
            let lm = match args.get(4).map(String::as_str) {
                Some("--lm") if args.len() == 6 => Some(args[5].clone()),
                _ => {
                    if args.len() != 4 {
                        usage();
                        std::process::exit(2);
                    }
                    None
                }
            };
            cmd_import_libime(&args[2], &args[3], lm.as_deref())
        }
        "build-lm" => {
            if args.len() != 4 {
                usage();
                std::process::exit(2);
            }
            cmd_build_lm(&args[2], &args[3])
        }
        "bench" => {
            if args.len() < 4 {
                usage();
                std::process::exit(2);
            }
            // bench <dict> <lm> [--user <user.dict>] <n>
            let mut rest = &args[4..];
            let user = if rest.first().map(String::as_str) == Some("--user") && rest.len() >= 2 {
                let u = rest[1].clone();
                rest = &rest[2..];
                Some(u)
            } else {
                None
            };
            let n: usize = rest.first().map_or(50, |s| s.parse().unwrap_or(50));
            cmd_bench(&args[2], &args[3], user.as_deref(), n)
        }
        "decode" => {
            if args.len() < 4 {
                usage();
                std::process::exit(2);
            }
            // decode <dict> <lm> [--user <user.dict>] [--tsv] [--top <n>] <pinyin>...
            let (user, opts, rest) = parse_decode_args(&args[4..]);
            cmd_decode(&args[2], &args[3], user.as_deref(), rest, opts)
        }
        "info" => {
            if args.len() != 3 {
                usage();
                std::process::exit(2);
            }
            cmd_info(&args[2])
        }
        "query" => {
            if args.len() < 4 {
                usage();
                std::process::exit(2);
            }
            cmd_query(&args[2], &args[3..])
        }
        "delete-user" => {
            // delete-user <user.dict> <pinyin> <word>
            if args.len() != 5 {
                usage();
                std::process::exit(2);
            }
            cmd_delete_user(&args[2], &args[3], &args[4])
        }
        _ => {
            usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "usage:\n  cnt-dict-tools build <in.tsv> <out.cntd>\n  cnt-dict-tools import-rime-table <table.txt> <out.tsv>\n  cnt-dict-tools import-libime <dict.txt> <out.tsv> [--lm <lm.arpa>]\n  cnt-dict-tools build-lm <lm.arpa> <out.cntl>\n  cnt-dict-tools decode <dict.cntd> <lm.cntl> [--user <user.dict>] <pinyin>..\n  cnt-dict-tools info <dict.cntd>\n  cnt-dict-tools query <dict.cntd> <pinyin>...\n  cnt-dict-tools delete-user <user.dict> <pinyin> <word>"
    );
}

fn parse_wordlist(path: &str) -> io::Result<Vec<(String, String, u32)>> {
    let f = BufReader::new(File::open(path)?);
    let mut pairs = Vec::new();
    for (lineno, line) in f.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            [pinyin, word] => pairs.push((pinyin.to_string(), word.to_string(), 1000)),
            [pinyin, word, freq] => {
                let freq: u32 = freq.parse().map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("line {}: bad freq {freq:?}", lineno + 1),
                    )
                })?;
                pairs.push((pinyin.to_string(), word.to_string(), freq));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: expected 'pinyin word [freq]'", lineno + 1),
                ))
            }
        }
    }
    Ok(pairs)
}

fn cmd_build(input: &str, output: &str) -> CliResult {
    let pairs = parse_wordlist(input)?;
    writer::write_to_file(&pairs, Path::new(output))?;
    println!("built {} entries -> {output}", pairs.len());
    Ok(())
}

/// 把 `Rime` 编译产物（`luna_pinyin.table.txt`）转成标准词表。
///
/// Rime 格式：`词<TAB>拼音(空格分隔音节)<TAB>权重`。
/// 转换规则：
/// - 音节空格去掉（ni hao → nihao，与我们的 key 一致）
/// - 权重为浮点，四舍五入转 u32，≤0 的丢弃
/// - 拼音或词为空的丢弃
fn cmd_import_rime_table(input: &str, output: &str) -> CliResult {
    let f = BufReader::new(File::open(input)?);
    let mut out = BufWriter::new(File::create(output)?);
    let mut imported = 0u64;
    let mut skipped = 0u64;
    for line in f.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split('\t');
        let (Some(word), Some(pinyin), Some(weight)) = (it.next(), it.next(), it.next()) else {
            skipped += 1;
            continue;
        };
        if word.is_empty() || pinyin.is_empty() {
            skipped += 1;
            continue;
        }
        let freq = match weight.trim().parse::<f64>() {
            Ok(w) if w > 0.0 => freq_u32(w),
            _ => {
                skipped += 1;
                continue;
            }
        };
        let key: String = pinyin.split_whitespace().collect();
        if key.is_empty() {
            skipped += 1;
            continue;
        }
        writeln!(out, "{key}\t{word}\t{freq}")?;
        imported += 1;
    }
    out.flush()?;
    println!("imported {imported} entries ({skipped} skipped) -> {output}");
    Ok(())
}

/// 把 `libime`（`fcitx5`）拼音词典转成标准词表。
///
/// libime 格式：`词<TAB>拼音(音节用 ' 分隔)<TAB>权重`，权重可省略。
///
/// 权重说明：dict.txt 里的权重并不可靠（0 是最常用基准，但多音字默认读音
/// 会被给到正权重，如 螫+0.22 反而超过 是），真正决定候选顺序的是语言模型。
/// 因此推荐传 `--lm lm.arpa`：用 ARPA 1-gram 的对数概率排序（是 -1.94，
/// 时 -2.88，螫 -6.17），顺序完全正确。
///
/// 转换规则：
/// - 去掉音节分隔符 `'`（ni'hao → nihao，与我们的 key 一致）
/// - 有 `--lm`：`freq = round((p - p_min) * 1e4) + 1`，不在 `LM` 中的长尾词 `freq = 1`
/// - 无 `--lm`：`freq = round((min(w,0) - min_w) * 1e5) + 1`，无权重条目 `freq = 1`
fn cmd_import_libime(input: &str, output: &str, lm: Option<&str>) -> CliResult {

    // 语言模型 unigram（若提供）：.arpa 文本或已编译的 .cntl
    let unigrams: Option<(std::collections::HashMap<String, f64>, f64)> = match lm {
        Some(path) if path.to_ascii_lowercase().ends_with(".cntl") => Some(read_cntl_unigrams(path)?),
        Some(path) => Some(read_arpa_unigrams(path)?),
        None => None,
    };

    // 先扫一遍找最小权重（次读音换算需要；多音字正权重按 0 封顶）。
    // 哨兵用 f64::MAX：NEG_INFINITY.min(x) 恒为 -inf（负无穷即最小值），是经典陷阱。
    let min_w = {
        let mut min_w = f64::MAX;
        let f = BufReader::new(File::open(input)?);
        for line in f.lines() {
            let line = line?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 2 || fields[1].is_empty() {
                continue;
            }
            if let Some(w) = fields.get(2)
                && let Ok(w) = w.trim().parse::<f64>()
            {
                min_w = min_w.min(w.min(0.0));
            }
        }
        if min_w.is_finite() {
            min_w
        } else {
            0.0
        }
    };

    // 转换输出
    let f = BufReader::new(File::open(input)?);
    let mut out = BufWriter::new(File::create(output)?);
    let mut imported = 0u64;
    let mut skipped = 0u64;
    for line in f.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let (Some(word), Some(pinyin)) = (fields.first(), fields.get(1)) else {
            skipped += 1;
            continue;
        };
        if word.is_empty() || pinyin.is_empty() {
            skipped += 1;
            continue;
        }
        let key: String = pinyin.chars().filter(|c| *c != '\'').collect();
        if key.is_empty() {
            skipped += 1;
            continue;
        }
        let freq = match &unigrams {
            Some((map, p_min)) => blend_freq(map, *p_min, word, fields.get(2).copied(), min_w),
            None => weight_freq(fields.get(2).copied(), min_w),
        };
        writeln!(out, "{key}\t{word}\t{freq}")?;
        imported += 1;
    }
    out.flush()?;
    println!("imported {imported} entries ({skipped} skipped) -> {output}");
    Ok(())
}

/// 读取 ARPA 语言模型的 1-grams 部分（在 2-grams 处停止，只解析 unigram）。
/// 返回 (word -> log10 概率, 概率最小值)。
fn read_arpa_unigrams(path: &str) -> io::Result<(std::collections::HashMap<String, f64>, f64)> {
    let f = BufReader::new(File::open(path)?);
    let mut map = std::collections::HashMap::new();
    let mut p_min = f64::MAX;
    let mut in_unigrams = false;
    for line in f.lines() {
        let line = line?;
        if line.starts_with("\\1-grams:") {
            in_unigrams = true;
            continue;
        }
        if line.starts_with("\\2-grams:") {
            break;
        }
        if !in_unigrams || line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(p), Some(word)) = (it.next(), it.next()) else {
            continue;
        };
        if word == "<unk>" {
            continue;
        }
        let Ok(p) = p.parse::<f64>() else {
            continue;
        };
        map.insert(word.to_string(), p);
        p_min = p_min.min(p);
    }
    Ok((map, p_min))
}

fn cmd_info(path: &str) -> CliResult {
    let d = MmapDict::open(path)?;
    println!("dict: {path}");
    println!("entries: {}", d.entry_count());
    println!("candidates: {}", d.cand_count());
    // 打印前 5 个 key 验证
    println!("first keys:");
    for i in 0..d.entry_count().min(5) {
        println!("  {:?}", d.key_at(i));
    }
    Ok(())
}

fn cmd_query(path: &str, pinyins: &[String]) -> CliResult {
    let d = MmapDict::open(path)?;
    for pinyin in pinyins {
        print!("{pinyin}: ");
        let exact = d.exact(pinyin);
        let prefix = d.prefix(pinyin);
        let mut out: Vec<String> = Vec::new();
        out.extend(exact.iter().map(|c| c.word.to_string()));
        for h in prefix.iter().filter(|h| h.key != pinyin) {
            out.push(format!("{}[{}]", h.word, h.key));
        }
        if out.is_empty() {
            println!("(no candidates)");
        } else {
            println!("{}", out.join(" "));
        }
    }
    Ok(())
}

/// 从用户词库删除一条 (拼音, 词)：撤销误学（对应 Ctrl+Delete 的 CLI 形式）。
fn cmd_delete_user(path: &str, pinyin: &str, word: &str) -> CliResult {
    let mut db = cnt_dict::UserDb::open(path)?;
    if db.delete(pinyin, word) {
        db.flush()?;
        println!("deleted: {pinyin}\t{word}");
    } else {
        println!("not found: {pinyin}\t{word}");
    }
    Ok(())
}

/// 把 ARPA 文本语言模型编译成 `.cntl` 二进制。
///
/// 只解析 unigram 与 bigram 段（到 `\\3-grams:` 停止），`<unk>` 等伪词跳过。
fn cmd_build_lm(input: &str, output: &str) -> CliResult {
    use cnt_lm::writer::{build, Bigram, Unigram};

    let f = BufReader::new(File::open(input)?);
    let mut unigrams: Vec<Unigram> = Vec::new();
    let mut bigrams: Vec<Bigram> = Vec::new();
    let mut section = 0u8; // 0=头, 1=1-grams, 2=2-grams
    for line in f.lines() {
        let line = line?;
        if line.starts_with("\\1-grams:") {
            section = 1;
            continue;
        }
        if line.starts_with("\\2-grams:") {
            section = 2;
            continue;
        }
        if line.starts_with("\\3-grams:") || line.starts_with("\\end\\") {
            break;
        }
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match section {
            1 => {
                // logprob word [backoff]
                let [logprob, word, ..] = fields.as_slice() else {
                    continue;
                };
                if *word == "<unk>" {
                    continue;
                }
                let backoff = fields.get(2).map_or(0.0, |b| b.parse().unwrap_or(0.0));
                unigrams.push(Unigram {
                    word: word.to_string(),
                    logprob: logprob.parse()?,
                    backoff,
                });
            }
            2 => {
                // logprob w1 w2 [backoff]
                let [logprob, w1, w2, ..] = fields.as_slice() else {
                    continue;
                };
                bigrams.push(Bigram {
                    w1: w1.to_string(),
                    w2: w2.to_string(),
                    logprob: logprob.parse()?,
                });
            }
            _ => {}
        }
    }

    let bytes = build(&unigrams, &bigrams)?;
    std::fs::write(Path::new(output), bytes)?;
    println!(
        "built {} unigrams, {} bigrams -> {output}",
        unigrams.len(),
        bigrams.len()
    );
    Ok(())
}

/// 整句解码（开发调试用）：加载词库 + 语言模型，对给定拼音串解码 Top-K 句子。
/// 解析 `decode` 的可选参数，返回 (用户库, 输出选项, 剩余的拼音列表)。
fn parse_decode_args(mut rest: &[String]) -> (Option<String>, DecodeOpts, &[String]) {
    let mut opts = DecodeOpts::default();
    let mut user: Option<String> = None;
    loop {
        match rest.first().map(String::as_str) {
            Some("--user") if rest.len() >= 2 => {
                user = Some(rest[1].clone());
                rest = &rest[2..];
            }
            Some("--top") if rest.len() >= 2 => {
                opts.top = rest[1].parse().unwrap_or(opts.top);
                rest = &rest[2..];
            }
            Some("--tsv") => {
                opts.tsv = true;
                rest = &rest[1..];
            }
            _ => return (user, opts, rest),
        }
    }
}

/// `decode` 子命令的输出选项。
#[derive(Clone, Copy)]
struct DecodeOpts {
    /// 每个输入打印多少候选
    top: usize,
    /// 机器可读输出（TSV，含解码器实际走的拼音键）
    tsv: bool,
}

impl Default for DecodeOpts {
    fn default() -> Self {
        Self { top: 10, tsv: false }
    }
}

fn cmd_decode(
    dict_path: &str,
    lm_path: &str,
    user_path: Option<&str>,
    pinyins: &[String],
    opts: DecodeOpts,
) -> CliResult {
    use cnt_decode::Decoder;
    use cnt_dict::PinyinModel;
    use cnt_lm::CntLm;

    // 用户库：未指定时用临时空文件（不污染真实用户数据）
    let tmp_user = std::env::temp_dir().join(format!("cnt-decode-{}.dict", std::process::id()));
    let user_str = user_path.unwrap_or_else(|| tmp_user.to_str().unwrap());
    let model = std::sync::Arc::new(PinyinModel::open(dict_path, user_str)?);
    let lm = std::sync::Arc::new(CntLm::open(lm_path)?);
    let decoder = Decoder::new(model, Some(lm), true);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for pinyin in pinyins {
        if !opts.tsv {
            writeln!(out, "{pinyin}:")?;
        }
        // 每次解码一棵 root span：decode 是诊断命令，没有 span 树等于瞎子
        // （`--features "fastrace/enable"` 构建时才有输出）
        let root = fastrace::Span::root(
            format!("decode:{pinyin}"),
            fastrace::collector::SpanContext::random(),
        );
        let guard = root.set_local_parent();
        let cands = decoder.candidates_scored(pinyin);
        drop(guard);
        drop(root);
        for (i, (cand, score)) in cands.iter().take(opts.top).enumerate() {
            // 拼音键 = 解码器实际走的读音路径（整句为多段，用 '-' 连接）；
            // fuzzy 测试据此区分「精确读音 / 模糊回退 / 尾音节补全」，
            // 不必在测试脚本里重复实现模糊规则表。
            let keys: Vec<&str> = cand.learned.iter().map(|l| l.pinyin.as_str()).collect();
            let keys = keys.join("-");
            // 覆盖长度：小于输入长度 = 部分候选（选中后只确认这一段）
            let cover = if cand.consumed >= pinyin.len() {
                "full".to_string()
            } else {
                format!("part:{}", &pinyin[..cand.consumed])
            };
            if opts.tsv {
                writeln!(out, "{pinyin}\t{}\t{}\t{score:.3}\t{keys}\t{cover}", i + 1, cand.text)?;
            } else {
                writeln!(out, "  {}. {}  [{score:.3}] {{{keys}}} {cover}", i + 1, cand.text)?;
            }
        }
    }
    out.flush()?;
    fastrace::flush();
    let _ = std::fs::remove_file(&tmp_user);
    Ok(())
}

/// 单个 span 名的耗时聚合（ns）。
#[derive(Default)]
struct BenchAgg {
    count: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    /// span 属性（工作量指标，如 beam 的 expand 次数）。
    props: Vec<(String, String)>,
}

/// fastrace 聚合统计（Arc<Mutex> 共享：后台上报线程写，bench 主线程 flush 后读）。
#[derive(Default)]
struct BenchStats {
    by_name: std::collections::HashMap<String, BenchAgg>,
    expand_total: u64,
}

/// fastrace 聚合 reporter：按 span 名汇总 duration，flush 后由 bench 打印阶段分布表。
/// `ConsoleReporter` 逐条 Debug 输出在批量解码下不可读，聚合表更适合性能判断。
struct AggregateReporter {
    stats: std::sync::Arc<std::sync::Mutex<BenchStats>>,
}

impl fastrace::collector::Reporter for AggregateReporter {
    fn report(&mut self, spans: Vec<fastrace::collector::SpanRecord>) {
        let mut st = self.stats.lock().unwrap();
        for s in spans {
            let e = st.by_name.entry(s.name.to_string()).or_default();
            e.count += 1;
            e.total_ns += s.duration_ns;
            e.max_ns = e.max_ns.max(s.duration_ns);
            e.min_ns = if e.min_ns == 0 || s.duration_ns < e.min_ns {
                s.duration_ns
            } else {
                e.min_ns
            };
            // 工作量指标现在是 span 属性（不再是事件），见 AGENTS.md 埋点约定
            if e.props.is_empty() && !s.properties.is_empty() {
                e.props = s
                    .properties
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
            }
            let expand: u64 = s
                .properties
                .iter()
                .find(|(k, _)| k == "expand")
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0);
            st.expand_total += expand;
        }
    }
}

/// `u64` 纳秒计数远小于 2^53，转 `f64` 无损。
#[allow(clippy::cast_precision_loss)]
fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1000.0
}

/// 性能基准：对一组常见拼音反复解码，报告每次候选生成的延迟统计。
/// 冷启动延迟：每次解码前清空词键缓存，返回排序后的耗时样本。
fn measure_cold(
    decoder: &cnt_decode::Decoder,
    samples: &[&str],
    rounds: usize,
) -> Vec<std::time::Duration> {
    use cnt_input::CandidateSource;
    let mut out = Vec::with_capacity(samples.len() * rounds);
    for _ in 0..rounds {
        for s in samples {
            decoder.clear_cache();
            let t = std::time::Instant::now();
            let _ = decoder.candidates(s);
            out.push(t.elapsed());
        }
    }
    out.sort_unstable();
    out
}

/// 打印 fastrace 阶段分解表（每样例首轮 span 树的聚合）。
///
/// 工作量指标以 **span 属性**形式一并打出（`expand=…`），见 AGENTS.md 的埋点约定。
fn print_stage_table(agg: &std::sync::Arc<std::sync::Mutex<BenchStats>>) {
    let stats = agg.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    println!();
    println!("fastrace 阶段分解（每样例首轮 span 树聚合）：");
    let mut names: Vec<&String> = stats.by_name.keys().collect();
    names.sort_by(|a, b| stage_rank(a).cmp(&stage_rank(b)).then(a.cmp(b)));
    for name in names {
        let a = &stats.by_name[name];
        let avg = ns_to_us(a.total_ns / a.count.max(1));
        let min = ns_to_us(a.min_ns);
        let max = ns_to_us(a.max_ns);
        let props = if a.props.is_empty() {
            String::new()
        } else {
            let kv: Vec<String> = a.props.iter().map(|(k, v)| format!("{k}={v}")).collect();
            format!("  [{}]", kv.join(" "))
        };
        println!(
            "  {name:<28} n={:<3} avg={avg:>8.1}µs  min={min:>7.1}µs  max={max:>8.1}µs{props}",
            a.count
        );
    }
    if stats.expand_total > 0 {
        println!("  展开次数合计（beam 属性）      {}", stats.expand_total);
    }
    drop(stats);
}

/// 阶段在表里的排序（按链路顺序，读起来像一条流水线）。
fn stage_rank(name: &str) -> usize {
    if name.starts_with("decode:") {
        return 0;
    }
    match name {
        "candidates" => 1,
        "beam" => 2,
        "keys_at" => 3,
        "lattice" => 4,
        "completions" => 5,
        _ => 6,
    }
}

fn cmd_bench(dict_path: &str, lm_path: &str, user_path: Option<&str>, n: usize) -> CliResult {
    use cnt_decode::Decoder;
    use cnt_input::CandidateSource;
    use cnt_dict::PinyinModel;
    use cnt_lm::CntLm;
    use std::time::Instant;

    let tmp_user = std::env::temp_dir().join(format!("cnt-decode-{}.dict", std::process::id()));
    let user_str = user_path.unwrap_or_else(|| tmp_user.to_str().unwrap());
    let model = std::sync::Arc::new(PinyinModel::open(dict_path, user_str)?);
    let lm = std::sync::Arc::new(CntLm::open(lm_path)?);
    let decoder = Decoder::new(model, Some(lm), true);

    // fastrace：聚合 reporter 输出每次解码的 span 树（lattice/beam/completions 阶段耗时）
    // 到 stdout；每样例第 1 轮完整上报，其余轮 cancel 避免统计重复。
    let agg = std::sync::Arc::new(std::sync::Mutex::new(BenchStats::default()));
    fastrace::set_reporter(
        AggregateReporter { stats: agg.clone() },
        fastrace::collector::Config::default(),
    );

    // 常见输入样例（覆盖单音节/多音节/补全/模糊音路径；`l`/`zh` = 纯声母，
    // 每个词的第一键，走「整串都还不是音节」的最宽补全展开）
    let samples = [
        "ni", "nihao", "womenzaigongzuo", "xianzai", "diyige", "sihou", "chijiuhu",
        "zhongguoren", "momingqimiao", "shijie", "womendoushizhongguoren", "xiexieni",
        "jintian", "diannao", "shurufa", "nuli", "leng", "le", "l", "zh",
    ];

    // 预热（加载页缓存等）
    for _ in 0..3 {
        for s in samples {
            decoder.candidates(s);
        }
    }

    // 冷启动延迟：每次解码前清空词键缓存（首次进入某个输入上下文的延迟）。
    // 与「热」数值一起报，避免缓存把基准做得比真实体验好看。
    // 冷样本方差大（依赖页缓存/分支状态），多跑几轮取分位数才稳
    let cold = measure_cold(&decoder, &samples, 10);

    let mut latencies = Vec::with_capacity(samples.len() * n);
    let mut total_keys = 0usize;
    for round in 0..n {
        for s in samples {
            // fastrace root span：每轮一棵树；仅第 1 轮上报，其余 cancel。
            let root = fastrace::Span::root(
                format!("decode:{s}"),
                fastrace::collector::SpanContext::random(),
            );
            let _guard = root.set_local_parent();
            let t = Instant::now();
            let cands = decoder.candidates(s);
            latencies.push(t.elapsed());
            total_keys += cands.len();
            if round != 0 {
                root.cancel();
            }
        }
    }
    fastrace::flush();

    latencies.sort();
    let count = latencies.len();
    let avg = latencies.iter().sum::<std::time::Duration>() / u32::try_from(count).unwrap();
    let p50 = latencies[count / 2];
    let p90 = latencies[count * 9 / 10];
    let p99 = latencies[count * 99 / 100];
    let max = latencies[count - 1];
    println!(
        "{} 次解码（{} 个样例 × {n} 轮），共 {total_keys} 个候选",
        count, samples.len()
    );
    println!("热（词键缓存命中，= 连续打字的第 2 键起）：");
    println!("  平均 {avg:?}  中位 {p50:?}  90% {p90:?}  99% {p99:?}  最差 {max:?}");
    let cold_avg = cold.iter().sum::<std::time::Duration>() / u32::try_from(cold.len()).unwrap();
    println!(
        "冷（每次清空词键缓存，= 首次进入该输入上下文）：\n  平均 {cold_avg:?}  中位 {:?}  最差 {:?}",
        cold[cold.len() / 2],
        cold[cold.len() - 1]
    );

    print_stage_table(&agg);

    let _ = std::fs::remove_file(&tmp_user);
    Ok(())
}
