#!/usr/bin/env python3
"""输入法候选质量 fuzzy 测试（详见同目录 SKILL.md）。

从词库随机取样拼音 → 跑解码器 → 判定候选质量，输出可对比的回归基准。

判定依据来自解码器自己：`cnt-dict-tools decode --tsv` 会输出每条候选**实际走的
拼音键**，所以脚本不需要复制模糊音规则表，也不会把合法的模糊/补全候选误报成
读音误配（这是手写判定最容易出错的地方）：

    第 7 列 correction      → 独立验证一次键盘编辑（兼容旧 6 列输出）
    键拼接 == 输入          → 精确读音
    键拼接 != 输入、等长     → 模糊回退（si→shi）
    输入是键拼接的前缀       → 尾音节补全（chijiu → chijiuceng）
    其余                    → 读音误配（最高优先级 bug）

用法：
    python3 fuzzy_test.py sample [--n 100] [--seed 42] [--user <user.dict>]
    python3 fuzzy_test.py mono   [--user <user.dict>]     # 常用单音节专项
    python3 fuzzy_test.py words 里 力 ...                 # 指定拼音看候选来源

改动代码后先 `cargo build --release -p cnt-dict-tools`，再跑同一 --seed 对比。
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import subprocess
import sys

# 仓库根目录（本脚本位于 .agents/skills/fuzzy-test/）
ROOT = pathlib.Path(__file__).resolve().parents[3]
TOOL = ROOT / "target/release/cnt-dict-tools"
DICT = ROOT / "data/cnt.dict"
LM = ROOT / "data/lm.cntl"
WORDLIST = ROOT / "data/wordlist.tsv"

TOP = 10
# 常用单音节专项：覆盖高频音节（含用户报过问题的 li/jian/hu/xian/jie/bei）
MONO_SYLLABLES = (
    "de le shi yi bu zai you ta wo men zhe ge shuo dao xiang "
    "li jian hu xian jie bei zhi wei lai guo tian da xin fa xia"
).split()
# 单音节专项的参考集大小（词库高频前 N 字应进候选前 TOP）
MONO_REF = 10
# 「常用词」的词频门槛（取样时保证一半以上是常用词）
COMMON_FREQ = 10

EXACT, FUZZY, COMPLETION, PARTIAL, MISMATCH = "精确", "模糊", "补全", "部分", "误配"
CORRECTION = "纠错"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_wordlist() -> list[tuple[str, str, int]]:
    if not WORDLIST.exists():
        die(f"缺少词表 {WORDLIST}（data/ 不入库，见 AGENTS.md）")
    rows = []
    with WORDLIST.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                rows.append((parts[0], parts[1], int(parts[2])))
            except ValueError:
                continue
    return rows


def decode(pinyins: list[str], user: str | None, top: int = TOP) -> dict[str, list[tuple[str, float, str, str, str]]]:
    """跑解码器，返回 {输入: [(候选, 分数, 拼音键, 覆盖, 来源)]}（顺序即候选顺序）。

    覆盖列是 `full` 或 `part:<被消耗的输入前缀>` —— 部分候选（Rime 式增量确认的
    修复入口）只覆盖输入的前一段，判定读音时要拿那一段比，而不是整串输入。
    """
    if not TOOL.exists():
        die(f"缺少 {TOOL}，先 cargo build --release -p cnt-dict-tools")
    for path in (DICT, LM):
        if not path.exists():
            die(f"缺少 {path}（data/ 不入库，见 AGENTS.md）")
    cmd = [str(TOOL), "decode", str(DICT), str(LM)]
    if user:
        cmd += ["--user", user]
    cmd += ["--tsv", "--top", str(top), *pinyins]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        die(f"decode 失败: {proc.stderr.strip()}")
    out: dict[str, list[tuple[str, float, str, str, str]]] = collections.defaultdict(list)
    for line in proc.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) not in (6, 7):
            continue
        pinyin, _rank, text, score, keys, cover = cols[:6]
        source = cols[6] if len(cols) == 7 else ""
        out[pinyin].append((text, float(score), keys, cover, source))
    return out


# 模糊音的规范化：把互为模糊的写法折叠成同一形式，用来区分「模糊回退」与「读音误配」。
# 与 cnt-decode 的 FUZZY_RULES 对应。
#
# 全串替换（不做音节切分）是有意的：拼音里 f/h/k/t/d/r/l/c/s 只可能出现在声母位置，
# 唯一的例外是韵尾 n 与 ang/eng/ing 的 g —— 韵母规则先跑掉 g，n→l 则在两侧同样
# 施加，等价性比较仍然成立。这样就不必在测试脚本里复制 410 音节表做切分。
# 折叠顺序有讲究（见 canon 的注释）：声母簇 → 安全声母 → 韵母（迭代到不动点）→ n/l
CANON_CLUSTERS = (("zh", "z"), ("ch", "c"), ("sh", "s"))
CANON_INITIALS = (("f", "h"), ("k", "g"), ("t", "d"), ("r", "l"))
CANON_FINALS = (("iang", "ian"), ("uang", "uan"), ("ang", "an"), ("eng", "en"), ("ing", "in"))


def canon(keys: str) -> str:
    """折叠模糊写法：`zhuchen`/`zucheng`、`shenkong`/`shengong`、`hanggui`/`hangui`
    都归到同一串。

    全串替换（不切音节）是有意的：拼音里 f/k/t/r/c/s 只可能出现在声母位置，唯一的
    例外是韵尾 n 与 ang/eng/ing 的 g。因此：

    1. 先折声母簇（zh/ch/sh），再折 f→h、k→g、t→d、r→l —— k→g 要在韵母规则之前，
       否则 `shenkong` 的 `kong` 变成 `gong` 后就错过了 eng/en 的折叠；
    2. 韵母规则迭代到**不动点**：`hanggui` 一次替换只会变成 `hangui`（仍含 ang），
       与 `han-gui` 不等，必须再折一轮；
    3. n→l 放最后：它会破坏韵尾 n（`an` → `al`），但两侧同样施加，等价性仍成立。

    这样就不必在测试脚本里复制 410 音节表做切分。
    """
    flat = keys.replace("-", "")
    for a, b in CANON_CLUSTERS + CANON_INITIALS:
        flat = flat.replace(a, b)
    for _ in range(len(flat)):  # 韵母折叠到不动点（长度是替换次数的天然上界）
        folded = flat
        for a, b in CANON_FINALS:
            folded = folded.replace(a, b)
        if folded == flat:
            break
        flat = folded
    return flat.replace("n", "l")


def one_typo(raw: str, canonical: str) -> bool:
    """独立验证纠错标签：恰好一次同行邻键、交换或重复字母删除。"""
    if not 4 <= len(raw) <= 48 or not raw.isascii() or not raw.isalpha() or not raw.islower():
        return False
    if len(raw) == len(canonical):
        changed = [i for i, (a, b) in enumerate(zip(raw, canonical)) if a != b]
        if len(changed) == 1:
            i = changed[0]
            return any(raw[i] in row and canonical[i] in row
                       and abs(row.index(raw[i]) - row.index(canonical[i])) == 1
                       for row in ("qwertyuiop", "asdfghjkl", "zxcvbnm"))
        if len(changed) == 2:
            i, j = changed
            return j == i + 1 and raw[i] == canonical[j] and raw[j] == canonical[i]
    if len(raw) == len(canonical) + 1:
        return any(raw[i] == raw[i + 1] and raw[:i] + raw[i + 1:] == canonical
                   for i in range(len(raw) - 1))
    return False


def classify(pinyin: str, keys: str, cover: str = "full", source: str = "") -> str:
    """按解码器实际走的拼音键判定候选来源。

    判定顺序要紧：`zhuchen → 著称(zhucheng)` 既像「补全一个 g」又像 en/eng 模糊，
    先做模糊归一才不会把模糊候选记成补全。

    `cover` 是解码器给的覆盖信息：部分候选只覆盖输入的前一段，拿那一段做判定
    （否则「你好」对 nihaoshijie 会被误报成读音误配）。
    """
    if cover.startswith("part:"):
        covered = cover[len("part:"):]
        return PARTIAL if classify(covered, keys) in (EXACT, FUZZY) else MISMATCH
    flat = keys.replace("-", "")
    if source == "correction":
        return CORRECTION if cover == "full" and one_typo(pinyin, flat) else MISMATCH
    if flat == pinyin:
        return EXACT
    if canon(keys) == canon(pinyin):
        return FUZZY
    if flat.startswith(pinyin) or canon(keys).startswith(canon(pinyin)):
        return COMPLETION
    return MISMATCH


def cmd_sample(args: argparse.Namespace) -> int:
    rows = load_wordlist()
    rng = random.Random(args.seed)
    common = [r for r in rows if r[2] > COMMON_FREQ]
    half = args.n // 2
    sample = rng.sample(common, min(half, len(common))) + rng.sample(rows, args.n - half)
    cands = decode([s[0] for s in sample], args.user)

    hit1 = hit5 = hit10 = 0
    missing: list[tuple[str, str, int, list[str]]] = []
    kinds: collections.Counter[str] = collections.Counter()
    mismatches: list[tuple[str, str, str, int]] = []
    common_late: list[tuple[str, str, int, int, list[str]]] = []

    for pinyin, word, freq in sample:
        got = cands.get(pinyin, [])
        texts = [t for t, _, _, _, _ in got]
        rank = texts.index(word) + 1 if word in texts else None
        if rank == 1:
            hit1 += 1
        if rank and rank <= 5:
            hit5 += 1
        elif rank and freq > COMMON_FREQ:
            common_late.append((pinyin, word, freq, rank, texts[:5]))
        if rank:
            hit10 += 1
        else:
            missing.append((pinyin, word, freq, texts[:5]))
        for i, (text, _score, keys, cover, source) in enumerate(got, start=1):
            kind = classify(pinyin, keys, cover, source)
            kinds[kind] += 1
            if kind == MISMATCH:
                mismatches.append((pinyin, text, keys, i))

    total_c = max(sum(kinds.values()), 1)
    print(f"fuzzy 结果（{len(sample)} 词，seed={args.seed}，"
          f"{'用户库 ' + args.user if args.user else '空用户库'}）：")
    print(f"  #1 命中 {hit1} | 前5 {hit5} | 前10 {hit10} | 未出现 {len(missing)}")
    print("  候选来源分布（前 %d）：%s" % (
        TOP,
        " ".join(f"{k} {kinds[k]}({kinds[k] * 100 // total_c}%)"
                 for k in (EXACT, FUZZY, COMPLETION, CORRECTION, PARTIAL, MISMATCH)),
    ))
    if mismatches:
        print(f"  读音误配 {len(mismatches)} 处（最高优先级 bug）：")
        for pinyin, text, keys, i in mismatches[:10]:
            print(f"    {pinyin:14} 「{text}」键={keys} (#{i})")
    if common_late:
        print(f"  常用词前 5 未命中 {len(common_late)}：")
        for pinyin, word, freq, rank, top5 in common_late[:8]:
            print(f"    {pinyin:14} {word}(freq {freq}) 排名 {rank}   候选: {' '.join(top5)}")
    if missing:
        print(f"  未出现 {len(missing)}：")
        for pinyin, word, freq, top5 in missing[:8]:
            print(f"    {pinyin:14} {word}(freq {freq})   候选: {' '.join(top5)}")
    # 判定基准（SKILL.md）：误配 ≈ 0 是硬指标
    return 1 if mismatches else 0


def cmd_mono(args: argparse.Namespace) -> int:
    rows = load_wordlist()
    by_pinyin: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for pinyin, word, freq in rows:
        if len(word) == 1:
            by_pinyin[pinyin].append((word, freq))
    cands = decode(MONO_SYLLABLES, args.user)

    total = hit = 0
    rows_out: list[tuple[str, str, list[str]]] = []
    for syl in MONO_SYLLABLES:
        ref = [w for w, _ in sorted(by_pinyin.get(syl, []), key=lambda x: -x[1])[:MONO_REF]]
        got = [t for t, _, _, _, _ in cands.get(syl, [])][:TOP]
        total += len(ref)
        hit += sum(1 for w in ref if w in got)
        missed = [w for w in ref if w not in got]
        if missed:
            rows_out.append((syl, "".join(missed), got))
    pct = hit * 100 // max(total, 1)
    print(f"单音节专项（{len(MONO_SYLLABLES)} 个常用音节，"
          f"{'用户库' if args.user else '空用户库'}）：")
    print(f"  词库高频前 {MONO_REF} 字进候选前 {TOP}：{hit}/{total} = {pct}%")
    for syl, missed, got in rows_out:
        print(f"    {syl:5} 缺 {missed:12} 候选: {' '.join(got)}")
    return 0


def cmd_words(args: argparse.Namespace) -> int:
    cands = decode(args.pinyins, args.user)
    for pinyin in args.pinyins:
        print(f"{pinyin}:")
        for i, (text, score, keys, cover, source) in enumerate(cands.get(pinyin, []), start=1):
            kind = classify(pinyin, keys, cover, source)
            print(f"  {i:2}. {text:8} [{score:8.3f}] {keys:16} {cover:16} {kind}")
    return 0


# 判定器自检用例：(输入, 解码器给的拼音键, 期望分类)
SELFTEST_CASES = (
    ("li", "li", EXACT),
    ("xianzai", "xian-zai", EXACT),
    ("sihou", "shihou", FUZZY),          # s/sh 平翘舌
    ("zhuchen", "zucheng", FUZZY),       # zh/z + en/eng 叠加
    ("zhuchen", "zhucheng", FUZZY),      # 只差一个 g，别记成「补全」
    ("shencun", "shenchun", FUZZY),      # 第二个音节的声母模糊
    ("hanggui", "han-gui", FUZZY),       # ang/an 需要折叠到不动点
    ("shenkong", "shengong", FUZZY),     # k/g 要先折，再折 eng/en
    ("huanggou", "huang-kou", FUZZY),
    ("liuniu", "niuniu", FUZZY),         # n/l
    ("zhongguor", "zhongguoren", COMPLETION),
    ("chijiu", "chijiuceng", COMPLETION),
    ("zhongguoren", "he", MISMATCH),     # 真误配
    ("shijian", "shijin", MISMATCH),     # 少一个音节的残缺读音
)


# 带覆盖列的自检用例：(输入, 拼音键, 覆盖, 期望分类)
SELFTEST_COVER_CASES = (
    ("nihaoshijie", "nihao", "part:nihao", PARTIAL),
    ("sihou", "shi", "part:si", PARTIAL),          # 部分候选也可以是模糊读音
    ("nihaoshijie", "he", "part:nihao", MISMATCH),  # 部分候选的读音也要对得上
    ("nihaoshijie", "ni-hao-shijie", "full", EXACT),
)

SELFTEST_CORRECTION_CASES = (
    ("nihso", "nihao", "full", "correction", CORRECTION),
    ("nihoa", "nihao", "full", "correction", CORRECTION),
    ("niihao", "nihao", "full", "correction", CORRECTION),
    ("niihso", "nihao", "full", "correction", MISMATCH),
    ("nihxo", "nihao", "full", "correction", MISMATCH),
    ("nihao", "nihao", "full", "correction", MISMATCH),
    ("nihso", "nihao", "part:ni", "correction", MISMATCH),
    ("nihso", "nihao", "full", "", MISMATCH),
)


def cmd_corrections(args: argparse.Namespace) -> int:
    """固定真实词库用例：三类错误均须找回目标，且纠错标签必须通过独立验证。"""
    cases = (
        ("nihso", "你好"), ("nihoa", "你好"), ("niihao", "你好"),
        ("shijir", "世界"), ("shijei", "世界"), ("shijjie", "世界"),
        ("diannso", "电脑"), ("diannoa", "电脑"), ("diannnao", "电脑"),
    )
    decoded = decode([p for p, _ in cases], args.user)
    failed = 0
    for pinyin, target in cases:
        got = decoded.get(pinyin, [])
        texts = [c[0] for c in got]
        rank = texts.index(target) + 1 if target in texts else None
        invalid = any(classify(pinyin, keys, cover, source) == MISMATCH
                      for _, _, keys, cover, source in got)
        ok = rank is not None and rank <= 5 and not invalid
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {pinyin:12} {target} 排名={rank} 误配={invalid}")
    print(f"纠错样例：{len(cases) - failed}/{len(cases)} 通过（前 5 命中且无误配）")
    return 1 if failed else 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """判定器自检：分类逻辑是启发式的，先保证它自己不误报再看统计。"""
    bad = [(p, k, want, got) for p, k, want in SELFTEST_CASES
           if (got := classify(p, k)) != want]
    bad += [(p, f"{k} [{c}]", want, got) for p, k, c, want in SELFTEST_COVER_CASES
            if (got := classify(p, k, c)) != want]
    bad += [(p, f"{k} [{c}/{s}]", want, got) for p, k, c, s, want in SELFTEST_CORRECTION_CASES
            if (got := classify(p, k, c, s)) != want]
    for pinyin, keys, want, got in bad:
        print(f"  FAIL {pinyin} 键={keys}: 期望 {want}，得到 {got}")
    total = len(SELFTEST_CASES) + len(SELFTEST_COVER_CASES) + len(SELFTEST_CORRECTION_CASES)
    print(f"判定器自检：{total - len(bad)}/{total} 通过")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="输入法候选质量 fuzzy 测试")
    ap.add_argument("--user", help="用真实用户库（默认空用户库，不污染用户数据）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="随机取样测候选质量（回归基准）")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sample)

    m = sub.add_parser("mono", help="常用单音节专项（精确高频字是否进前 10）")
    m.set_defaults(func=cmd_mono)

    sub.add_parser("selftest", help="自检分类逻辑（不需要词库）").set_defaults(func=cmd_selftest)
    sub.add_parser("corrections", help="真实词库纠错召回与误配回归").set_defaults(func=cmd_corrections)

    w = sub.add_parser("words", help="看指定拼音的候选及其来源判定")
    w.add_argument("pinyins", nargs="+")
    w.set_defaults(func=cmd_words)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
