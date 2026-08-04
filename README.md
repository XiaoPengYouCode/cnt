# cnt

一个用 Rust 写的简易简体中文拼音输入法（IBus 引擎）。

## 工作区结构

```
crates/
├── cnt-config        配置：TOML 加载（当前仅候选词数一项）
├── cnt-ibus          IBus D-Bus 协议：对象序列化 + 私有总线地址发现
├── cnt-dict          词库：二进制(.cntd) + mmap 零拷贝读取 + 用户调频/用户词库
├── cnt-store         二进制 mmap 存储底座：错误类型/字节读取/区域校验（cnt-dict 与 cnt-lm 共享）
├── cnt-lm            n-gram 语言模型：二进制(.cntl) + mmap 二分查找（自研，无外部库）
├── cnt-score         打分端口层（零依赖）：NgramLm 打分接口 + Rescorer 重排契约 + 触发/融合策略
├── cnt-decode        拼音切分（标准 410 音节表）+ 整句解码（beam search + 词级格）
├── cnt-dict-tools    词库/LM CLI：build / build-lm / decode / info / query
├── cnt-input         输入逻辑（纯状态机，无 IO；定义候选接口）
├── cnt-engine         IBus Factory/Engine D-Bus 服务端
├── cnt-daemon        主程序：加载词库+LM、连接 IBus、注册组件、事件循环
└── cnt-test-client   开发用测试客户端（模拟应用程序走 IBus 链路）
```

所有依赖在根 `Cargo.toml` 的 `[workspace.dependencies]` 统一定义，
各 crate 通过 `workspace = true` 引用。

## 架构

```
应用程序 ──► ibus-daemon ──► cnt-daemon（本输入法引擎）
                 │
                 └──► ibus-panel（画候选窗，不需要我们写 GUI）
```

引擎只需通过 D-Bus 把数据发出去：`UpdatePreeditText`（拼音预编辑）、
`UpdateLookupTable`（候选）、`CommitText`（上屏）。

词库采用两层结构：

1. **静态基础词库**：自定义二进制格式（`.cntd`，定长小端整数 + 字节偏移），
   运行时 mmap 零拷贝 + 二分查找（精确/前缀）。不用 bincode —— bincode 面向
   「反序列化成拥有型结构」，而 mmap 需要「驻留内存、原地切片」。
2. **用户个性化模型**（`~/.local/share/cnt/user.dict`，明文 tsv，原子写盘）：
   用户每次通过候选上屏都会记录 `(拼音, 词)` 计数，查询时与基础词频混合打分：

   ```
   score(词 | 拼音) = base_freq(词) + USER_BOOST × user_count(拼音, 词)
   ```

   即「越用越顺手」：多选几次的词会自动往前排。

## 使用

```bash
# 1. 准备数据（一次性）：从 fcitx5 官方数据服务器下载
#    https://download.fcitx-im.org/data/dict-*.tar.zst        → dict_sc.txt（30 万词条）
#    https://download.fcitx-im.org/data/lm_sc.arpa-*.tar.zst  → lm_sc.arpa（语言模型）

# 2. 转换 + 编译词库（.cntd，mmap 零拷贝读取）
cargo run -p cnt-dict-tools -- import-libime dict_sc.txt data/wordlist.tsv --lm lm_sc.arpa
cargo run -p cnt-dict-tools -- build data/wordlist.tsv data/cnt.dict

# 3. 编译语言模型（.cntl：270k unigram + 480 万 bigram）
cargo run -p cnt-dict-tools -- build-lm lm_sc.arpa data/lm.cntl

# 4. 整句解码验证（开发调试用）
cargo run -p cnt-dict-tools -- decode data/cnt.dict data/lm.cntl xianzai womenzaigongzuo

# 5. 启动引擎（CNT_DICT / CNT_LM 指定数据文件；默认 ~/.local/share/cnt/）
CNT_DICT=data/cnt.dict CNT_LM=data/lm.cntl cargo run -p cnt-daemon

# 6. 另一终端：测试客户端模拟输入
cargo run -p cnt-test-client
#   输入 a-z、space、enter、backspace、esc、1-9、up/down/pageup/pagedown，空行退出
```

环境变量：

- `CNT_DICT`：词库路径（默认 `$XDG_DATA_HOME/cnt/dict.cntd`）
- `CNT_LM`：语言模型路径（默认 `$XDG_DATA_HOME/cnt/lm.cntl`；缺失时降级为单字/词候选）
- `CNT_USER_DB`：用户数据路径（默认 `$XDG_DATA_HOME/cnt/user.dict`）

用户学习数据每 60s 自动写盘（临时文件 + rename 原子替换）。

## 词库数据说明

词库（`data/` 下的 tsv 与二进制）**不入库**，通过数据管线独立分发：

1. **来源**：fcitx5 的 libime 数据（`dict_sc.txt`：简体拼音词典；`lm_sc.arpa`：语言模型）
2. **转换**（`cnt-dict-tools import-libime --lm`）：
   - 音节分隔符 `'` 去掉（`ni'hao` → `nihao`）
   - 候选排序用语言模型 1-gram 对数概率（`是 -1.94` > `时 -2.88` > `螫 -6.17`），
     dict.txt 自带的权重不可靠（多音字默认读音会被给到正权重）
   - LM 地板词（如「你好」，LM 按 你+好 二元组建模）若 dict 权重为 0，
     抬到常用词带，避免排到生僻字后面
3. **编译**：`build` 生成 `.cntd`（定长小端整数 + 字节偏移，mmap 零拷贝二分查找）

### 整句解码（cnt-decode）

自研（无开源库）：

1. **音节切分**：标准 410 无调音节表 + 全切分 DP，生成切分格（lattice）
2. **词级解码**：格上扩展单音节词与多音节词（如 `gong-zuo` → 工作），
   beam search（beam=8）+ 可达性剪枝（丢弃走不到末尾的错误切分路径）
3. **打分**：`logP(w1) + Σ logP(wi|wi-1)`，bigram 缺失时 Katz backoff；
   用户调频作为 log10 加成参与句子评分
4. **候选**：Top-K 整句 + 整键词候选 + 尾音节补全合并，去重后发给 IBus

候选排序有两条**结构性硬约束**（分数调参担保不了，必须靠结构）：

- **补全词不得插到精确候选之前**：补全是「猜你还没打完」，输入 `jian` 时
  `jiang` 的高频词（将/奖/讲）不该占掉 见/件/间 的位置；
- **两处以上模糊不得占 #1**：单处模糊是模糊音的初衷（`sihou → 时候` 仍 #1），
  但 `zhuchen → 组成`（zh/z + en/eng）这种同时错两个音的不得抢榜首。

候选规模按**输入音节数**自适应（最短跳数 DP 算出）——单音节的搜索空间只有一个
位置却有几十个同音字要展示，长输入反之（展开是乘性的）：

| 输入 | 每键候选词 | 整句候选 | 模糊额外惩罚 |
|---|---|---|---|
| 单音节 | 20 | 15 | -3.0（无上下文佐证，`li → 你` 纯噪声） |
| 双音节 | 12 | 10 | 0 |
| 更长 | 8 | 5 | 0 |

不在 LM 的词（OOV）按词库词频做 log10 修正，而非一律 `-5.0`——否则词频 1 的
生僻字与词频 2 万的常用字同分，单音节的 5~10 位会被生僻字占满。

### 搜索与打分解耦（cnt-score）

行业实践里神经模型并没有取代动态规划：搜索（词图 + beam/Viterbi）仍是骨架，
模型只接管**打分**（典型做法是对 n-gram 选出的 top-N 做 lattice rescoring）。
cnt 把这个边界固定成端口（dependency inversion，方向永远是「实现 → 端口 ← 解码器」）：

```text
  cnt-lm  (n-gram, mmap)   ─┐
  cnt-nnlm (小模型，待建)   ─┼─► cnt-score ◄── cnt-decode（只认契约，不认模型）
```

| 端口 | 谁调 | 频次 | 分派 | 当前实现 |
|---|---|---|---|---|
| `NgramLm` | beam 内逐步打分 | 每按键上千次 | 静态（`Decoder<L: NgramLm = CntLm>` 单态化） | `cnt-lm::CntLm` |
| `Rescorer` | 整句重排 | 每按键 ≤1 次、仅 top-N | 动态（`dyn`，可按配置装卸） | 无（默认不重排） |

重排的触发与融合由 `RescorePolicy` 描述，默认保守——只在「基线自己也没把握」时花钱：

- 候选 <2 条 / 只有单段（无上下文）→ 不重排；
- `#1` 与 `#2` 分差 ≥ `gap`（1.5 log10）→ 基线已确定，不重排；
- 参与重排的只有前 `top_n`（默认 10）条，窗口外顺序不动；
- 融合而不替代：`final = base + λ × model`（默认 λ=0.5）——用户调频、用户词、
  模糊音惩罚这些硬约束都编码在 `base` 里，不能交给模型丢掉；
- 模型未就绪/超时可返回 `None`，调用方保持基线顺序——重排永远不能让输入法不可用。

埋点：`rescore` span + `rescored` 事件（候选数、模型名），直接在 bench 的阶段
聚合表里与 `lattice`/`beam`/`completions` 并列对比，决定是否值得开启。
未装重排器时这一段零开销（bench 实测：重构前 248.7µs / 后 249.3µs 平均，中位 206µs 持平）。

待建的 `cnt-nnlm`（小神经模型推理，实现 `Rescorer`）只需依赖 `cnt-score`，
不碰解码器与词库；目标规格：字级 1M~10M 参数、int8 量化、单次重排 <1ms。

### 用户模型（cnt-dict）

- **动态调频**：用户通过候选上屏即计数，混合打分排序
- **时间衰减**：计数按 30 天半衰期衰减（jiff 时间戳），习惯漂移后旧词淡出
- **词库上限**：2 万条封顶，超出淘汰有效计数最低的词条
- **新词学习**：提交的多段句子按相邻段拼成复合词（郑+爽 → zhengshuang/郑爽），
  下次输入完整拼音直接出整词

### 中文标点（cnt-input）

拼音模式下标点键直接上屏中文标点（简体，参考 `Rime` `luna_pinyin` 映射）：
`，。／；：？！「」『』（）《》～＠＃＄％＾＆＊－＿＋＝｜·`，
`"` 与 `'` 为智能引号（`“”` / `‘’` 交替开合）。
组合输入中按标点：先提交候选再附带标点（`ni` + `,` → `你，`）。

**半角例外**：只有「紧跟 ASCII 字母/数字的第一个标点」用半角（`3,` / `abc.` / `x="`），
其余一律全角。判定只看紧邻的前一个字符，所以 `3,` 后再打标点回到全角，
空格也不算 latin 上下文。上下文按优先级取：

1. 我们自己刚上屏 / 刚转发的字符（最新，不依赖应用配合）；
2. 应用下发的 surrounding text（`cursor_pos` 按**字符**计）；
3. 两者都无 → 全角（中文输入法的安全默认）。

焦点切换/重置会丢弃上下文，并发 `RequireSurroundingText` 请应用下发光标周围文本。

### 中/英切换（Shift）

- **Shift 单击**（350ms 内按下释放、期间未打其他键）→ 切换中/英模式；
  组合中切换会先把预编辑上屏
- **英模式**：所有按键透传（应用原生输入）
- **Shift+字母**：中模式下未组合时透传（输出大写）；组合中先提交预编辑再透传
- 模式按输入上下文隔离，切换窗口不影响其他窗口，`focus_out` 不清除

### 配置（cnt-config）

TOML 配置文件：`~/.config/cnt/config.toml`（`CNT_CONFIG` 环境变量可覆盖路径），
当前仅一项：

```toml
# 每页候选词数（范围 5~30，默认 10）
page_size = 10
```

### 模糊音（cnt-decode）

`FUZZY_RULES` 双向规则：平翘舌（z/zh、c/ch、s/sh）、边鼻音（n/l、f/h、r/l）、
前后鼻音（an/ang、en/eng、in/ing、ian/iang、uan/uang）等。
用户输入变体（如 `si` 想表达 `shi`）在切分层映射回标准音节后查词，
精确读音优先，不被模糊遮蔽（`sihou → 时候`、`shifou → 是否` 同时正确）。

标准词表格式：`拼音<TAB>词<TAB>频率`，`#` 开头为注释。

## 安装（作为第三个输入法，与英文 / Rime 互不冲突）

```bash
# 1. 先按上文生成 data/cnt.dict 和 data/lm.cntl
# 2. 运行安装脚本（最后一步 ibus restart 会重启会话输入法，属正常现象）
./scripts/install.sh
```

安装布局：

| 内容 | 位置 | 权限 |
|---|---|---|
| 二进制 | `~/.local/bin/cnt-daemon` | 用户级 |
| 词库/语言模型 | `~/.local/share/cnt/{dict.cntd,lm.cntl}` | 用户级 |
| 配置 | `~/.config/cnt/config.toml` | 用户级 |
| ibus 组件 | `/usr/share/ibus/component/cnt.xml` | sudo（唯一特权操作） |

> 为什么组件 XML 必须进系统目录：ibus 1.5.x 源码（`ibusregistry.c`）里
> 用户目录组件扫描是 `#if 0` 注释掉的（"FIXME ... user dir"），只扫描系统目录。

**互不冲突的保证**：英文输入是 xkb 键盘布局（不在 IBus 引擎体系内）；
Rime 组件名 `im.rime.Rime`、引擎名 `rime`；cnt 组件名
`org.freedesktop.IBus.Cnt`、引擎名 `cnt` —— 全部唯一。cnt 只在被显式选中时
由 ibus 按 XML 的 `exec` 启动，不自启、不抢全局引擎。

卸载：

```bash
sudo rm /usr/share/ibus/component/cnt.xml
rm -rf ~/.local/bin/cnt-daemon ~/.local/share/cnt
ibus restart
```

## 性能

每按键的解码延迟（`cnt-dict-tools bench data/cnt.dict data/lm.cntl 200`，18 个样例）：

| | 平均 | 中位 | 90% |
|---|---|---|---|
| 热（词键缓存命中，= 连续打字的第 2 键起） | ~74µs | ~68µs | ~111µs |
| 冷（首次进入该输入上下文） | ~150µs | ~140µs | — |

> 测量环境：Intel Core Ultra 5 336H / governor `powersave` / load ≈ 2 /
> `cargo build --release`（trace 关闭；开启 trace 约 +4%）。
>
> **绝对值只在同一台机器同一状态下有意义**：同一份二进制在降频状态下实测会慢
> 2~3 倍（74µs → 200µs+）。改动前后要比性能，用 `git worktree` 建旧版本
> **交替**跑两个二进制看相对差值，别跨时间点比绝对值（见 AGENTS.md）。
>
> 本轮优化即是这样测的：旧版与新版交替各跑 3 次，旧版 714~760µs、
> 新版（热）~200µs —— 同一降频状态下的 **3.5x**。

关键设计（都是先埋 span 看数据再改，见 AGENTS.md 的可观测性约定）：

- **词键缓存跨按键**：打字是增量的，敲 `womenzaigongzuo` 的每个字母都会把整个
  前缀重新解码，同一批词键（wo/women/zai/gongzuo…）反复查词库。缓存
  `词键 → 候选词表`（含 LM 下标、句首基础分、用户调频加成）
- **`learn()` 精确失效**：只丢这次动过的键，不整体清空 —— 否则每次上屏后的下一句
  都退回冷路径（实测每键 +26%）。真实打字的增量特性也在这里体现：追加一个字母只
  新增「以新末尾结束」的少数词键，前面位置的键全部命中，所以连续打字几乎全是热的
- **每个位置的词键只枚举一次**：词键只取决于「位置 + 音节格」，与假设无关；
  此前每条假设都重做字符串拼接与词库前缀查询，`BEAM=8` 就是 8 倍冗余
- **假设是 `Copy` 的小结构，路径存在 arena**：此前每条假设持有 `Vec<(Arc, Arc)>`，
  每次展开都要克隆整个 Vec（一次堆分配 + 全部引用计数），而每次解码有约 500 次展开
- **bigram 按「行」查**：同一前词的后继在 bigram 表里是连续区间。存活假设
  （≤ BEAM 条）先定位一次行，其所有后继在行内二分——把「几十次跨 60MB mmap 的
  随机二分」压成「一次定位 + 行内小范围二分」。注意行定位本身是两次全表二分，
  放到「每次展开」里会反而更慢（实测 +50%）
- **排序键先算好再排**：`ranked_words` 的比较器里曾调 `user_count`，每次比较都要
  抢用户库的锁 + 两次哈希查找，20 个候选就是上百次加锁
- **词库词零拷贝**：`ranked_words` 返回 `Cow`，词库词直接借 mmap，少一次 `String` 中转

## 候选质量回归测试

```bash
cargo build --release -p cnt-dict-tools
S=.agents/skills/fuzzy-test/fuzzy_test.py
python3 $S selftest   # 判定器自检
python3 $S sample     # 100 词随机取样（同 --seed 可复现），有读音误配则退出码 1
python3 $S mono       # 单音节专项：词库高频前 10 字是否进候选前 10
python3 $S words li sihou zhuchen   # 单个拼音的候选、分数、拼音键与来源判定
```

判定依据来自解码器自己：`cnt-dict-tools decode --tsv` 会输出每条候选**实际走的
拼音键**，脚本据此区分「精确读音 / 模糊回退 / 尾音节补全 / 读音误配」，
不必在测试里复制模糊音规则表。详见 `.agents/skills/fuzzy-test/SKILL.md`。

## 测试

```bash
cargo test
```

覆盖：二进制格式读写、mmap 精确/前缀查询、打分排序、用户调频持久化、
按键状态机（选词/翻页/退格/转交）。
