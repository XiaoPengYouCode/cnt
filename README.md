# cnt

一个用 Rust 写的简易简体中文拼音输入法（IBus 引擎）。

## 工作区结构

```
crates/
├── cnt-config        配置：TOML 加载（候选词数 + 语音输入）
├── cnt-ibus          IBus D-Bus 协议：对象序列化 + 私有总线地址发现
├── cnt-dict          词库：二进制(.cntd) + mmap 零拷贝读取 + 用户调频/用户词库
├── cnt-store         二进制 mmap 存储底座：错误类型/字节读取/区域校验（cnt-dict 与 cnt-lm 共享）
├── cnt-lm            n-gram 语言模型：二进制(.cntl) + mmap 二分查找（自研，无外部库）
├── cnt-score         打分端口层（零依赖）：NgramLm 打分接口 + Rescorer 重排契约 + 触发/融合策略
├── cnt-decode        拼音切分（标准 410 音节表）+ 整句解码（beam search + 词级格）
├── cnt-dict-tools    词库/LM CLI：build / build-lm / decode / info / query
├── cnt-input         输入逻辑（纯状态机，无 IO；候选接口 + 热键解析）
├── cnt-asr           语音端口层（零依赖）：Recognizer / Punctuator / TextScorer 契约 + 文本后处理
├── cnt-asr-lm        适配器：用 n-gram + 用户词库给 ASR 的 n-best 打分（热词偏置）
├── cnt-asr-onnx      识别 + 标点后端（ort/CPU）：自研 fbank/LFR/CMVN/CTC + CT-Transformer 标点
├── cnt-audio         麦克风采集（cpal）+ 带窗 sinc 重采样 + 自适应能量 VAD 切句
├── cnt-voice         语音编排：按住说话 / 切换式常开，每句一棵 span 树
├── cnt-asr-tools     语音 CLI：transcribe / bench（离线）、record / live（开麦克风）
├── cnt-engine         IBus Factory/Engine D-Bus 服务端
├── cnt-daemon        主程序：加载词库+LM、连接 IBus、注册组件、事件循环
└── cnt-test-client   开发用测试客户端（模拟应用程序走 IBus 链路）
```

所有依赖在根 `Cargo.toml` 的 `[workspace.dependencies]` 统一定义，
各 crate 通过 `workspace = true` 引用。

## 架构

```
应用程序 ──► ibus-daemon ──► cnt-daemon（本输入法引擎）
                 │                    │
                 │                    └─► 麦克风 ─► cnt-audio ─► cnt-asr-onnx（本地推理）
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
#   输入 a-z、space、enter、backspace、esc、1-9、left/right/up/down/pageup/pagedown，空行退出
```

环境变量：

- `CNT_DICT`：词库路径（默认 `$XDG_DATA_HOME/cnt/dict.cntd`）
- `CNT_LM`：语言模型路径（默认 `$XDG_DATA_HOME/cnt/lm.cntl`；缺失时降级为单字/词候选）
- `CNT_USER_DB`：用户数据路径（默认 `$XDG_DATA_HOME/cnt/user.dict`）
- `CNT_ASR_DIR`：声学模型目录（默认 `$XDG_DATA_HOME/cnt/asr`；缺失时语音自动关闭）
- `CNT_PUNCT_DIR`：标点模型目录（默认 `$XDG_DATA_HOME/cnt/punct`；缺失时输出无标点）

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
   用户调频是 **log-linear 混合**（libime 口径）：`max(lm, logsumexp(lm+wa, user+wb))`，
   用户分是概率尺度（`log10(count/(count+K))`）、逐步 lift 有界 —— 拼接路径的量级由
   LM 决定，单字计数顶多把该段往上抬一小截（不再有 从(+2.0)+是(+2.0) 线性叠加盖过 LM）
4. **候选**：Top-K 整句 + 整键词候选 + 尾音节补全合并，去重后发给 IBus

候选分四组（组间是硬顺序，组内按分数；这些是分数调参担保不了的，必须靠结构）：

| 组 | 内容 | 为什么在这个位置 |
|---|---|---|
| 0 | **整词覆盖输入**（整键词 / beam 单段） | 你已经打完的读音优先；整词先于一切拼接 |
| 1 | **拼接路径**（多段 beam 句子，仅当组 0 有整词时出现） | 对齐 librime `has_exact_match_phrase`：整词覆盖存在时不和整词裸拼，但仍可见可选（整词全错时能翻页救回）；无整词覆盖（长输入）时句子留在组 0 |
| 2 | **补全候选**（尾音节没打完，`zhongguor → 中国人`）+ 词库长尾字形 | 「猜你还没打完」不该插到已打完的读音前面（输入 `jian` 时前排不能被 `jiang` 的词占掉） |
| 3 | **部分候选**（只覆盖前一段，`nihaoshijie → 你好`） | 修复入口，不干扰正常整句选词 |

「整词覆盖」判据排除词库长尾（freq ≤ 100 且不在 LM，如 冲矢）：只有这种不可靠
整词时拼接仍留组 0，正确候选不被词频 1 的异体整词挤下去。重排（rescore）只在
**组内**进行，模型融合分不跨组改写硬顺序。

另有一条硬约束：**两处以上模糊不得占 #1** —— 单处模糊是模糊音的初衷
（`sihou → 时候` 仍 #1），但 `zhuchen → 组成`（zh/z + en/eng）这种同时错两个音的
不得抢榜首。

### 键位

| 键 | 作用 |
|---|---|
| `a`-`z` | 输入拼音 |
| `空格` | 选中光标处候选（部分候选 → 确认该段，继续组合） |
| `1`-`9`、`0` | 直接选该序号的候选（`0` = 页内第 10 个；超出本页的序号不响应） |
| `←` `→` / `↑` `↓` | **在候选间移动光标**（跨页自动翻页）—— 候选窗默认横排，左右与上下都映射到「上一个/下一个候选」 |
| `PageUp` `PageDown` / `-` `=` | 翻页（光标落到新页首） |
| `退格` | 删未确认的字母；未确认部分为空时撤销上一次部分确认 |
| `Esc` | 丢弃整个组合（含已确认段） |
| `回车` | 把未确认的拼音原文当英文提交 |
| `Shift` 单击 | 中/英模式切换 |
| 按住 `右 Alt` | 语音输入：按住说话，松手识别上屏（需开启 `[voice]`） |
| `Ctrl+Shift+空格` | 语音输入：切换式常开听写（VAD 自动切句） |
| 录音中 `Esc` | 放弃这次语音，一个字都不上屏 |

### 增量确认（Rime 式）

整句候选不对时不必删光重打：选中**部分候选**只确认那一段，剩余拼音继续组合。

```
输入 nihaoshijie   候选: 你好时节 / 你好世界 / … / 你好(part) / 你(part)
选「你好」          预编辑变成「你好 shi jie」，剩余 shijie 重新解码
选「世界」          一次上屏「你好世界」，学习段 [(nihao,你好), (shijie,世界)]
```

- 候选自带 `consumed`（消耗掉多少输入字节）—— 不能从拼音键推算，模糊音的键长度
  与输入不同（输入 `sihou` 走的键是 `shihou`）
- 预编辑 = **已确认的汉字 + 分节显示的未确认拼音**（`你好 shi jie`），
  用户能看见「切分成什么」和「确认到哪」，而不是一串连写字母
- **浏览候选时内联预览**：光标离开 #1 后，预编辑把选中候选放进来
  （整句 → `你好世界`；部分候选 → `你好 shi jie`，未覆盖的仍是拼音），
  空格确认前就能看到结果。光标在 #1 时不预览 —— 还没挑就保持拼音显示，
  无状态判定、移回去就恢复
- 退格：先删未确认的字母；未确认部分为空时，撤销上一次部分确认（那段拼音退回来）
- Esc 丢弃整个组合（含已确认段）；标点/Shift 切换/回车都会把已确认段一起上屏
- 学习在最终上屏时一次完成，所以「你好 + 世界」也会被拼成新词 `nihaoshijie`。
  **学习规则对齐 librime `UpdateElements`**：纯单字拼接提交（从+是）不给单字
  bump —— 那是短语级意图，不该刷高 从/是 这类高频字；路径含任一真多音节词时
  （我们+在+工作 含「工作」）全部段（含单字）都调频。单字调频仍来自直接选择。
  自动造出的新词（复合词）是**低可信度起点**（对齐 librime dee=0.1）：首现
  有效计数 ×0.1，只可见不竞争，第 2 次确认才转正满权重（tsv 第 5 列 `confirmed`）。

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

- **动态调频**：用户通过候选上屏即计数；拼音解码侧是 **log-linear 混合**（libime 口径，
  见上「打分」），语音 n-best 是粗粒度有界加法（同一把尺，`cnt-score::policy::user`）
- **时间衰减**：计数按 30 天半衰期衰减（jiff 时间戳），习惯漂移后旧词淡出
- **词库上限**：2 万条封顶，超出淘汰有效计数最低的词条（未转正词天然优先被淘汰）
- **低可信度起点**：自动造出的新词（不在静态词库）首现 `confirmed=false`，
  有效计数 ×0.1（对齐 librime dee=0.1），只可见不竞争；第 2 次确认转正满权重。
  词库已有的词调频不受影响。tsv 第 5 列持久化（旧 4 列数据视为已转正，兼容）
- **新词学习**：提交的多段句子按相邻段拼成复合词（郑+爽 → zhengshuang/郑爽），
  下次输入完整拼音直接出整词；学过的词也能用在**句子中间**——多音节链剪枝
  （`has_key_prefix`）与用户库一起判断前缀存在性，且不在 LM 词表的词用「句首基础分
  + backoff」当伪 unigram（而非一律 UNK -12），否则学过的词在句中必然输给逐字拼接

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

TOML 配置文件：`~/.config/cnt/config.toml`（`CNT_CONFIG` 环境变量可覆盖路径）：

```toml
# 每页候选词数（范围 5~10，默认 8）
# 上限 10 是选择键决定的：数字键只有 `1`-`9` 加 `0`（第 10 个）
page_size = 8

# 语音输入（默认关：模型不随程序分发，得先装）
[voice]
enabled = true
model_dir = "~/.local/share/cnt/asr"
language = "auto"                    # auto/zh/en/ja/ko/yue
itn = true                           # 数字规整 + 标点
threads = 4
device = ""                          # 麦克风设备名子串，空 = 系统默认
ptt_key = "Alt_R"                    # 按住说话（右 Alt）
toggle_key = "Control+Shift+space"   # 切换式常开
max_seconds = 60.0                   # PTT 单次最长录音
trailing_silence_ms = 700            # 常开：停顿多久算一句结束
vad_margin_db = 10.0                 # 高于噪声底多少 dB 算语音
```

逐项解析而不是整体反序列化：**单项写错不影响其他项**（输入法不能因为一个写错的
热键就不能打字），越界值夹紧到合法区间。

### 模糊音（cnt-decode）

`FUZZY_RULES` 双向规则：平翘舌（z/zh、c/ch、s/sh）、边鼻音（n/l、f/h、r/l）、
前后鼻音（an/ang、en/eng、in/ing、ian/iang、uan/uang）等。
用户输入变体（如 `si` 想表达 `shi`）在切分层映射回标准音节后查词，
精确读音优先，不被模糊遮蔽（`sihou → 时候`、`shifou → 是否` 同时正确）。

标准词表格式：`拼音<TAB>词<TAB>频率`，`#` 开头为注释。

## 语音输入

**纯本地**：模型在自己机器上跑，音频只在内存里流动，不上传、不落盘
（除非你显式用 `cnt-asr-tools record`）。麦克风只在会话期间打开，
会话一结束（松手 / 关常开 / 窗口失焦 / 按 Esc）立刻关设备。

### 领域模型与端口

四件事变化速率完全不同，所以拆成四个 crate；变化最快的两件（识别模型、标点模型）
被固定成**端口**，其余代码只依赖契约：

```text
  cnt-audio            cnt-asr（端口层，零依赖）           cnt-voice        cnt-engine
 ┌──────────┐     ┌──────────────────────────────┐    ┌────────────┐   ┌──────────┐
 │ cpal 采集 │────►│ trait Recognizer             │◄───│  会话编排   │──►│ 预编辑   │
 │ sinc 重采样│     │ trait Punctuator             │    │  PTT/常开   │   │ CommitText│
 │ 能量 VAD  │     │ text: 字节级BPE拼装/标点归一  │    │  span 树    │   │ 热键     │
 └──────────┘     └──────────────┬───────────────┘    └────────────┘   └──────────┘
                                 │ 实现
                        cnt-asr-onnx（ort / CPU，将来 NPU）
                        ├─ SenseVoice   : fbank → LFR → CMVN → CTC 贪心
                        └─ CtPunctuator : 词 id → 6 类标点
```

| 端口 | 粒度 | 频次 | 分派 | 当前实现 | 缺省退化 |
|---|---|---|---|---|---|
| `Recognizer` | 一整段语音 0.3~60 s | 每句 1 次 | `dyn` | `SenseVoice`（Fun-ASR-Nano CTC） | 无（语音功能关闭） |
| `TextScorer` | n-best 里每条文本 | 每句 ≤8 次 | `dyn` | `LmTextScorer`（`cnt-lm` + 用户词库） | None（保持声学顺序） |
| `Punctuator` | 一句文本几十字 | 每句 ≤1 次 | `dyn` | `CtPunctuator`（CT-Transformer） | `NoPunct` 直通 |

### n-best 重排：让语言模型纠「音对字错」

CTC 贪心只给 1-best，而语音识别的主要错误恰恰是**音对字错**——正确答案通常
就躺在第 2、3 名里。所以改用 **CTC prefix beam search** 出 n-best，再用
`cnt-lm`（27 万 unigram + 480 万 bigram）重排：

```text
声学 n-best                              融合分 = 声学(log10) + 0.5 × LM(log10)
  #0 -0.47  开饭时间早上九点至下午五点     ← 声学 #1
  #1 -1.71  开放时间早上九点至下午五点     ← LM 重排后胜出（正确）
  #2 -4.15  开饭时间朝上九点至下午五点
```

三条保守约束（与拼音侧重排同源）：候选 <2 条不重排；#1 领先 #2 超过 1.5 log10
不重排（声学已确定，插手只会把「说得不常见但确实说了」改成「常见但不是我说的」）；
融合而非替代。

**热词偏置**：`user.dict` 里的词按 `min(选择次数, 10) × 0.2` 加分——与拼音候选
排序用的是同一把尺（`cnt_score::policy::user`）。通用声学模型不可能知道你把
「工站」当常用词，而这份数据正是你自己一次次选出来的，是本地方案独有的信息。

**一道必须有的闸**：未登录字占比 >40% 时 `TextScorer` 返回 `None` 拒绝打分。
中文 n-gram 给日语句子打分时假名全是未登录字，分数只反映「文本有多长」，
重排会系统性选最短的那条把句尾吃掉（实测 `…パンを買う` → `…パンを買`）。
这与拼音侧「不同覆盖长度的假设不可比」是同一类错误：**不可比的东西不要比**。

**为什么标点是独立端口**（而不是识别器的内部细节）：实测三个模型三种情况——
`Fun-ASR-Nano` 的 CTC 头词表**有**标点 token 但输出光板文本；`SenseVoice-Small`
自带标点；`FireRedASR2` 全家词表根本没有标点。也就是说「会不会标点」是声学模型的
**偶然属性**，而「上屏文本必须有句读」是输入法的**固有需求**。拆开之后，换声学模型
不用重做标点，换标点模型不碰声学。这也是 `FunASR` 官方 pipeline 的做法。

端口契约里有两条硬规则，实现必须守：

- **标点只加符号、不改字**——用户看到「我说的不是这个」比没标点更糟；
- **任何一段失败都退回上一级结果**（标点失败用原文、识别失败不上屏），
  一句话绝不能因为增强环节出错而丢掉。

### 交互方案

| 模式 | 触发 | 切句 | 适用 |
|---|---|---|---|
| **按住说话**（默认 `Alt_R`，即右 Alt） | 按住录音，松手识别 | 由你的手决定 | 短句，确定性最高 |
| **切换式常开**（默认 `Control+Shift+space`） | 按一下开/关 | VAD 自动切句，逐句上屏 | 长段口述 |

```text
       ┌─────────────── Esc（放弃，一个字都不上屏）───────────────┐
       ▼                                                          │
    [空闲] ──按住 ptt──► [录音中 🎤] ──松手──► [识别中] ──► 上屏 ──┘
       │                     ▲   │
       └──toggle──► [常开 🎤]─┘   └──VAD 切出一句──► 识别 ──► 上屏（会话继续）
```

设计要点，每条都对应一个具体的失败场景：

- **PTT 默认用右 Alt 而不是字母键**：按住期间该键对应用必须无副作用，
  字母键会一直往应用里灌字符。该事件**照常转发**给应用，Alt 行为不变。
  （不用右 Ctrl 是因为很多新键盘把它换成了 Copilot 键；`ptt_key` 可配成
  `Alt_R`/`Control_R`/`Menu`/`F13`~`F24` 等任意 keysym。）
- **Esc 取消**：说错了、被人打断、误触——必须有作废出口，否则唯一选择是
  让错的内容上屏再删。
- **开始语音前先把拼音预编辑上屏**：打了半句拼音又想说话时，半截拼音既不能丢，
  也不能和语音结果混在一起。
- **窗口失焦 / 输入法被禁用 → 立刻停录**：麦克风不跟着焦点漂，
  这既是隐私底线，也避免结果上屏到错误的应用。
- **按键路径上不做任何等待**：识别的几百毫秒全在 `cnt-voice` 的 OS 线程里，
  引擎只 spawn 一个事件泵。IBus 的按键是串行的，卡住就会丢键。
- **PTT 有最长录音上限**（默认 60 s）：按键卡住不会变成无限录音。
- 常开模式下键盘照常可用；语音结果按句上屏，不干扰正在打的拼音。

### IME 显示方案

非流式模型**没有中间结果**，所以界面必须用别的东西证明「它在听」，
否则用户会反复松手重按（以为没生效）。预编辑区的规格：

| 状态 | 预编辑显示 | 候选窗 |
|---|---|---|
| 录音中（PTT） | `🎤 说话中 2.3s ▂▃▄▅▁` | 隐藏 |
| 录音中（常开） | `🎤 常开听写 12.7s ▂▃▁▁▁` | 隐藏 |
| 识别中 | `🎤 识别中…` | 隐藏 |
| 取消 | `🎤 已取消`（随即清除） | 隐藏 |
| 组合中同时录音 | `🎤 说话中 1.2s ▂▃▄▁▁ 你好 shi jie` | 正常拼音候选 |

- **时长 + 音量条每 200 ms 刷一次**（`VoiceEvent::Level`）：音量条全空
  等于告诉你「没听到声音」——麦克风选错了、被静音了，一眼就能发现，
  不用等到识别出空结果才怀疑。
- **语音结果直接 `CommitText`，不进候选窗**：一整句话放进候选列表没有意义
  （候选是「同音异形的选择」，语音结果只有一条）。要改就改已上屏的文本。
- 语音提示与拼音预编辑**共存**：提示在前、拼音在后，中间一个空格。
- 出错（麦克风被占用、模型报错）只记日志并清掉提示，**不弹窗、不阻塞打字**。

### 模型

```bash
bash scripts/fetch-asr-model.sh   # 声学 188 MB → data/asr，标点 65 MB → data/punct
./scripts/install.sh              # 一起装到 ~/.local/share/cnt/{asr,punct}
```

| | 模型 | 体积 | 结构 |
|---|---|---|---|
| 声学 | `Fun-ASR-Nano-2512` 的 CTC 导出（通义，Apache-2.0） | 251 MB | 单次前向 CTC |
| 标点 | `CT-Transformer ct-punc` zh-en（`FunASR`） | 73 MB | 文本级 transformer |

选型是按条件筛的（本地 CPU RTF ≤ 0.3 / 常驻 ≤ 500 MB / 必须有标点 /
ONNX 单次前向），Fun-ASR-Nano 在**工业**测试集上是开源第一：平均 WER 16.72，
优于 FireRedASR2 (22.63)、Paraformer v2 (23.49)、GLM-ASR-Nano (26.13)、
Whisper-large-v3 (33.39)，逼近闭源 Seed-ASR (15.95)；方言 28.18 对 FireRedASR2 的 52.82。
落选原因：`FireRedASR2` 全家词表无标点无数字（实测确认）；`Qwen3-ASR-0.6B` /
`GLM-ASR-Nano` / `Fun-ASR-Nano` 全量版都是自回归且 >800 MB。

前端（fbank / LFR / CMVN / CTC / base64 字节级 BPE 拼装）**全部自研**，不引 C++ 特征库；
模型契约不靠假设，用 `cnt-asr-tools info` 探查后按声明动态适配：

```
$ cnt-asr-tools info data/asr
输入： x  Float32 [1,-1,560]        ← 只有一个（SenseVoice-Small 有四个）
输出： logits Float32 [-1,-1,60515]
metadata: blank_id=60514  lfr_window_size=7  lfr_window_shift=6
          normalize_samples=0  model_type=sense_voice_ctc
词表：60515，base64 字节级 BPE，blank=60514
```

浮点输入 = 特征，其余整型输入按名字识别（`len`/`lang`/`itn`），blank 从 `<blk>` 取，
词表编码自动识别明文/base64。**换模型第一步是打印事实，不是猜**——
「输入名对上了但语义不对」会得到「能跑但输出乱码」，那是最费时间的失败。

### 实测（Intel Core Ultra 5 336H，4 线程，release）

```
bench（zh.wav 5.59s，8 轮 × 3 次交替）：平均 88~92ms（RTF 0.016）

fastrace 阶段分解（span 属性直接可读）：
  asr_transcribe   84933µs  [samples=89472 frames=93 tokens=13 nbest=8 vocab=60515]
    asr_frontend     650µs（0.8%）   ← fbank 638 / lfr 12 / cmvn 0.03
    asr_infer      84265µs（99%）    ← encoder
      ctc_beam      4959µs（6%）     ← prefix beam search
  punctuate         ~3000µs
```

3 秒的短句约 50 ms，「松手到上屏 < 300 ms」的预算很宽裕。

> **n-best 是免费的**：CTC prefix beam search 起初比贪心慢 22%（+20ms），
> 原因是逐帧做了全量 softmax 归一化（6 万类 × 93 帧 ≈ 560 万次 `exp()`）。
> 但归一化常数对所有假设**完全相同**，而我们只用到假设之间的排序与分差——
> 常数会消掉。去掉之后 `ctc_beam` 20.1ms → 4.9ms，与贪心持平
> （交替测量：贪心 88/93/93ms，beam+重排 92/90/88ms）。
>
> 早先 README 记录的「232ms」是**降频状态**下测的，不是代码变慢——
> 这正是 AGENTS.md 要求「交替跑新旧二进制」的原因。

### CLI（开发/诊断）

原则：**能不开麦克风就不开麦克风**。

```bash
# 离线（wav 进、文本出，不碰麦克风）
cargo run --release -p cnt-asr-tools -- info       data/asr
cargo run --release -p cnt-asr-tools -- transcribe data/asr a.wav [--punct data/punct] [--no-punct]
cargo run --release -p cnt-asr-tools -- bench      data/asr a.wav 20

# 需要麦克风（会在 stderr 明确提示）
cargo run --release -p cnt-asr-tools -- record     out.wav 3
cargo run --release -p cnt-asr-tools -- live       data/asr 5 [--continuous]
```

`bench` 输出延迟分位数 + RTF + fastrace 阶段聚合表，口径与 `cnt-dict-tools bench` 一致。
`live` 走的是和引擎完全相同的链路（含标点），可以当交互预演。

### 可观测性

每句话一棵 root span `voice_utterance`，子 span 覆盖前端/推理/解码/标点，
属性带音频时长、耗时与 **RTF**（推理耗时 / 音频时长）——这是语音链路唯一有意义的
性能指标：<1 才可能跟得上说话，PTT 体验上要求 <0.3。重采样与 VAD 也各自埋了 span
（无上层 context 时零开销，采集回调里也能安全埋）。


## 安装（作为第三个输入法，与英文 / Rime 互不冲突）

构建依赖（一次性）：麦克风采集走 ALSA（cpal → alsa-sys → libasound），
需要 **dev** 包里的头文件与 `alsa.pc`（系统默认只装运行时库，编译会失败）：

```bash
sudo apt install libasound2-dev      # Debian/Ubuntu
# Fedora: sudo dnf install alsa-lib-devel   /   Arch: sudo pacman -S alsa-lib
```

PipeWire 自带 ALSA 兼容层，所以走 ALSA 在 PipeWire 系统上照样录音，不需额外后端。
`scripts/install.sh` 会先检查这个依赖，缺就自动装。

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
  `词键 → 候选词表`（含 LM 下标、句首基础分、用户计数）
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
按键状态机（选词/翻页/退格/转交）、热键解析；
语音侧：重采样（直流增益/分块一致性）、VAD 切句（停顿不断句/短噪声丢弃/超长强切）、
fbank（帧数口径/窗/mel 滤波器）、LFR 边界（左右 padding）、CMVN、CTC 贪心去重、
token 拼装与中英混排标点归一。

语音部分的单测**不需要模型也不需要麦克风**（端口层用假识别器，前端用合成信号），
`cargo test` 在 CI 里能直接跑。
