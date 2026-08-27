# 语音输入

**纯本地**：模型在自己机器上跑，音频只在内存里流动，不上传、不落盘
（除非你显式用 `cnt-asr-tools record`）。麦克风只在会话期间打开，
会话一结束（松手 / 关常开 / 窗口失焦 / 按 Esc）立刻关设备。

## 领域模型与端口

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
                        ├─ Fun-ASR-Nano : fbank → LFR → CMVN → CTC prefix beam
                        └─ CtPunctuator : 词 id → 6 类标点
```

| 端口 | 粒度 | 频次 | 分派 | 当前实现 | 缺省退化 |
|---|---|---|---|---|---|
| `Recognizer` | 一整段语音 0.3~60 s | 每句 1 次 | `dyn` | `Fun-ASR-Nano`（通义，CTC 导出） | 无（语音功能关闭） |
| `TextScorer` | n-best 里每条文本 | 每句 ≤8 次 | `dyn` | `LmTextScorer`（`cnt-lm` + 用户词库） | None（保持声学顺序） |
| `Punctuator` | 一句文本几十字 | 每句 ≤1 次 | `dyn` | `CtPunctuator`（CT-Transformer） | `NoPunct` 直通 |

## n-best 重排：让语言模型纠「音对字错」

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

## 为什么标点是独立端口

实测三个模型三种情况——`Fun-ASR-Nano` 的 CTC 头词表**有**标点 token 但输出光板文本；
`SenseVoice-Small` 自带标点；`FireRedASR2` 全家词表根本没有标点。也就是说
「会不会标点」是声学模型的**偶然属性**，而「上屏文本必须有句读」是输入法的
**固有需求**。拆开之后，换声学模型不用重做标点，换标点模型不碰声学。
这也是 `FunASR` 官方 pipeline 的做法。

端口契约里有两条硬规则，实现必须守：

- **标点只加符号、不改字**——用户看到「我说的不是这个」比没标点更糟；
- **任何一段失败都退回上一级结果**（标点失败用原文、识别失败不上屏），
  一句话绝不能因为增强环节出错而丢掉。

## 交互方案

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

## IME 显示方案

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

## 模型

```bash
bash scripts/fetch-asr-model.sh   # 声学 188 MB → data/asr，标点 65 MB → data/punct
./scripts/install.sh              # 一起装到 ~/.local/share/cnt/{asr,punct}
```

| | 模型 | 体积 | 结构 |
|---|---|---|---|
| 声学 | `Fun-ASR-Nano-2512` 的 CTC 导出（通义，Apache-2.0，int8） | 188 MB | 单次前向 CTC |
| 标点 | `CT-Transformer ct-punc` zh-en（`FunASR`，int8） | 65 MB | 文本级 transformer |

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

## 实测（Intel Core Ultra 5 336H，4 线程，release）

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

## CLI（开发/诊断）

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

## 可观测性

每句话一棵 root span `voice_utterance`，子 span 覆盖前端/推理/解码/标点，
属性带音频时长、耗时与 **RTF**（推理耗时 / 音频时长）——这是语音链路唯一有意义的
性能指标：<1 才可能跟得上说话，PTT 体验上要求 <0.3。重采样与 VAD 也各自埋了 span
（无上层 context 时零开销，采集回调里也能安全埋）。
