# 架构

## 工作区结构

18 个 crate 的 Cargo workspace，所有依赖在根 `Cargo.toml` 的
`[workspace.dependencies]` 统一定义（精确锁定版本），各 crate 通过
`workspace = true` 引用。

```
crates/
├── cnt-config        配置：TOML 加载（候选词数 + 语音输入）
├── cnt-ibus          IBus D-Bus 协议：对象序列化 + 私有总线地址发现
├── cnt-dict          词库：二进制(.cntd) + mmap 零拷贝读取 + 用户调频/用户词库
├── cnt-store         二进制 mmap 存储底座：错误类型/字节读取/区域校验（cnt-dict 与 cnt-lm 共享）
├── cnt-lm            n-gram 语言模型：二进制(.cntl) + mmap 二分查找（自研，无外部库）
├── cnt-score         打分端口层（零依赖）：NgramLm 打分接口 + Rescorer 重排契约 + 触发/融合策略
├── cnt-decode        拼音切分（标准 410 音节表）+ 整句解码（beam search + 词级格）
├── cnt-dict-tools    词库/LM CLI：build / build-lm / decode / info / query / bench
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

## 数据流

```
应用程序 ──► ibus-daemon ──► cnt-daemon（本输入法引擎）
                 │                    │
                 │                    └─► 麦克风 ─► cnt-audio ─► cnt-asr-onnx（本地推理）
                 └──► ibus-panel（画候选窗，不需要我们写 GUI）
```

引擎只需通过 D-Bus 把数据发出去：`UpdatePreeditText`（拼音预编辑）、
`UpdateLookupTable`（候选）、`CommitText`（上屏）；需要上下文时发
`RequireSurroundingText` 请应用下发光标周围文本。

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

## 搜索与打分解耦（cnt-score）

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
