# AGENTS.md

## 硬性约束

- **禁止在调试过程中安装/注册引擎到系统输入法**（ibus 等）。
- 除 `cargo build` / `cargo test` / `cargo clippy` 外，**未经用户明确允许，不得**：
  - 启动 `cnt-daemon` 或其他会执行 `RegisterComponent` 的操作；
  - 运行端到端测试（如通过 D-Bus 发送按键、验证上屏链路）。
- 若用户允许上述操作，结束后必须清理进程，并确认 ibus 活动引擎不再指向 cnt。

## 工程约定

- **时间库**：需要时用 `jiff`，**禁止使用 `chrono`**。
- **错误处理**：用 `thiserror` 定义显式错误类型，**禁止使用 `anyhow`**。
- **依赖**：所有依赖统一定义在根 `Cargo.toml` 的 `[workspace.dependencies]`，
  精确锁定版本（`=x.y.z`），各 crate 只写 `xxx.workspace = true`。
- **词库**：`data/` 不入库，通过 `cnt-dict-tools` 数据管线独立分发。
  语音模型同理（`data/asr`、`data/punct`，共 ~324 MB，走 `scripts/fetch-asr-model.sh`）。
- **产物不入库**：性能采样与诊断产物（`perf.data`、`perf.data.old`、`*.perf`、
  `flamegraph.svg`）、模型与词库二进制、`target/` 一律不提交。
  已在 `.gitignore` 里挡住，**但不要用 `git add -A` 碎片式提交**：
  提交前先 `git status --porcelain -uall` 看清单，并确认没有 >1 MB 的文件
  （`perf.data` 就是这么溢进去的，后来靠重写历史才拿出来）。

## 可观测性（埋点基础设施）

**原则：代码充分透明。** 热点/延迟路径必须埋点，性能判断以数据为准，不靠感觉。

- **技术栈**：`log`（日志）+ `fastrace`（span 树）+ `logforth`（分发，日志挂 span）。
  禁止 `eprintln!` 调试输出；**禁止用手工 `Instant` 计时当正式观测**（临时定位可以，用完必删）。
- **trace 开关**：fastrace 的 `enable` 是编译期开关，依赖 feature 在构建图内全局合并。
  发布构建（install.sh、`cargo build --release -p cnt-daemon`）默认关闭——span 宏编译为
  noop、零开销；诊断/bench 时 `cargo build --release --features "fastrace/enable"`
  全局开启（一条命令影响整个构建图）。
- **埋点规范**（一致、可复现、可复用；按 crate 角色分工）：

  | 角色 | crate | 必须有 | 禁止 |
  |---|---|---|---|
  | **端口层** | cnt-asr, cnt-score, cnt-input, cnt-config, cnt-store, cnt-ibus | 不埋（纯契约/纯数据结构/一次性加载）；例外：端口层里的纯算法若单次 >10µs，埋一个阶段 span | 依赖 fastrace 之外的观测设施 |
  | **库（算法/IO）** | cnt-decode, cnt-dict, cnt-lm, cnt-asr-onnx, cnt-audio, cnt-voice, cnt-asr-lm | `LocalSpan::enter_with_local_parent` 埋**阶段边界**；工作量用 `add_property` | `Span::enter_with_local_parent`（重）、`set_reporter`、`flush`、逐次查询埋点 |
  | **应用（二进制）** | cnt-daemon, cnt-dict-tools, cnt-asr-tools, cnt-test-client | 启动即 `set_reporter`（**所有子命令**，不只 bench）；退出/批次结束 `flush()` | 一个长会话用一棵大 root（常开听写要按句切） |

  **root span 建在「处理用户操作的那一层」，不一定是二进制 crate**：
  cnt-daemon 只负责装 reporter + flush，真正的 root span 在 cnt-engine
  （`process_key_event`）和 cnt-voice（`voice_release_to_commit`/`voice_segment`）里 ——
  它们才知道「一次用户操作」的边界。照表检查时别因为 daemon 里没有 `Span::root` 就以为漏了。

  **三类 span，缺一不可**：
  1. **端到端**（用户感知的那个数字）：`process_key_event`（按键→候选窗刷新完）、
     `voice_release_to_commit`（松手→上屏）、`client_key_roundtrip`（客户端视角往返）。
     **必须覆盖到用户看见结果为止**——只埋内部阶段会得到「span 树很快、人却觉得慢」的盲区。
     踩过：按键 root span 早先只覆盖同步段，把 D-Bus 的 UI 刷新排除在外，于是
     「候选窗闪」这类抱怨在 trace 里完全看不到，而且「>5ms 才上报」的阈值也漏掉了它。
  2. **阶段分解**：每一级耗时占比（`lattice`/`beam`/`keys_at` — `fbank`/`asr_infer`/`ctc_beam`/`punctuate`）。
  3. **工作量指标**：以**属性**形式挂在对应 span 上（`frames`/`nbest`/`expand`/`rtf`/`samples`）。

  **属性 vs 事件**（OpenTelemetry 口径，别混）：
  - **属性**（`LocalSpan::add_property` / `Span::with_property`）= 这个 span **干了多少活、
    在什么参数下**。聚合表能按 span 名直接汇总，是性能分析的主要输入。
  - **事件**（`Event::new`）= 期间**发生了某件事**，如 `rescored`（语言模型改写了一次结果）。
    不要用事件承载工作量指标。

  **`Span` 还是 `LocalSpan`**：默认 `LocalSpan`（fastrace 为单线程子 span 优化的类型）。
  只有两种情况用 `Span`：① 需要 `Span::root` 开一棵树；② 需要在 span 结束后补属性
  （如算完 RTF 再挂上去）或跨 await 传递（配 `FutureExt::in_span`）。

  **跨 await 的异步段**用 `Span::enter_with_parent(name, &root)` + `FutureExt::in_span`
  接进同一棵树；不要图省事让它掉出 trace。跨**任务/线程**时先用独立 root 量出数量级，
  确认值得再做 `SpanContext` 传递。

  **自查**：`bash scripts/check-tracing.sh`（规范的可执行版本，违规退出码 1）。
  它检查：库 crate 不用重 `Span`（例外须在上方 3 行内写 `// trace-exception: 原因`）、
  库里不出现 reporter/flush、应用 crate 有 reporter+flush、工作量指标没塞进 `Event`。
  **改完埋点跑一次**，和 `cargo clippy` 一样属于提交前的例行检查。

  **落地顺序**（新功能一律照这个来）：
  1. 先问「用户感知的那个数字是什么」→ 建端到端 root span，**含最后一步 IO**；
  2. 再按链路自上而下补阶段子 span（`LocalSpan`）；
  3. 再把工作量指标挂成属性；
  4. 应用侧接上 reporter + flush，确认 span 树真的能打出来（诊断命令没 reporter =
     埋了也看不见，踩过）；
  5. **最后**才优化：span 树确认瓶颈 → 改 → 聚合表验证 → 交替测量确认没劣化。

- **性能对照**：`cnt-dict-tools bench <dict> <lm> --user <u.dict> <rounds>` 输出
  **热/冷** wall-time 分位数（热 = 词键缓存命中 = 连续打字的第 2 键起；冷 = 每次
  清空缓存）+ fastrace 阶段聚合表；改动前后各跑一次对比。
- **测量纪律**：机器状态会漂移（频率/负载），跨时间点的绝对值不可比——用
  `git worktree` 建旧版本，**交替**跑新旧两个二进制再比（本项目实测：同一份
  二进制在降频状态下慢 2~3 倍，足以把优化误判成回退）。报数字时带上机器状态，
  并注明是否开了 trace（开启约 +4%，别拿 trace-on 的构建对比 trace-off 的）。
- **埋点开销预算**（fastrace benches：约 40ns/span、500ns/root span）：
  阶段级 span（每次解码个位数）成本可忽略；即使密到每候选词一个 span
  （约 500/解码）也只 +30%，诊断期完全可用——热路径别怕埋，怕的是拿不可比的
  数字下结论。
- **daemon 诊断**：每按键一棵 `process_key_event` root span，<5ms 的 cancel 不上报，
  慢按键的 span 树才会落盘——stderr 出现树即延迟异常，据此定位卡顿阶段。
- **工作流**：先埋 span 再优化 → span 树确认瓶颈 → 聚合表验证效果 → 确认质量无回退。
