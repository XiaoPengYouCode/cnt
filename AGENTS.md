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
- **埋点**：
  - **库代码用 `LocalSpan::enter_with_local_parent("名")`**（不是 `Span::…`）——
    `LocalSpan` 是 fastrace 为「单线程内的子 span」优化的类型，更轻；
    上层无 context 时零开销，热路径放心埋。
  - 应用代码建 root span（`Span::root` + `set_local_parent`），退出/批次结束
    `fastrace::flush()`。span 名小写下划线。
  - **属性 vs 事件**（按 OpenTelemetry 口径，别混）：
    - **属性**（`LocalSpan::add_property` / `Span::with_property`）描述这个 span
      **干了多少活、在什么参数下** —— `frames=93`、`nbest=8`、`expand=8866`、`rtf=0.016`。
      聚合表能按 span 名直接汇总，是性能分析的主要输入。
    - **事件**（`Event::new`）只用于「期间发生了某件事」—— 如 `rescored`
      （语言模型改写了一次结果）。事件不适合承载工作量指标。
  - **端到端延迟必须单独埋一个 span**：只埋内部各阶段会产生「span 树看着很快、
    人却觉得慢」的盲区。语音的 `voice_release_to_commit`（松手→上屏）就是为此存在的——
    它覆盖排空音频、识别、重排、标点、事件传递的全过程，PTT 的 300ms 预算管的是这个数字。
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
