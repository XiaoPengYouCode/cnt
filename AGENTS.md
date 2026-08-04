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

## 可观测性（埋点基础设施）

**原则：代码充分透明。** 热点/延迟路径必须埋点，性能判断以数据为准，不靠感觉。

- **技术栈**：`log`（日志）+ `fastrace`（span 树）+ `logforth`（分发，日志挂 span）。
  禁止 `eprintln!` 调试输出；**禁止用手工 `Instant` 计时当正式观测**（临时定位可以，用完必删）。
- **埋点**：库代码用 `Span::enter_with_local_parent("名")` 建子 span——上层无 context 时
  零开销，热路径放心埋；应用代码建 root span（`Span::root` + `set_local_parent`），
  退出/批次结束 `fastrace::flush()`。span 名小写下划线，工作量指标用 `Event` 属性
  （如 `expand(count=…)`）。
- **性能对照**：`cnt-dict-tools bench <dict> <lm> --user <u.dict> <rounds>` 输出
  wall-time 分位数 + fastrace 阶段聚合表；改动前后各跑一次对比。
- **daemon 诊断**：每按键一棵 `process_key_event` root span，<5ms 的 cancel 不上报，
  慢按键的 span 树才会落盘——stderr 出现树即延迟异常，据此定位卡顿阶段。
- **工作流**：先埋 span 再优化 → span 树确认瓶颈 → 聚合表验证效果 → 确认质量无回退。
