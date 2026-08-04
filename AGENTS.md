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
- **日志/追踪**：统一 `log`（宏）+ `fastrace`（span）+ `logforth`（分发/appender）。
  禁止新增 `eprintln!` 调试输出。
- **依赖**：所有依赖统一定义在根 `Cargo.toml` 的 `[workspace.dependencies]`，
  精确锁定版本（`=x.y.z`），各 crate 只写 `xxx.workspace = true`。
- **词库**：`data/` 不入库，通过 `cnt-dict-tools` 数据管线独立分发。
