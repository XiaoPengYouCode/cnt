# cnt

**一个用 Rust 写的简体中文拼音输入法**，跑在 IBus 上：整句解码、越用越顺手、自带纯本地语音输入。

## 特性

- **整句解码**：自研 beam search + 词级格（无开源解码库），标准 410 音节表；热键延迟 ~80µs
- **越用越顺手**：用户调频 + 自动学习新词，30 天半衰期淡出、2 万条封顶
- **纯本地语音输入**：按住说话 / 切换式常开听写，音频不出本机（Fun-ASR-Nano + CT-Transformer 标点，前端全自研）
- **模糊音**：平翘舌 / 边鼻音 / 前后鼻音 / 常见声母混淆，按代价分档，精确读音优先
- **增量确认（Rime 式）**：部分候选只确认一段，剩余拼音继续组合，不用删光重打
- **轻量**：mmap 零拷贝词库 + 自研 n-gram LM（270k unigram + 480 万 bigram），与英文输入、Rime 互不冲突

## 架构

```
应用程序 ──► ibus-daemon ──► cnt-daemon（本输入法引擎）
                 │                    │
                 │                    └─► 麦克风 ─► cnt-audio ─► cnt-asr-onnx（本地推理）
                 └──► ibus-panel（画候选窗，不需要我们写 GUI）
```

## 快速上手

```bash
# 1. 准备数据（一次性）——词库/语言模型不入库，从 fcitx5 官方源下载转换
#    dict_sc.txt（约 30 万词条）→ data/cnt.dict（编译后 13 万词条）
#    lm_sc.arpa → data/lm.cntl
cargo run -p cnt-dict-tools -- import-libime dict_sc.txt data/wordlist.tsv --lm lm_sc.arpa
cargo run -p cnt-dict-tools -- build       data/wordlist.tsv data/cnt.dict
cargo run -p cnt-dict-tools -- build-lm    lm_sc.arpa data/lm.cntl

# 2. 启动引擎（数据默认在 ~/.local/share/cnt/，可用环境变量覆盖）
CNT_DICT=data/cnt.dict CNT_LM=data/lm.cntl cargo run -p cnt-daemon

# 3. 另一终端：测试客户端模拟输入（a-z、空格、回车、退格、1-9、方向键，空行退出）
cargo run -p cnt-test-client
```

环境变量：`CNT_DICT`、`CNT_LM`、`CNT_USER_DB`、`CNT_ASR_DIR`、`CNT_PUNCT_DIR`
（默认都在 `$XDG_DATA_HOME/cnt/`）、`CNT_CONFIG`（配置路径）。

## 键位

| 键 | 作用 |
|---|---|
| `a`-`z` | 输入拼音 |
| `'` | 音节分隔：组合中强制切分（`xi'an` → 西安，不被当 xian/现）；未组合时是智能引号 |
| `空格` | 选中光标处候选（部分候选 → 确认该段，继续组合） |
| `1`-`9`、`0` | 直接选该序号的候选（`0` = 页内第 10 个；超出本页的序号不响应） |
| `Ctrl+Delete` | 删除光标处候选（忘记误学的词，撤销自动学习） |
| `←` `→` / `↑` `↓` | 在候选间移动光标（跨页自动翻页） |
| `PageUp` `PageDown` / `-` `=` | 翻页（光标落到新页首） |
| `退格` | 删未确认的字母；未确认部分为空时撤销上一次部分确认 |
| `Esc` | 丢弃整个组合（含已确认段） |
| `回车` | 把未确认的拼音原文当英文提交 |
| `Shift` 单击 | 中/英模式切换（350ms 内按下释放） |
| 按住 `右 Alt` | 语音输入：按住说话，松手识别上屏（需开启 `[voice]`） |
| `Ctrl+Shift+空格` | 语音输入：切换式常开听写（VAD 自动切句） |
| 录音中 `Esc` | 放弃这次语音，一个字都不上屏 |

## 语音输入

纯本地：模型在你机器上跑，音频只在内存里流动，不上传、不落盘。

- **按住说话**（默认右 `Alt`）：短句，松手识别
- **切换式常开**（默认 `Ctrl+Shift+空格`）：长段口述，VAD 自动切句、逐句上屏

识别用 Fun-ASR-Nano 的 CTC 导出（188 MB，int8）+ CT-Transformer 标点（65 MB），
`cnt-lm` 对 n-best 重排纠「音对字错」，`user.dict` 热词偏置。
链路细节（端口契约、IME 显示、实测数据）：[docs/voice-input.md](docs/voice-input.md)。

```bash
bash scripts/fetch-asr-model.sh   # 声学 188 MB → data/asr，标点 65 MB → data/punct
```

## 安装

```bash
# 1. 先按「快速上手」生成 data/cnt.dict 和 data/lm.cntl
# 2. 安装（最后一步 ibus restart 会重启会话输入法，属正常现象）
./scripts/install.sh          # 拼音 + 语音（自动下载模型，约 324 MB）
./scripts/install.sh --no-voice   # 只装拼音
```

安装布局：

| 内容 | 位置 | 权限 |
|---|---|---|
| 主程序 | `~/.local/bin/cnt-daemon` | 用户级 |
| 词库/语言模型 | `~/.local/share/cnt/{dict.cntd,lm.cntl}` | 用户级 |
| 语音模型 | `~/.local/share/cnt/{asr,punct}` | 用户级 |
| 配置 | `~/.config/cnt/config.toml`（缺 `[voice]` 时自动补上并开启语音） | 用户级 |
| ibus 组件 | `/usr/share/ibus/component/cnt.xml` | sudo（唯一特权操作） |

构建依赖：麦克风采集走 ALSA，需要 dev 包（`sudo apt install libasound2-dev`，
Fedora 用 `alsa-lib-devel` / Arch 用 `alsa-lib`）；PipeWire 自带 ALSA 兼容层，
无需额外后端。`install.sh` 会先检查，缺就自动装。

卸载：

```bash
sudo rm /usr/share/ibus/component/cnt.xml
rm -f ~/.local/bin/cnt-daemon
rm -rf ~/.local/share/cnt
ibus restart
```

## 测试

```bash
cargo test      # 二进制格式 / mmap 查询 / 打分 / 用户调频 / 按键状态机 / 语音前端与 VAD
```

候选质量回归（读音误配检测）：`cargo build --release -p cnt-dict-tools` 后跑
`.agents/skills/fuzzy-test/fuzzy_test.py`（`selftest` / `sample` / `mono`），
详见 [docs/pinyin-engine.md](docs/pinyin-engine.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 工作区结构、架构、搜索/打分解耦（cnt-score） |
| [docs/pinyin-engine.md](docs/pinyin-engine.md) | 整句解码、候选分组、增量确认、模糊音、质量回归 |
| [docs/input-behavior.md](docs/input-behavior.md) | 中文标点、中/英切换、配置 |
| [docs/dictionary.md](docs/dictionary.md) | 词库数据管线、用户模型与写盘策略 |
| [docs/voice-input.md](docs/voice-input.md) | 语音输入链路全貌（端口/n-best 重排/模型/实测） |
| [docs/performance.md](docs/performance.md) | 性能数据、测量纪律、热路径设计 |
