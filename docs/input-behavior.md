# 输入行为：标点 / 中英切换 / 配置

## 中文标点（cnt-input）

拼音模式使用中文标点与 ASCII 符号混合的默认映射：
`，。、；：？！【】『』（）《》＄·`，
`"` 与 `'` 为智能引号（`“”` / `‘’` 交替开合）。
`[` / `]` 默认输出 `【` / `】`；`\` 默认输出 `、`，`_` 默认输出 `——`，
`^` 默认输出 `……`；`+ - * = / @ # % & | ~` 无论上下文都保持 ASCII。
组合输入中按标点：先提交候选再附带标点（`ni` + `,` → `你，`，
`ni` + `+` → `你+`）。例外：有未确认拼音时 `-` / `=` 优先翻页，
`'` 用于拼音音节分隔；这些按键此时不输出符号。

**上下文例外**：紧跟 ASCII 字母/数字的第一个标点使用 ASCII 映射（`3,` / `abc.` / `x=`），
例如 `[`、`\`、`_`、`^` 在这种上下文中仍输出原 ASCII 字符。其他情况使用上面的默认映射。
判定只看紧邻的前一个字符，所以 `3,` 后再打标点回到默认映射，
空格也不算 latin 上下文。上下文按优先级取：

1. 我们自己刚上屏 / 刚转发的字符（最新，不依赖应用配合）；
2. 应用下发的 surrounding text（`cursor_pos` 按**字符**计）；
3. 两者都无 → 中文模式默认映射。

焦点切换/重置会丢弃上下文，并发 `RequireSurroundingText` 请应用下发光标周围文本。

### 映射依据与后续候选（2026-09-30 调研）

- [微软拼音官方说明](https://support.microsoft.com/en-us/windows/hardware/input-devices/microsoft-simplified-chinese-ime)
  将中/英文、全/半角、中文/英文标点设为独立切换项。中文模式不要求所有符号全角。
- [Fcitx 5 简体标点表](https://github.com/fcitx/fcitx5-chinese-addons/blob/master/modules/punctuation/punc.mb.zh_CN)
  将 `【】` 列为方括号键的首个映射；
  [雾凇拼音](https://github.com/iDvel/rime-ice/blob/main/default.yaml) 的 `half_shape`
  保留 ASCII `+ - * =`，但方括号使用 `〖〗`。输入法之间有差异，不存在统一键位标准。
- 两者都将 `_` 映射为 `——`，cnt 采用这一默认映射。
  `/`、`@ # % & | ~` 保留 ASCII，`\` 映射为 `、`，`^` 映射为 `……`，
  参考雾凇的 `half_shape`。
- [Unicode East Asian Width](https://unicode.org/reports/tr11/) 解释全角兼容字符与旧式混合宽度编码的关系；
  [W3C 中文排版需求](https://www.w3.org/TR/clreq/#h-punctuation-marks) 描述中文句读、括号等的用途和排版。
  据此，本项目的设计取舍是：中文句读照顾阅读排版，运算与标识符符号照顾混合文本和软件互操作。
  这是对资料的归纳，不代表各输入法厂商公开陈述的设计动机。

后续改进建议（尚未实施）：数字后标点可进一步细分为小数点、时间分隔等场景，
避免现有“任何字母/数字后的第一个标点都用 ASCII”规则在中文句末输出英文逗号、句号。

## 中/英切换（Shift）

- **Shift 单击**（350ms 内按下释放、期间未打其他键）→ 切换中/英模式；
  组合中切换会先把预编辑上屏
- **英模式**：所有按键透传（应用原生输入）
- **Shift+字母**：中模式下未组合时透传（输出大写）；组合中先提交预编辑再透传
- 模式按输入上下文隔离，切换窗口不影响其他窗口，`focus_out` 不清除

## 配置（cnt-config）

TOML 配置文件：`~/.config/cnt/config.toml`（`CNT_CONFIG` 环境变量可覆盖路径）：

```toml
# 每页候选词数（范围 5~10，默认 8）
# 上限 10 是选择键决定的：数字键只有 `1`-`9` 加 `0`（第 10 个）
page_size = 8
# 整词内的单次键盘纠错（邻键误触 / 字母颠倒 / 重复按键）
# 关闭不影响方音模糊；启动时读取
typo_correction = true

# 语音输入（默认关：模型不随程序分发，得先装）
[voice]
enabled = true
model_dir = "~/.local/share/cnt/asr"
language = "auto"                    # auto/zh/en/ja/ko/yue
itn = true                           # 数字规整 + 标点
threads = 4
device = ""                          # 麦克风设备名子串，空 = 系统默认
punctuation = true                   # 标点恢复（独立的小模型，见 cnt-asr 的 Punctuator 端口）
punct_dir = "~/.local/share/cnt/punct"
ptt_key = "Alt_R"                    # 按住说话（右 Alt）
toggle_key = "Control+Shift+space"   # 切换式常开
max_seconds = 60.0                   # PTT 单次最长录音
trailing_silence_ms = 700            # 常开：停顿多久算一句结束
vad_margin_db = 10.0                 # 高于噪声底多少 dB 算语音
```

**逐项解析**而不是整体反序列化：**单项写错不影响其他项**（输入法不能因为一个写错的
热键就不能打字），越界值夹紧到合法区间（`threads` 0~64、`max_seconds` 1~600、
`trailing_silence_ms` 100~5000、`vad_margin_db` 1~40）。`~` 开头的路径会展开。
