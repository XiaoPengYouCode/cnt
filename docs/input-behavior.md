# 输入行为：标点 / 中英切换 / 配置

## 中文标点（cnt-input）

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
