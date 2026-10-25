# 词库与用户模型

## 词库数据说明

词库（`data/` 下的 tsv 与二进制）**不入库**，通过数据管线独立分发：

1. **来源**：fcitx5/libime 的 `dict_sc.txt`，原始文件长期保留在
   `data/raw/libime/`，与转换产物一起由 `.gitignore` 的 `/data/` 忽略。
   固定数据包为 `https://download.fcitx-im.org/data/dict-20260703.tar.zst`，
   SHA-256：`c686cab6df8964c48d596f57d205bac31fc72870b06a83017e44503df8c09697`。
   包内 `dict_extb.txt` 也保留，但默认不合入；本地领域词维护在
   `data/wordlist-domain.tsv`，重建时追加。
2. **转换**（`cnt-dict-tools import-libime --lm`）：
   - 原始文件的空格和制表符均是字段分隔符；无权重的词条默认读音权重为 0。
   - 音节分隔符 `'` 去掉（`ni'hao` → `nihao`）。
   - TSV 为 `拼音键、词、排序词频、log10 读音权重` 四列；旧的两列/三列仍可编译。
   - 排序词频由 `unigram + 读音权重` 连续换算，用于无 LM 查询等路径；
     权重为 0 的 LM 地板词保持原有词频提升。解码器直接使用 LM 分数和第 4 列，
     不从排序词频反推权重，也不重复累加。
   - **读音权重独立保存**，按 libime 的语义与 LM 分数相加，不按阈值归零。
     候选裁剪使用合成分数；整句续接使用 bigram 分数再加该词的读音权重。
3. **编译**：生成 v2 `.cntd`（每候选增加一个 f32 读音权重）。新程序兼容读 v1，
   旧词库缺少权重时保留历史评分规则；旧程序不能读 v2，发布时需同步更新程序和词库。

日常维护改 `data/raw/libime/dict_sc.txt` 或 `data/wordlist-domain.tsv`，不要只改
会被重建覆盖的 `data/wordlist.tsv`。已有 `data/lm.cntl` 时运行：

```bash
bash scripts/rebuild-dict.sh
```

脚本只在原始主词库缺失时下载并校验固定数据包，通常离线使用本地原始文件；
转换和编译成功后替换 `data/wordlist.tsv`、`data/cnt.dict`，不安装或启动输入法。
2026-10-25 完整导入 300,234 条原始词条，追加 2,551 条领域词。
旧导入器仅识别制表符，漏掉 95,175 条空格分隔记录；已一并修正。

回归测试包含：空格/制表符导入、读音权重的二进制往返、v1 兼容、多个常用
读音共存、候选裁剪前计入读音权重、bigram 续接只计入一次权重，以及非有限值拒绝。
可用 `cargo test -p cnt-dict -p cnt-decode -p cnt-dict-tools` 运行，均为离线测试。

语言模型 `.cntl`（`build-lm` 生成，格式自研）：270k unigram + 480 万 bigram，
mmap 二分查找，trigram 留作扩展（Katz backoff 已预留）。

## 用户模型（cnt-dict）

- **动态调频**：用户通过候选上屏即计数；拼音解码侧是 **log-linear 混合**（libime 口径，
  见 docs/pinyin-engine.md「打分」），语音 n-best 是粗粒度有界加法（同一把尺，
  `cnt-score::policy::user`）。单条计数封顶 100。
- **时间衰减**：计数按 30 天半衰期衰减（jiff 时间戳），习惯漂移后旧词淡出
- **词库上限**：2 万条封顶，超出淘汰有效计数最低的词条（未转正词天然优先被淘汰）
- **低可信度起点**：自动造出的新词（不在静态词库）首现 `confirmed=false`，
  有效计数 ×0.1（对齐 librime dee=0.1），只可见不竞争；第 2 次确认转正满权重。
  词库已有的词调频不受影响。tsv 第 5 列持久化（旧 4 列数据视为已转正，兼容）
- **新词学习**：提交的多段句子按相邻段拼成复合词（郑+爽 → zhengshuang/郑爽），
  下次输入完整拼音直接出整词；学过的词也能用在**句子中间**——多音节链剪枝
  （`has_key_prefix`）与用户库一起判断前缀存在性，且不在 LM 词表的词用「句首基础分
  + backoff」当伪 unigram（而非一律 UNK -12），否则学过的词在句中必然输给逐字拼接
- **忘记此词**（撤销误学）：候选误选 / 拼错的复合词会污染排序——组合中按
  `Ctrl+Delete` 删除光标处候选的全部学习段（词库静态词不受影响，仅去掉调频加成），
  候选窗即时刷新；CLI 等价操作 `cnt-dict-tools delete-user <user.dict> <pinyin> <word>`

### 写盘策略

用户学习数据 **30s 周期兑底**（长时间不切走输入框的连续输入），
`focus_out`/`disable` 时立即 flush，优雅退出时再 flush 一次。
写盘是临时文件 + rename 原子替换。
