//! 拼音输入逻辑：维护输入缓冲区与候选列表，处理按键。
//!
//! 纯逻辑、无 IO：候选通过 `CandidateSource` 抽象注入；
//! 每个候选携带可学习信息（句子 = 多个 (拼音, 词) 对），
//! 上层在提交时据此记录用户调频。

pub mod hotkey;

pub use hotkey::Hotkey;

/// 默认每页候选数（可用配置覆盖；与 `cnt_config::DEFAULT_PAGE_SIZE` 一致）
pub const PAGE_SIZE: usize = 8;

/// 数字选择键最多覆盖的候选数：`1`-`9` 再加 `0`（= 页内第 10 个）。
///
/// 超过这个数的候选只能靠方向键走过去 —— 所以配置侧把 `page_size` 的上限
/// 压在这里，不让候选窗里出现「看得见按不到」的序号。
pub const MAX_SELECT_KEYS: usize = 10;

/// 一个学习片段：拼音 + 对应的词（单字/词 = 1 段，整句 = 多段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LearnedWord {
    pub pinyin: String,
    pub word: String,
}

impl LearnedWord {
    #[must_use]
    pub fn new(pinyin: impl Into<String>, word: impl Into<String>) -> Self {
        Self {
            pinyin: pinyin.into(),
            word: word.into(),
        }
    }
}

/// 一个候选（词、整句，或只覆盖输入前一段的「部分候选」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub text: String,
    /// 用于用户调频学习的拼音-词段：单字/词 = 1 段，整句 = 多段。
    pub learned: Vec<LearnedWord>,
    /// 该候选消耗掉的**输入字节数**。
    ///
    /// 不能从 `learned` 的拼音键推出来：模糊音的键和输入长度不同
    /// （输入 `sihou` 走的键是 `shihou`）。小于当前输入长度 = 部分候选，
    /// 选中后只确认这一段、剩余拼音继续组合（Rime 式增量确认）。
    pub consumed: usize,
}

impl Candidate {
    /// 覆盖整个输入的候选（整句 / 整键词 / 补全）。
    #[must_use]
    pub fn whole(text: impl Into<String>, learned: Vec<LearnedWord>, input_len: usize) -> Self {
        Self {
            text: text.into(),
            learned,
            consumed: input_len,
        }
    }

    /// 只覆盖输入前 `consumed` 字节的部分候选。
    #[must_use]
    pub fn partial(text: impl Into<String>, learned: Vec<LearnedWord>, consumed: usize) -> Self {
        Self {
            text: text.into(),
            learned,
            consumed,
        }
    }

    /// 是否覆盖了全部输入（`consumed >= input_len`）。
    #[must_use]
    pub const fn covers_all(&self, input_len: usize) -> bool {
        self.consumed >= input_len
    }
}

/// 候选来源：输入逻辑查询候选的统一接口（由 cnt-decode 的整句解码器实现）。
pub trait CandidateSource {
    fn candidates(&self, pinyin: &str) -> Vec<Candidate>;

    /// 查询候选时带上已经确认的前文。
    ///
    /// 默认回退到旧接口，保持纯输入逻辑和测试用候选源的兼容性；解码器可以
    /// 用这些片段初始化语言模型上下文，让「确认了前半句后继续输入」仍然受
    /// 前文影响，而不是每次从句首重新开始。
    fn candidates_with_context(&self, pinyin: &str, _context: &[LearnedWord]) -> Vec<Candidate> {
        self.candidates(pinyin)
    }
}

/// 输入模式（中/英）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Chinese,
    English,
}

/// 按键处理结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// 本引擎不处理，转发给应用程序
    Forward,
    /// 已处理（更新了状态，需要刷新界面）
    Handled,
    /// 提交一段文字
    Commit {
        text: String,
        /// 上屏内容对应的学习段（整句 = 多段）；直接提交原始拼音串时为空。
        learned: Vec<LearnedWord>,
    },
    /// 提交组合，并把原按键转发给应用（Shift+字母：先上屏预编辑，大写字母交给应用）。
    CommitAndForward {
        text: String,
        learned: Vec<LearnedWord>,
    },
}

/// ASCII 标点键 → 中文标点（简体，参考 `Rime` `luna_pinyin` 的默认映射）。
/// `"` 与 `'` 为智能引号，交替开合（用 `quote_open` 状态）。
/// `half_width`：前一个字符是数字/英文时用半角标点（`3，` → `3,`）。
fn punct_of(keyval: u32, quote_open: &mut bool, half_width: bool) -> Option<String> {
    let c = char::from_u32(keyval)?;
    // (全角, 半角)
    let pair = match c {
        ',' => ("，", ","),
        '.' => ("。", "."),
        '/' | '\\' => ("／", "/"),
        ';' => ("；", ";"),
        ':' => ("：", ":"),
        '?' => ("？", "?"),
        '!' => ("！", "!"),
        '[' => ("「", "["),
        ']' => ("」", "]"),
        '{' => ("『", "{"),
        '}' => ("』", "}"),
        '(' => ("（", "("),
        ')' => ("）", ")"),
        '<' => ("《", "<"),
        '>' => ("》", ">"),
        '~' => ("～", "~"),
        '@' => ("＠", "@"),
        '#' => ("＃", "#"),
        '$' => ("＄", "$"),
        '%' => ("％", "%"),
        '^' => ("＾", "^"),
        '&' => ("＆", "&"),
        '*' => ("＊", "*"),
        '-' => ("－", "-"),
        '_' => ("＿", "_"),
        '+' => ("＋", "+"),
        '=' => ("＝", "="),
        '|' => ("｜", "|"),
        '`' => ("·", "`"),
        '"' => {
            // 英文/数字后（写代码、写 `x="1"`）要的是直引号，不是弯引号
            if half_width {
                return Some('"'.to_string());
            }
            let p = if *quote_open { "”" } else { "“" };
            *quote_open = !*quote_open;
            return Some(p.to_string());
        }
        '\'' => {
            if half_width {
                return Some('\''.to_string());
            }
            let p = if *quote_open { "’" } else { "‘" };
            *quote_open = !*quote_open;
            return Some(p.to_string());
        }
        _ => return None,
    };
    Some(if half_width { pair.1 } else { pair.0 }.to_string())
}

/// 标点宽度判定：光标前一个字符是否为 ASCII 字母/数字。
///
/// 语义就是「英文/数字后面紧跟的**第一个**标点用半角」：只看紧邻的那一个字符，
/// 所以 `abc,` 之后再打标点（前一字符是 `,`）会回到全角。
///
/// `cursor_chars` 是 **字符** 偏移 —— `IBus` 的 `SetSurroundingText` 的 `cursor_pos`
/// 按字符计（不是字节）。曾经把它当字节下标用：中文一字 3 字节，字符偏移 N 会
/// 落在文本约 1/3 处，只要前文有任何英文/数字就误判成半角，表现为「所有标点都变半角」。
///
/// 光标越界（应用给的 surrounding text 过期）时返回 `false`，即回到全角——
/// 宁可全角错一次，也不要在中文里冒出半角标点。
#[must_use]
pub fn latin_before_cursor(text: &str, cursor_chars: usize) -> bool {
    if cursor_chars == 0 {
        return false; // 行首/空缓冲：没有前一个字符
    }
    text.chars()
        .nth(cursor_chars - 1)
        .is_some_and(|c| c.is_ascii_alphanumeric())
}

/// X11 keysym（IBus 的 keyval 即 keysym）
pub mod keysym {
    pub const SPACE: u32 = 0x0020;
    pub const BACKSPACE: u32 = 0xff08;
    pub const RETURN: u32 = 0xff0d;
    pub const ESCAPE: u32 = 0xff1b;
    pub const LEFT: u32 = 0xff51;
    pub const UP: u32 = 0xff52;
    pub const RIGHT: u32 = 0xff53;
    pub const DOWN: u32 = 0xff54;
    pub const PAGE_UP: u32 = 0xff55;
    pub const PAGE_DOWN: u32 = 0xff56;
    pub const A: u32 = 0x61;
    pub const Z: u32 = 0x7a;
    pub const DIGIT0: u32 = 0x30;
    pub const DIGIT1: u32 = 0x31;
    pub const DIGIT9: u32 = 0x39;
    /// 翻页键（Rime 行为）：- 上一页，= 下一页
    pub const MINUS: u32 = 0x002d;
    pub const EQUAL: u32 = 0x003d;
    /// 音节分隔键（Rime 惯例）：组合中强制切分（xi'an → xi+an）；
    /// 未组合时是智能引号（`handle_punctuation`）
    pub const APOSTROPHE: u32 = 0x0027;
}

/// 数字选择键 → 页内序号：`1`-`9` → 0..=8，`0` → 9（候选窗里的第 10 个）。
const fn select_offset(keyval: u32) -> Option<usize> {
    match keyval {
        keysym::DIGIT0 => Some(MAX_SELECT_KEYS - 1),
        keysym::DIGIT1..=keysym::DIGIT9 => Some((keyval - keysym::DIGIT1) as usize),
        _ => None,
    }
}

/// 输入组合状态（聚合根）：字段私有，行为通过方法暴露。
pub struct EngineState {
    /// 未确认的拼音串（已确认的部分不在里面）
    buffer: String,
    /// 已确认的段（拼音, 词）：选中「部分候选」后先攒在这里，不马上上屏
    confirmed: Vec<LearnedWord>,
    /// 已确认段拼出的汉字（预编辑左半部分，避免每次重新拼接）
    confirmed_text: String,
    /// 全部候选（多页）
    candidates: Vec<Candidate>,
    /// 当前页码（从 0 开始）
    page: usize,
    /// 当前页内的光标位置
    cursor: usize,
    /// 每页候选数（配置可调）
    page_size: usize,
    /// 智能引号开合状态
    quote_open: bool,
    /// 中/英模式（不随 `clear`/`focus_out` 清除）
    mode: InputMode,
}

impl Default for EngineState {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineState {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_page_size(PAGE_SIZE)
    }

    /// 指定每页候选数创建状态机。
    #[must_use]
    pub const fn with_page_size(page_size: usize) -> Self {
        Self {
            buffer: String::new(),
            confirmed: Vec::new(),
            confirmed_text: String::new(),
            candidates: Vec::new(),
            page: 0,
            cursor: 0,
            page_size,
            quote_open: false,
            mode: InputMode::Chinese,
        }
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.confirmed.clear();
        self.confirmed_text.clear();
        self.candidates.clear();
        self.page = 0;
        self.cursor = 0;
        // 引号开合状态是会话级的（交替产生 “/”），不随组合清除
    }

    /// 组合中（还有未确认拼音，或有已确认但未上屏的段）。
    #[must_use]
    pub const fn is_composing(&self) -> bool {
        !self.buffer.is_empty() || !self.confirmed_text.is_empty()
    }

    /// 已确认段拼出的汉字（预编辑左半部分；未做增量确认时为空）。
    #[must_use]
    pub fn confirmed_text(&self) -> &str {
        &self.confirmed_text
    }

    /// 把「已确认 + 本次选中」合成一次上屏，并清空组合。
    fn commit_with(&mut self, text: &str, learned: Vec<LearnedWord>) -> Action {
        let mut full = std::mem::take(&mut self.confirmed_text);
        full.push_str(text);
        let mut segs = std::mem::take(&mut self.confirmed);
        segs.extend(learned);
        self.clear();
        Action::Commit {
            text: full,
            learned: segs,
        }
    }

    /// 选中一个候选：整段覆盖则上屏，只覆盖前一段则**确认这一段**、
    /// 剩余拼音继续组合（Rime 式增量确认 —— 整句错了不必删光重打）。
    fn take_candidate(&mut self, cand: Candidate, source: &dyn CandidateSource) -> Action {
        if cand.covers_all(self.buffer.len()) {
            return self.commit_with(&cand.text, cand.learned);
        }
        self.confirmed_text.push_str(&cand.text);
        self.confirmed.extend(cand.learned);
        self.buffer.drain(..cand.consumed);
        self.refresh(source);
        Action::Handled
    }

    /// 撤销最后一次「部分确认」，把那段拼音退回未确认串的前面。
    fn unconfirm_last(&mut self, source: &dyn CandidateSource) -> bool {
        let Some(last) = self.confirmed.pop() else {
            return false;
        };
        // confirmed_text 去掉这一段的词
        let keep = self.confirmed_text.len() - last.word.len();
        self.confirmed_text.truncate(keep);
        self.buffer.insert_str(0, &last.pinyin);
        self.refresh(source);
        true
    }

    /// 重查候选并把翻页/光标复位。
    fn refresh(&mut self, source: &dyn CandidateSource) {
        self.candidates = if self.buffer.is_empty() {
            Vec::new()
        } else {
            source.candidates_with_context(&self.buffer, &self.confirmed)
        };
        self.page = 0;
        self.cursor = 0;
    }

    /// 当前输入模式。
    #[must_use]
    pub const fn mode(&self) -> InputMode {
        self.mode
    }

    /// 切换中/英模式；若切换前在组合中，返回待上屏的预编辑（commit 后交给上层）。
    pub fn toggle_mode(&mut self) -> Option<(String, Vec<LearnedWord>)> {
        self.mode = match self.mode {
            InputMode::Chinese => InputMode::English,
            InputMode::English => InputMode::Chinese,
        };
        self.take_composing()
    }

    /// 把当前组合中的内容取出来（用于「打断组合」的场合：中英切换、开始语音输入）。
    ///
    /// 有选中的候选就用它，否则原样吐出拼音串——**不能直接丢**：
    /// 用户已经打了字，任何情况下让它凭空消失都是数据丢失。
    pub fn take_composing(&mut self) -> Option<(String, Vec<LearnedWord>)> {
        if !self.is_composing() {
            return None;
        }
        let (text, learned) = self.selected().cloned().map_or_else(
            || (self.buffer.clone(), Vec::new()),
            |c| (c.text, c.learned),
        );
        let Action::Commit { text, learned } = self.commit_with(&text, learned) else {
            unreachable!("commit_with 只返回 Commit")
        };
        Some((text, learned))
    }

    /// 已输入的拼音串。
    #[must_use]
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// 全部候选（只读）。
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// 每页候选数。
    #[must_use]
    pub const fn page_size(&self) -> usize {
        self.page_size
    }

    /// 当前页码（从 0 开始）。
    #[must_use]
    pub const fn page(&self) -> usize {
        self.page
    }

    /// 光标在全部候选中的绝对位置（供 `IBusLookupTable.cursor_pos`，面板据此算页）。
    #[must_use]
    pub const fn cursor_abs(&self) -> usize {
        self.page * self.page_size + self.cursor
    }

    /// 第 index 个候选（越界返回 None）。
    #[must_use]
    pub fn candidate_at(&self, index: usize) -> Option<Candidate> {
        self.candidates.get(index).cloned()
    }

    /// 预编辑预览：正在**浏览候选**时，返回 `(选中候选的文本, 它消耗的输入字节数)`。
    ///
    /// 判据是「光标离开首位」——光标在 #1 时说明用户还没挑，预编辑保持分节拼音
    /// （看得见自己打了什么、怎么切分的）；一旦按方向键浏览，预编辑就同步预览
    /// 选中的候选（`你好世界` / 部分候选则是 `你好 shi jie`），空格确认前就能看到结果。
    ///
    /// 无状态判定：光标移回 #1 就回到拼音显示，行为可预测。
    #[must_use]
    pub fn preview(&self) -> Option<(&str, usize)> {
        if self.cursor_abs() == 0 {
            return None;
        }
        self.selected().map(|c| (c.text.as_str(), c.consumed))
    }

    /// 当前光标处的候选（若有）。
    #[must_use]
    pub fn selected(&self) -> Option<&Candidate> {
        self.candidates
            .get(self.page * self.page_size + self.cursor)
    }

    /// 处理一个按键（keyval 为 keysym），返回动作。
    ///
    /// `punct_half_width`：标点前的字符是数字/英文时用半角（由上层根据上下文给出）。
    pub fn handle_key(
        &mut self,
        keyval: u32,
        source: &dyn CandidateSource,
        punct_half_width: bool,
    ) -> Action {
        // 英模式：一律转发（应用原生输入）
        if self.mode == InputMode::English {
            return Action::Forward;
        }

        // 大写字母（Shift+字母）：未组合直接转发（应用输出大写）；
        // 组合中先提交预编辑再转发原键
        if (0x41..=0x5a).contains(&keyval) {
            if self.buffer.is_empty() {
                return Action::Forward;
            }
            let (text, learned) = self.selected().cloned().map_or_else(
                || (self.buffer.clone(), Vec::new()),
                |c| (c.text, c.learned),
            );
            let Action::Commit { text, learned } = self.commit_with(&text, learned) else {
                unreachable!("commit_with 只返回 Commit")
            };
            return Action::CommitAndForward { text, learned };
        }

        // 字母键：进入/继续拼音组合
        let lower = keyval;
        if (keysym::A..=keysym::Z).contains(&lower) {
            self.buffer.push(char::from_u32(lower).unwrap_or('a')); // lower 必在 a-z 范围
            self.refresh(source);
            return Action::Handled;
        }

        // 音节分隔键（Rime 惯例）：组合中 `'` = 强制切分（xi'an → xi+an，
        // 不被自动切分当成 xian/现）；未组合时保持智能引号（handle_punctuation）。
        // 要在 handle_punctuation 之前拦截，否则会被当标点「提交候选+引号」。
        if !self.buffer.is_empty() && keyval == keysym::APOSTROPHE {
            self.buffer.push('\'');
            self.refresh(source);
            return Action::Handled;
        }

        // 翻页键（Rime 行为：- 上一页、= 下一页）：组合中优先翻页
        if !self.buffer.is_empty() && (keyval == keysym::MINUS || keyval == keysym::EQUAL) {
            if keyval == keysym::MINUS {
                self.page_up();
            } else {
                self.page_down();
            }
            return Action::Handled;
        }

        // 标点键：未组合时直接上屏中文标点；组合中先提交候选再附带标点
        if let Some(action) = self.handle_punctuation(keyval, punct_half_width) {
            return action;
        }

        // 未在组合中（既无未确认拼音也无已确认段）：一律转发
        if !self.is_composing() {
            return Action::Forward;
        }

        // 数字键选候选：`1`-`9` 选页内第 1-9 个，`0` 选第 10 个（与候选窗标号一致）。
        // 序号落在本页之外（如 page_size=8 时按 `9`/`0`）就吞掉：既不能跳去选下一页
        // 的候选（用户按的是「本页第 9 个」），也不能漏成数字插进正在组合的文本里。
        if let Some(offset) = select_offset(keyval) {
            if offset >= self.page_size {
                return Action::Handled;
            }
            let idx = self.page * self.page_size + offset;
            return self
                .candidates
                .get(idx)
                .cloned()
                .map_or(Action::Handled, |cand| self.take_candidate(cand, source));
        }

        match keyval {
            // 空格：提交光标处候选；回车：把拼音原文当英文直接提交（不上屏候选、
            // 不产生学习数据）。两者都会清空组合。
            // 空格：选中光标处候选（部分候选 → 确认这一段，继续组合）；
            // 回车：把未确认的拼音原文当英文提交（已确认的汉字仍在前面）。
            keysym::SPACE => {
                if let Some(cand) = self.selected().cloned() {
                    self.take_candidate(cand, source)
                } else {
                    let buf = self.buffer.clone();
                    self.commit_with(&buf, Vec::new())
                }
            }
            keysym::RETURN => {
                let buf = self.buffer.clone();
                self.commit_with(&buf, Vec::new())
            }

            // 退格
            keysym::BACKSPACE => {
                if self.buffer.is_empty() {
                    // 未确认部分已空：撤销上一次「部分确认」，那段拼音退回来重选
                    self.unconfirm_last(source);
                } else {
                    self.buffer.pop();
                    if self.buffer.is_empty() && self.confirmed.is_empty() {
                        self.clear();
                    } else {
                        self.refresh(source);
                    }
                }
                Action::Handled
            }

            // Esc：取消
            keysym::ESCAPE => {
                self.clear();
                Action::Handled
            }

            // 方向键：在候选间移动光标（跨页自动翻页）。
            // 候选窗默认横排，所以左右与上下都映射到「上一个/下一个候选」，
            // 与 Rime/搜狗一致；翻页交给 Page_Up/Page_Down 与 -/= 。
            keysym::LEFT | keysym::UP => {
                self.cursor_up();
                Action::Handled
            }
            keysym::RIGHT | keysym::DOWN => {
                self.cursor_down();
                Action::Handled
            }
            keysym::PAGE_UP => {
                self.page_up();
                Action::Handled
            }
            keysym::PAGE_DOWN => {
                self.page_down();
                Action::Handled
            }

            _ => Action::Forward,
        }
    }

    /// 标点键：未组合直接上屏中文标点；组合中先提交候选再附带标点。
    fn handle_punctuation(&mut self, keyval: u32, half_width: bool) -> Option<Action> {
        let punct = punct_of(keyval, &mut self.quote_open, half_width)?;
        if !self.is_composing() {
            self.clear();
            return Some(Action::Commit {
                text: punct,
                learned: Vec::new(),
            });
        }
        // 组合中：已确认段 + 当前选中候选 + 标点，一次上屏
        let buf = self.buffer.clone();
        let (mut text, learned) = self
            .selected()
            .cloned()
            .map_or_else(|| (buf, Vec::new()), |c| (c.text, c.learned));
        text.push_str(&punct);
        Some(self.commit_with(&text, learned))
    }

    /// 上一页。
    pub const fn page_up(&mut self) {
        if self.page > 0 {
            self.page -= 1;
            self.cursor = 0; // 翻页后光标落到新页首，避免指向上一页的位置
        }
    }

    /// 下一页。
    pub const fn page_down(&mut self) {
        let max_page = self.candidates.len().saturating_sub(1) / self.page_size;
        if self.page < max_page {
            self.page += 1;
            self.cursor = 0;
        }
    }

    /// 光标上移（含跨页）。
    pub const fn cursor_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        } else if self.page > 0 {
            self.page -= 1;
            self.cursor = self.page_size - 1;
        }
    }

    /// 光标下移（含跨页）。
    pub const fn cursor_down(&mut self) {
        let total = self.candidates.len();
        let cur = self.page * self.page_size + self.cursor;
        if cur + 1 < total {
            self.cursor += 1;
            if self.cursor >= self.page_size {
                self.cursor = 0;
                self.page += 1;
            }
        }
    }

    /// 当前页的候选（文本列表，供 `IBusLookupTable`）。
    #[must_use]
    pub fn page_candidates(&self) -> Vec<String> {
        let start = self.page * self.page_size;
        self.candidates
            .iter()
            .skip(start)
            .take(self.page_size)
            .map(|c| c.text.clone())
            .collect()
    }

    /// 光标在当前页内的位置（供 `IBusLookupTable.cursor_pos`）。
    ///
    /// # Panics
    /// 仅在光标超出 `u32` 范围时（不可能，受页大小限制）。
    #[must_use]
    pub fn cursor_in_page(&self) -> u32 {
        u32::try_from(self.cursor).expect("cursor < page size")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用候选源：按拼音返回单字/词候选（learned = 单个 (拼音, 词) 对）。
    struct TestSource {
        map: std::collections::HashMap<&'static str, Vec<&'static str>>,
    }

    impl CandidateSource for TestSource {
        fn candidates(&self, pinyin: &str) -> Vec<Candidate> {
            self.map
                .get(pinyin)
                .map(|v| {
                    v.iter()
                        .map(|w| {
                            Candidate::whole(
                                *w,
                                vec![LearnedWord::new(pinyin.to_string(), w.to_string())],
                                pinyin.len(),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    /// 页内序号 → 选择键（0..=8 → `1`-`9`，9 → `0`）。
    fn select_key(offset: usize) -> u32 {
        if offset == 9 {
            keysym::DIGIT0
        } else {
            keysym::DIGIT1 + u32::try_from(offset).expect("页内序号 < 10")
        }
    }

    fn source() -> TestSource {
        TestSource {
            map: std::collections::HashMap::from([
                ("ni", vec!["你", "尼", "泥"]),
                ("shi", vec!["是", "时", "事", "十"]),
                ("nihao", vec!["你好"]),
            ]),
        }
    }

    #[test]
    fn letter_starts_composition() {
        let mut st = EngineState::new();
        let d = source();
        assert!(matches!(st.handle_key(0x6e, &d, false), Action::Handled)); // n
        assert_eq!(st.buffer(), "n");
        assert!(matches!(st.handle_key(0x69, &d, false), Action::Handled)); // i
        assert_eq!(st.buffer(), "ni");
        assert!(st.candidates().iter().any(|c| c.text == "你"));
    }

    #[test]
    fn apostrophe_is_syllable_separator_while_composing() {
        let src = TestSource {
            map: std::collections::HashMap::default(),
        };
        let mut st = EngineState::default();
        st.handle_key(keysym::A, &src, false);
        // 组合中 `'` = 分隔符：进 buffer，不触发标点（不提交、不输出引号）
        assert_eq!(
            st.handle_key(keysym::APOSTROPHE, &src, false),
            Action::Handled
        );
        assert_eq!(st.buffer, "a'");
    }

    #[test]
    fn apostrophe_is_smart_quote_when_not_composing() {
        let src = TestSource {
            map: std::collections::HashMap::default(),
        };
        let mut st = EngineState::default();
        let act = st.handle_key(keysym::APOSTROPHE, &src, false);
        match act {
            Action::Commit { text, learned } => {
                assert_eq!(text, "‘");
                assert!(learned.is_empty());
            }
            other => panic!("未组合时 `'` 应输出智能引号，got {other:?}"),
        }
    }

    #[test]
    fn space_commits_first_candidate_and_learns() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false); // n
        st.handle_key(0x69, &d, false); // i
        assert_eq!(st.candidates()[0].text, "你");
        match st.handle_key(keysym::SPACE, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "你");
                assert_eq!(
                    learned,
                    vec![LearnedWord::new("ni".to_string(), "你".to_string())]
                );
            }
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn enter_commits_raw_pinyin_as_english() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false); // n
        st.handle_key(0x69, &d, false); // i
        assert_eq!(st.candidates()[0].text, "你");
        // Enter 应提交拼音原文（不上屏候选），且不产生学习数据
        match st.handle_key(keysym::RETURN, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "ni");
                assert!(learned.is_empty());
            }
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn enter_commits_partial_pinyin() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false); // n（无候选的未完成拼音）
        match st.handle_key(keysym::RETURN, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "n");
                assert!(learned.is_empty());
            }
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn empty_buffer_forwards() {
        let mut st = EngineState::new();
        let d = source();
        assert!(matches!(
            st.handle_key(keysym::SPACE, &d, false),
            Action::Forward
        ));
        assert!(matches!(st.handle_key(0x20, &d, false), Action::Forward));
    }

    #[test]
    fn digit_selects_candidate_and_learns() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x73, &d, false); // s
        st.handle_key(0x68, &d, false); // h
        st.handle_key(0x69, &d, false); // i
        // "shi" 的候选：是 时 事 十 ...
        let third = st.candidates().get(2).cloned().unwrap();
        match st.handle_key(0x33, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, third.text);
                assert_eq!(
                    learned,
                    vec![LearnedWord::new("shi".to_string(), "事".to_string())]
                );
            }
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn digits_one_to_ten_select_their_candidate() {
        // page_size=10 时，页内 10 个候选必须个个能用数字键选到：
        // `1`-`9` 对应 #1-#9，`0` 对应 #10（候选窗里的标号就是 0）。
        let words = vec![
            "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一",
        ];
        let d = TestSource {
            map: std::collections::HashMap::from([("shi", words.clone())]),
        };
        for (i, want) in words.iter().take(10).enumerate() {
            let mut st = EngineState::with_page_size(10);
            for k in "shi".bytes() {
                st.handle_key(u32::from(k), &d, false);
            }
            let key = select_key(i);
            match st.handle_key(key, &d, false) {
                Action::Commit { text, learned } => {
                    assert_eq!(&text, want, "第 {} 个候选选错了", i + 1);
                    assert_eq!(
                        learned,
                        vec![LearnedWord::new("shi".to_string(), (*want).to_string())]
                    );
                }
                other => panic!("第 {} 个候选：expected commit, got {other:?}", i + 1),
            }
        }
    }

    #[test]
    fn digits_select_within_current_page() {
        // 第 2 页（page_size=10）：`1` 选 #11、`0` 选 #20 —— 序号是页内的，不是全局的
        let words: Vec<&'static str> = vec![
            "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10", "a11", "a12",
            "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        ];
        let d = TestSource {
            map: std::collections::HashMap::from([("shi", words)]),
        };
        for (offset, want) in [(0usize, "a11"), (9, "a20")] {
            let mut st = EngineState::with_page_size(10);
            for k in "shi".bytes() {
                st.handle_key(u32::from(k), &d, false);
            }
            st.handle_key(keysym::PAGE_DOWN, &d, false);
            assert_eq!(st.page(), 1);
            match st.handle_key(select_key(offset), &d, false) {
                Action::Commit { text, .. } => assert_eq!(text, want),
                other => panic!("expected commit, got {other:?}"),
            }
        }
    }

    #[test]
    fn digits_beyond_page_size_are_swallowed() {
        // page_size=8：`9`/`0` 落在本页之外 —— 既不得跳去选下一页的候选，
        // 也不得转发成数字插进正在组合的文本里
        let words = vec!["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
        let d = TestSource {
            map: std::collections::HashMap::from([("shi", words)]),
        };
        let mut st = EngineState::with_page_size(8);
        for k in "shi".bytes() {
            st.handle_key(u32::from(k), &d, false);
        }
        assert_eq!(st.handle_key(keysym::DIGIT9, &d, false), Action::Handled);
        assert_eq!(st.handle_key(keysym::DIGIT0, &d, false), Action::Handled);
        assert!(st.is_composing(), "超出本页的序号不得上屏");
        // 页内最后一个（`8`）仍然能选
        match st.handle_key(keysym::DIGIT1 + 7, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "八"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn digit_without_composition_is_forwarded() {
        // 未组合时数字键还是数字（包括 `0`）
        let mut st = EngineState::new();
        let d = source();
        assert_eq!(st.handle_key(keysym::DIGIT0, &d, false), Action::Forward);
        assert_eq!(st.handle_key(keysym::DIGIT1, &d, false), Action::Forward);
    }

    #[test]
    fn word_exact_match_first() {
        let mut st = EngineState::new();
        let d = source();
        for k in "nihao".bytes() {
            st.handle_key(u32::from(k), &d, false);
        }
        assert_eq!(st.candidates()[0].text, "你好");
    }

    #[test]
    fn commit_raw_pinyin_when_no_candidate() {
        let mut st = EngineState::new();
        let d = source();
        // "xy" 无候选
        st.handle_key(0x78, &d, false); // x
        st.handle_key(0x79, &d, false); // y
        match st.handle_key(keysym::SPACE, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "xy");
                assert!(learned.is_empty());
            }
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn punctuation_commits_chinese() {
        let mut st = EngineState::new();
        let d = source();
        match st.handle_key(0x2c, &d, false) {
            // ',' 未组合 → 直接上屏 ，
            Action::Commit { text, learned } => {
                assert_eq!(text, "，");
                assert!(learned.is_empty());
            }
            other => panic!("expected commit, got {other:?}"),
        }
        match st.handle_key(0x2e, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "。"), // '.'
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn punctuation_after_composition_appends() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false); // n
        st.handle_key(0x69, &d, false); // i → 你
        match st.handle_key(0x2c, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "你，");
                assert_eq!(
                    learned,
                    vec![LearnedWord::new("ni".to_string(), "你".to_string())]
                );
            }
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    /// 支持部分候选的测试源：`nihaoshijie` 给整句 + 「你好」这个部分候选。
    struct PartialSource;

    impl CandidateSource for PartialSource {
        fn candidates(&self, pinyin: &str) -> Vec<Candidate> {
            match pinyin {
                "nihaoshijie" => vec![
                    Candidate::whole(
                        "你好时节",
                        vec![
                            LearnedWord::new("ni", "你"),
                            LearnedWord::new("hao", "好"),
                            LearnedWord::new("shijie", "时节"),
                        ],
                        pinyin.len(),
                    ),
                    Candidate::partial(
                        "你好",
                        vec![LearnedWord::new("nihao", "你好")],
                        "nihao".len(),
                    ),
                ],
                "shijie" => vec![Candidate::whole(
                    "世界",
                    vec![LearnedWord::new("shijie", "世界")],
                    pinyin.len(),
                )],
                _ => Vec::new(),
            }
        }
    }

    /// 逐字母输入一串拼音。
    fn type_pinyin(st: &mut EngineState, src: &dyn CandidateSource, s: &str) {
        for c in s.chars() {
            st.handle_key(c as u32, src, false);
        }
    }

    #[test]
    fn partial_candidate_confirms_prefix_and_keeps_composing() {
        // Rime 式增量确认：整句候选不对时，选「你好」只确认这一段，
        // 剩下的 shijie 继续组合，再选「世界」才一次上屏。
        let mut st = EngineState::new();
        let src = PartialSource;
        type_pinyin(&mut st, &src, "nihaoshijie");

        // 选第 2 个候选（部分候选 你好）：不该上屏，而是确认 + 继续组合
        assert_eq!(
            st.handle_key(keysym::DIGIT1 + 1, &src, false),
            Action::Handled
        );
        assert_eq!(st.confirmed_text(), "你好");
        assert_eq!(st.buffer(), "shijie", "已确认的拼音要从未确认串里去掉");
        assert!(st.is_composing());
        assert_eq!(
            st.candidates().first().map(|c| c.text.as_str()),
            Some("世界"),
            "剩余拼音应重新解码"
        );

        // 再选「世界」：整段覆盖 → 连同已确认部分一次上屏，学习段也要拼齐
        match st.handle_key(keysym::DIGIT1, &src, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "你好世界");
                assert_eq!(
                    learned,
                    vec![
                        LearnedWord::new("nihao", "你好"),
                        LearnedWord::new("shijie", "世界"),
                    ],
                    "已确认段 + 本次选中，一起交给调频/新词学习"
                );
            }
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn backspace_undoes_partial_confirmation() {
        // 未确认部分删空后，退格撤销上一次「部分确认」，那段拼音退回来重选
        let mut st = EngineState::new();
        let src = PartialSource;
        type_pinyin(&mut st, &src, "nihaoshijie");
        st.handle_key(keysym::DIGIT1 + 1, &src, false); // 确认 你好
        for _ in 0..6 {
            st.handle_key(keysym::BACKSPACE, &src, false); // 删掉 shijie
        }
        assert_eq!(st.buffer(), "");
        assert_eq!(st.confirmed_text(), "你好");

        st.handle_key(keysym::BACKSPACE, &src, false); // 再退一次：撤销确认
        assert_eq!(st.confirmed_text(), "");
        assert_eq!(st.buffer(), "nihao", "拼音要退回未确认串");
        assert!(st.is_composing());
    }

    #[test]
    fn escape_drops_confirmed_prefix_too() {
        let mut st = EngineState::new();
        let src = PartialSource;
        type_pinyin(&mut st, &src, "nihaoshijie");
        st.handle_key(keysym::DIGIT1 + 1, &src, false);
        assert_eq!(st.handle_key(keysym::ESCAPE, &src, false), Action::Handled);
        assert!(!st.is_composing());
        assert_eq!(st.confirmed_text(), "");
        assert_eq!(st.buffer(), "");
    }

    #[test]
    fn punctuation_commits_confirmed_prefix() {
        // 组合中打标点：已确认段 + 当前候选 + 标点，一次上屏
        let mut st = EngineState::new();
        let src = PartialSource;
        type_pinyin(&mut st, &src, "nihaoshijie");
        st.handle_key(keysym::DIGIT1 + 1, &src, false); // 确认 你好，剩 shijie
        match st.handle_key(0x2c, &src, false) {
            Action::Commit { text, .. } => assert_eq!(text, "你好世界，"),
            other => panic!("expected commit, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn half_width_punctuation_after_digit() {
        let mut st = EngineState::new();
        let d = source();
        // 前一个字符是数字 → 半角标点
        match st.handle_key(0x2c, &d, true) {
            Action::Commit { text, .. } => assert_eq!(text, ","),
            other => panic!("expected commit, got {other:?}"),
        }
        // 无数字上下文 → 全角
        match st.handle_key(0x2c, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "，"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn half_width_quotes_after_latin() {
        let mut st = EngineState::new();
        let d = source();
        // 英文/数字后的引号要直引号（写代码），且不扰动开合状态
        for _ in 0..2 {
            match st.handle_key(0x22, &d, true) {
                Action::Commit { text, .. } => assert_eq!(text, "\""),
                other => panic!("expected commit, got {other:?}"),
            }
        }
        // 中文上下文仍然是智能弯引号，从开引号开始
        match st.handle_key(0x22, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "“"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn latin_context_uses_char_offset_not_bytes() {
        // 回归：cursor 是字符偏移（IBus 语义）。曾经当字节下标用，
        // “abc你好世界” 的光标（字符 6）会取到字节 5（“你” 的尾字节），
        // 而光标 4 会取到字节 3 = 'c' → 中文后面误判成半角。
        let text = "abc你好世界";
        assert!(!latin_before_cursor(text, 4)); // 前一字符 = 你
        assert!(!latin_before_cursor(text, 7)); // 末尾，前一字符 = 界
        assert!(latin_before_cursor(text, 3)); // 前一字符 = c
        // 边界：行首、空文本、过期（越界）光标 → 保守取全角
        assert!(!latin_before_cursor(text, 0));
        assert!(!latin_before_cursor("", 0));
        assert!(!latin_before_cursor(text, 999));
        // 半角标点后再打标点 → 回到全角（“只有紧跟英文/数字的第一个标点半角”）
        assert!(!latin_before_cursor("3,", 2));
        assert!(latin_before_cursor("3", 1));
        // 空格不算 latin 上下文（“abc ” + 标点 → 全角）
        assert!(!latin_before_cursor("abc ", 4));
    }

    #[test]
    fn smart_quotes_alternate() {
        let mut st = EngineState::new();
        let d = source();
        match st.handle_key(0x22, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "“"),
            other => panic!("expected commit, got {other:?}"),
        }
        match st.handle_key(0x22, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "”"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn arrow_keys_move_selection_within_candidates() {
        // 方向键在候选间移动（不是翻页）：移动后空格提交的是光标处的候选
        let mut st = EngineState::with_page_size(3);
        let d = TestSource {
            map: std::collections::HashMap::from([("shi", vec!["是", "时", "事", "使", "市"])]),
        };
        st.handle_key(0x73, &d, false); // s
        st.handle_key(0x68, &d, false); // h
        st.handle_key(0x69, &d, false); // i
        assert_eq!(st.cursor_abs(), 0);

        // 右/下：下一个候选
        st.handle_key(keysym::RIGHT, &d, false);
        assert_eq!(st.cursor_abs(), 1);
        st.handle_key(keysym::DOWN, &d, false);
        assert_eq!(st.cursor_abs(), 2);
        // 跨页：page_size=3，第 4 个候选在第 2 页
        st.handle_key(keysym::DOWN, &d, false);
        assert_eq!((st.page(), st.cursor_abs()), (1, 3));
        // 左/上：退回上一页末尾
        st.handle_key(keysym::LEFT, &d, false);
        assert_eq!((st.page(), st.cursor_abs()), (0, 2));
        // 末尾不越界
        for _ in 0..10 {
            st.handle_key(keysym::RIGHT, &d, false);
        }
        assert_eq!(st.cursor_abs(), 4, "不得越过最后一个候选");
        // 开头不越界
        for _ in 0..10 {
            st.handle_key(keysym::LEFT, &d, false);
        }
        assert_eq!((st.page(), st.cursor_abs()), (0, 0));

        // 移到第 2 个候选后空格：提交的是「时」
        st.handle_key(keysym::RIGHT, &d, false);
        match st.handle_key(keysym::SPACE, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "时"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn preview_only_while_browsing_candidates() {
        // 光标在 #1：不预览（预编辑保持分节拼音，看得见自己打了什么）
        let mut st = EngineState::new();
        let src = PartialSource;
        type_pinyin(&mut st, &src, "nihaoshijie");
        assert_eq!(st.preview(), None, "光标在首位说明还没挑，不该预览");

        // 移到 #2（部分候选 你好）：预览它，并告诉上层它只覆盖了 nihao
        st.handle_key(keysym::RIGHT, &src, false);
        assert_eq!(st.preview(), Some(("你好", "nihao".len())));

        // 移回 #1：回到拼音显示（无状态判定，可预测）
        st.handle_key(keysym::LEFT, &src, false);
        assert_eq!(st.preview(), None);

        // 确认部分候选后，剩余组合的光标又回到首位 → 不预览
        st.handle_key(keysym::RIGHT, &src, false);
        st.handle_key(keysym::SPACE, &src, false);
        assert_eq!(st.buffer(), "shijie");
        assert_eq!(st.preview(), None);
    }

    #[test]
    fn paging_keys_reset_cursor_into_page() {
        // Page_Down/Page_Up 仍然是翻页，且光标要落在新页内
        let mut st = EngineState::with_page_size(2);
        let d = TestSource {
            map: std::collections::HashMap::from([("shi", vec!["是", "时", "事", "使"])]),
        };
        st.handle_key(0x73, &d, false);
        st.handle_key(0x68, &d, false);
        st.handle_key(0x69, &d, false);
        st.handle_key(keysym::RIGHT, &d, false); // 光标到页内第 2 个
        st.handle_key(keysym::PAGE_DOWN, &d, false);
        assert_eq!((st.page(), st.cursor_abs()), (1, 2), "翻页后光标落到新页首");
        st.handle_key(keysym::PAGE_UP, &d, false);
        assert_eq!((st.page(), st.cursor_abs()), (0, 0));
    }

    #[test]
    fn minus_equal_page_through_candidates() {
        let mut st = EngineState::with_page_size(3);
        let d = source();
        for k in "shi".bytes() {
            st.handle_key(u32::from(k), &d, false);
        }
        // shi 有 4 个候选，page_size 3 → 2 页
        assert_eq!(st.page(), 0);
        assert_eq!(st.handle_key(keysym::EQUAL, &d, false), Action::Handled); // = 下一页
        assert_eq!(st.page(), 1);
        assert_eq!(st.handle_key(keysym::MINUS, &d, false), Action::Handled); // - 上一页
        assert_eq!(st.page(), 0);
        // 未组合时 - / = 是标点
        st.clear();
        match st.handle_key(keysym::MINUS, &d, false) {
            Action::Commit { text, .. } => assert_eq!(text, "－"),
            other => panic!("expected commit, got {other:?}"),
        }
    }

    #[test]
    fn page_size_is_configurable() {
        let mut st = EngineState::with_page_size(3);
        let d = source();
        for k in "shi".bytes() {
            st.handle_key(u32::from(k), &d, false);
        }
        assert_eq!(st.page_candidates().len(), 3); // 每页 3 个
    }

    #[test]
    fn mode_toggle_switches() {
        let mut st = EngineState::new();
        assert_eq!(st.mode(), InputMode::Chinese);
        assert!(st.toggle_mode().is_none());
        assert_eq!(st.mode(), InputMode::English);
        assert!(st.toggle_mode().is_none());
        assert_eq!(st.mode(), InputMode::Chinese);
    }

    #[test]
    fn english_mode_forwards_everything() {
        let mut st = EngineState::new();
        let d = source();
        st.toggle_mode(); // → English
        // 字母/空格/标点全部转发
        assert!(matches!(st.handle_key(0x6e, &d, false), Action::Forward)); // n
        assert!(matches!(
            st.handle_key(keysym::SPACE, &d, false),
            Action::Forward
        ));
        assert!(matches!(st.handle_key(0x2c, &d, false), Action::Forward)); // ,
        assert!(!st.is_composing());
    }

    #[test]
    fn shifted_letter_forwards_when_idle() {
        let mut st = EngineState::new();
        let d = source();
        // Shift+n（大写 keyval 0x4e），未组合 → 转发给应用（输出 N）
        assert!(matches!(st.handle_key(0x4e, &d, false), Action::Forward));
    }

    #[test]
    fn shifted_letter_commits_and_forwards_when_composing() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false);
        st.handle_key(0x69, &d, false); // ni → 你
        match st.handle_key(0x4e, &d, false) {
            // Shift+n：提交预编辑 你，转发 N
            Action::CommitAndForward { text, learned } => {
                assert_eq!(text, "你");
                assert_eq!(
                    learned,
                    vec![LearnedWord::new("ni".to_string(), "你".to_string())]
                );
            }
            other => panic!("expected CommitAndForward, got {other:?}"),
        }
        assert!(!st.is_composing());
    }

    #[test]
    fn toggle_mode_commits_preedit() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false);
        st.handle_key(0x69, &d, false); // ni → 你
        let commit = st.toggle_mode().unwrap();
        assert_eq!(commit.0, "你");
        assert_eq!(
            commit.1,
            vec![LearnedWord::new("ni".to_string(), "你".to_string())]
        );
        assert!(!st.is_composing());
        assert_eq!(st.mode(), InputMode::English);
    }

    #[test]
    fn mode_survives_clear() {
        let mut st = EngineState::new();
        st.toggle_mode(); // → English
        st.clear();
        assert_eq!(st.mode(), InputMode::English);
    }

    #[test]
    fn backspace_and_escape() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false);
        st.handle_key(0x69, &d, false);
        st.handle_key(keysym::BACKSPACE, &d, false);
        assert_eq!(st.buffer(), "n");
        st.handle_key(0x69, &d, false);
        st.handle_key(keysym::ESCAPE, &d, false);
        assert!(!st.is_composing());
    }
}
