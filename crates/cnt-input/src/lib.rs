//! 拼音输入逻辑：维护输入缓冲区与候选列表，处理按键。
//!
//! 纯逻辑、无 IO：候选通过 `CandidateSource` 抽象注入；
//! 每个候选携带可学习信息（句子 = 多个 (拼音, 词) 对），
//! 上层在提交时据此记录用户调频。

/// 默认每页候选数（可用配置覆盖）
pub const PAGE_SIZE: usize = 10;

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

/// 一个候选（词或整句）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub text: String,
    /// 用于用户调频学习的拼音-词段：单字/词 = 1 段，整句 = 多段。
    pub learned: Vec<LearnedWord>,
}

/// 候选来源：输入逻辑查询候选的统一接口（由 cnt-decode 的整句解码器实现）。
pub trait CandidateSource {
    fn candidates(&self, pinyin: &str) -> Vec<Candidate>;
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
            let p = if *quote_open { "”" } else { "“" };
            *quote_open = !*quote_open;
            return Some(p.to_string());
        }
        '\'' => {
            let p = if *quote_open { "’" } else { "‘" };
            *quote_open = !*quote_open;
            return Some(p.to_string());
        }
        _ => return None,
    };
    Some(if half_width { pair.1 } else { pair.0 }.to_string())
}

/// X11 keysym（IBus 的 keyval 即 keysym）
pub mod keysym {
    pub const SPACE: u32 = 0x0020;
    pub const BACKSPACE: u32 = 0xff08;
    pub const RETURN: u32 = 0xff0d;
    pub const ESCAPE: u32 = 0xff1b;
    pub const UP: u32 = 0xff52;
    pub const DOWN: u32 = 0xff54;
    pub const PAGE_UP: u32 = 0xff55;
    pub const PAGE_DOWN: u32 = 0xff56;
    pub const A: u32 = 0x61;
    pub const Z: u32 = 0x7a;
    pub const DIGIT1: u32 = 0x31;
    pub const DIGIT9: u32 = 0x39;
    /// 翻页键（Rime 行为）：- 上一页，= 下一页
    pub const MINUS: u32 = 0x002d;
    pub const EQUAL: u32 = 0x003d;
}

/// 输入组合状态（聚合根）：字段私有，行为通过方法暴露。
pub struct EngineState {
    /// 已输入的拼音串
    buffer: String,
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
        self.candidates.clear();
        self.page = 0;
        self.cursor = 0;
        // 引号开合状态是会话级的（交替产生 “/”），不随组合清除
    }

    #[must_use]
    pub const fn is_composing(&self) -> bool {
        !self.buffer.is_empty()
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
        if self.buffer.is_empty() {
            return None;
        }
        let commit = self.selected().cloned().map_or_else(
            || (self.buffer.clone(), Vec::new()),
            |c| (c.text, c.learned),
        );
        self.clear();
        Some(commit)
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

    /// 当前光标处的候选（若有）。
    #[must_use]
    pub fn selected(&self) -> Option<&Candidate> {
        self.candidates.get(self.page * self.page_size + self.cursor)
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
            let commit = self.selected().cloned().map_or_else(
                || (self.buffer.clone(), Vec::new()),
                |c| (c.text, c.learned),
            );
            self.clear();
            return Action::CommitAndForward {
                text: commit.0,
                learned: commit.1,
            };
        }

        // 字母键：进入/继续拼音组合
        let lower = keyval;
        if (keysym::A..=keysym::Z).contains(&lower) {
            self.buffer.push(char::from_u32(lower).unwrap_or('a')); // lower 必在 a-z 范围
            self.candidates = source.candidates(&self.buffer);
            self.page = 0;
            self.cursor = 0;
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

        // 未在组合中：一律转发
        if self.buffer.is_empty() {
            return Action::Forward;
        }

        match keyval {
            // 空格：提交光标处候选；回车：把拼音原文当英文直接提交（不上屏候选、
            // 不产生学习数据）。两者都会清空组合。
            keysym::SPACE | keysym::RETURN => {
                let buf = self.buffer.clone();
                let (text, learned) = if keyval == keysym::SPACE {
                    self.selected().cloned().map_or_else(
                        || (buf, Vec::new()),
                        |c| (c.text, c.learned),
                    )
                } else {
                    (buf, Vec::new()) // 回车：拼音原文提交
                };
                self.clear();
                Action::Commit { text, learned }
            }

            // 退格
            keysym::BACKSPACE => {
                self.buffer.pop();
                if self.buffer.is_empty() {
                    self.clear();
                } else {
                    self.candidates = source.candidates(&self.buffer);
                    self.page = 0;
                    self.cursor = 0;
                }
                Action::Handled
            }

            // Esc：取消
            keysym::ESCAPE => {
                self.clear();
                Action::Handled
            }

            // 上下翻页
            keysym::UP | keysym::PAGE_UP => {
                self.page_up();
                Action::Handled
            }
            keysym::DOWN | keysym::PAGE_DOWN => {
                self.page_down();
                Action::Handled
            }

            // 1-9：选择候选
            keysym::DIGIT1..=keysym::DIGIT9 => {
                let idx = self.page * self.page_size + (keyval - keysym::DIGIT1) as usize;
                match self.candidates.get(idx).cloned() {
                    Some(cand) => {
                        self.clear();
                        Action::Commit {
                            text: cand.text,
                            learned: cand.learned,
                        }
                    }
                    None => Action::Handled,
                }
            }

            _ => Action::Forward,
        }
    }

    /// 标点键：未组合直接上屏中文标点；组合中先提交候选再附带标点。
    fn handle_punctuation(&mut self, keyval: u32, half_width: bool) -> Option<Action> {
        let punct = punct_of(keyval, &mut self.quote_open, half_width)?;
        if self.buffer.is_empty() {
            self.clear();
            return Some(Action::Commit {
                text: punct,
                learned: Vec::new(),
            });
        }
        let buf = self.buffer.clone();
        let (mut text, learned) = if let Some(cand) = self.selected().cloned() {
            (cand.text, cand.learned)
        } else {
            (buf, Vec::new())
        };
        text.push_str(&punct);
        self.clear();
        Some(Action::Commit { text, learned })
    }

    /// 上一页。
    pub const fn page_up(&mut self) {
        if self.page > 0 {
            self.page -= 1;
        }
    }

    /// 下一页。
    pub const fn page_down(&mut self) {
        let max_page = self.candidates.len().saturating_sub(1) / self.page_size;
        if self.page < max_page {
            self.page += 1;
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
                        .map(|w| Candidate {
                            text: w.to_string(),
                            learned: vec![LearnedWord::new(pinyin.to_string(), w.to_string())],
                        })
                        .collect()
                })
                .unwrap_or_default()
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
    fn space_commits_first_candidate_and_learns() {
        let mut st = EngineState::new();
        let d = source();
        st.handle_key(0x6e, &d, false); // n
        st.handle_key(0x69, &d, false); // i
        assert_eq!(st.candidates()[0].text, "你");
        match st.handle_key(keysym::SPACE, &d, false) {
            Action::Commit { text, learned } => {
                assert_eq!(text, "你");
                assert_eq!(learned, vec![LearnedWord::new("ni".to_string(), "你".to_string())]);
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
                assert_eq!(learned, vec![LearnedWord::new("shi".to_string(), "事".to_string())]);
            }
            other => panic!("expected commit, got {other:?}"),
        }
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
                assert_eq!(learned, vec![LearnedWord::new("ni".to_string(), "你".to_string())]);
            }
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
        assert!(matches!(st.handle_key(keysym::SPACE, &d, false), Action::Forward));
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
                assert_eq!(learned, vec![LearnedWord::new("ni".to_string(), "你".to_string())]);
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
        assert_eq!(commit.1, vec![LearnedWord::new("ni".to_string(), "你".to_string())]);
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
