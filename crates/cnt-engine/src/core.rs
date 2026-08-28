//! `EngineCore` —— 一个输入上下文的**共享内核**：状态机 + UI 发送 + 上屏。
//!
//! 为什么要把它从 `Engine` 里拆出来：语音输入是**异步、非按键驱动**的事件源
//! （识别在另一个线程，几百毫秒后才有结果），它也要能改预编辑、能上屏。
//! 而 zbus 的接口对象由 object server 持有，拿不到 `Arc<Engine>`。
//!
//! 于是把「所有需要被两个方向共享的东西」放进 `Arc<EngineCore>`：
//!
//! ```text
//!   按键（zbus 方法调用）─┐
//!                        ├─► Arc<EngineCore> ─► UpdatePreeditText / CommitText
//!   语音事件（tokio 任务）┘
//! ```
//!
//! `Engine` 自己只留「按键专属」的状态（Shift 单击检测、语音热键运行时）。

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use zbus::connection::Connection;

use cnt_decode::Decoder;
use cnt_input::{EngineState, LearnedWord, latin_before_cursor};

/// `IBus` 引擎接口名。
pub(crate) const ENGINE_IFACE: &str = "org.freedesktop.IBus.Engine";

/// 界面状态快照（所有数据均为 owned，可安全跨 await）。
pub(crate) struct UiState {
    /// 预编辑文本：语音提示 + 已确认的汉字 + 分节显示的未确认拼音（`你好 shi jie`）。
    pub preedit: String,
    pub all_cands: Vec<String>,
    /// 光标在全部候选中的绝对位置（面板据此计算当前页）。
    pub cursor_abs: u32,
}

/// 一个输入上下文的共享内核。
pub struct EngineCore {
    conn: Connection,
    path: String,
    page_size: usize,
    state: Mutex<EngineState>,
    decoder: Arc<Decoder>,
    /// 应用提供的 surrounding text（文本, 光标**字符**偏移）；应用不支持时为 None。
    surrounding: Mutex<Option<(String, usize)>>,
    /// 我们自己刚上屏的最后一个字符（比 surrounding 更新）。
    ///
    /// 很多应用不会在每次 `CommitText` 后重发 surrounding text，导致缓存过期：
    /// 先打英文再打中文，前一字符会一直停在那个英文字母上 → 标点永远半角。
    /// 所以以自己的上屏为权威，应用下次发 surrounding 时再交回去。
    last_commit_char: Mutex<Option<char>>,
    /// 语音会话的状态提示（`🎤 说话中…`）；None = 没在录音。
    voice_hint: Mutex<Option<String>>,
}

impl EngineCore {
    /// 新建内核。
    pub(crate) const fn new(
        conn: Connection,
        path: String,
        decoder: Arc<Decoder>,
        page_size: usize,
    ) -> Self {
        Self {
            conn,
            path,
            page_size,
            state: Mutex::new(EngineState::with_page_size(page_size)),
            decoder,
            surrounding: Mutex::new(None),
            last_commit_char: Mutex::new(None),
            voice_hint: Mutex::new(None),
        }
    }

    /// 解码器（候选来源 + 用户学习）。
    pub(crate) const fn decoder(&self) -> &Arc<Decoder> {
        &self.decoder
    }

    /// 锁定并取组合状态（毒锁恢复，内部状态永不让锁失败 panic）。
    pub(crate) fn lock_state(&self) -> MutexGuard<'_, EngineState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// 当前语音提示。
    fn hint(&self) -> Option<String> {
        self.voice_hint
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// 设置语音提示并立即刷新界面（None = 清除）。
    pub(crate) async fn set_voice_hint(&self, hint: Option<String>) {
        *self
            .voice_hint
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = hint;
        let ui = self.snapshot();
        if ui.preedit.is_empty() && ui.all_cands.is_empty() {
            self.hide_ui().await;
        } else {
            self.update_ui(&ui).await;
        }
    }

    /// 取当前状态快照（不持锁跨 await）。
    pub(crate) fn snapshot(&self) -> UiState {
        let st = self.lock_state();
        let mut preedit = String::new();
        if let Some(hint) = self.hint() {
            preedit.push_str(&hint);
        }
        let body = render_preedit(st.confirmed_text(), st.preview(), st.buffer(), |py| {
            self.decoder.display_pinyin(py)
        });
        if !body.is_empty() {
            if !preedit.is_empty() {
                preedit.push(' ');
            }
            preedit.push_str(&body);
        }
        UiState {
            preedit,
            all_cands: st.candidates().iter().map(|c| c.text.clone()).collect(),
            cursor_abs: u32::try_from(st.cursor_abs()).expect("cursor fits u32"),
        }
    }

    /// 更新候选窗口与预编辑文本（上屏前的拼音）。
    pub(crate) async fn update_ui(&self, ui: &UiState) {
        // 预编辑文本（拼音缓冲区 / 语音提示）
        let (preedit, cursor, visible) = if ui.preedit.is_empty() {
            (String::new(), 0u32, false)
        } else {
            (
                ui.preedit.clone(),
                u32::try_from(ui.preedit.chars().count()).expect("preedit fits u32"),
                true,
            )
        };
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "UpdatePreeditText",
                &(cnt_ibus::text(&preedit), cursor, visible, 1u32), // mode=1 下划线
            )
            .await;

        // 候选列表：发送全部候选 + 绝对光标位置。
        // IBus 面板按 `cursor / page_size * page_size` 计算当前页，
        // 候选数多于每页时自动显示翻页按钮（点击 → 引擎的 PageDown/PageUp）。
        let visible = !ui.all_cands.is_empty();
        let table = cnt_ibus::lookup_table(
            &ui.all_cands,
            u32::try_from(self.page_size).expect("page size fits u32"),
            ui.cursor_abs,
            visible,
        );
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "UpdateLookupTable",
                &(table, visible),
            )
            .await;
    }

    /// 隐藏预编辑文本与候选窗口。
    pub(crate) async fn hide_ui(&self) {
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "UpdatePreeditText",
                &(cnt_ibus::text(""), 0u32, false, 0u32),
            )
            .await;
        let table = cnt_ibus::lookup_table(
            &[],
            u32::try_from(self.page_size).expect("page size fits u32"),
            0,
            false,
        );
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "UpdateLookupTable",
                &(table, false),
            )
            .await;
    }

    /// 提交一段文字并清空状态。
    pub(crate) async fn commit(&self, text: &str) {
        // 记住自己上屏的最后一个字符：下一个标点的宽度判定以此为准，
        // 不再依赖应用是否及时重发 surrounding text。
        *self
            .last_commit_char
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = text.chars().last();
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "CommitText",
                &(cnt_ibus::text(text),),
            )
            .await;
        self.lock_state().clear();
        self.hide_ui().await;
    }

    /// 语音识别结果上屏：上屏后把语音提示重新画回去（会话还在继续）。
    pub(crate) async fn commit_voice(&self, text: &str) {
        self.commit(text).await;
        if self.hint().is_some() {
            let ui = self.snapshot();
            self.update_ui(&ui).await;
        }
    }

    /// 丢弃半角判定的上下文（焦点切换/重置：旧位置的前一字符已无意义）。
    pub(crate) fn forget_context(&self) {
        *self
            .last_commit_char
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
        *self
            .surrounding
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// 转发给应用的可打印 ASCII 键也要计入上下文（数字、英文模式的字母都走转发，
    /// 应用未必重发 surrounding）。这正是「3, / abc,」要半角的那一类情形。
    pub(crate) fn note_forwarded_key(&self, keyval: u32) {
        if let Some(c) = char::from_u32(keyval).filter(char::is_ascii_graphic) {
            *self
                .last_commit_char
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(c);
        }
    }

    /// 记录应用下发的 surrounding text。
    pub(crate) fn set_surrounding(&self, text: String, cursor: usize) {
        // 应用的数据是新鲜的（包含光标移动/删除等我们看不到的编辑），
        // 一旦收到就不再用自己缓的上屏字符。
        *self
            .last_commit_char
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = None;
        *self
            .surrounding
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some((text, cursor));
    }

    /// 标点半角判定：自己刚上屏的字符优先，否则看应用给的 surrounding text。
    ///
    /// 两边都没信息时返回 false（全角）——中文输入法的默认应当是全角。
    pub(crate) fn punct_half_width(&self) -> bool {
        let last = *self
            .last_commit_char
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(c) = last {
            return c.is_ascii_alphanumeric();
        }
        self.surrounding
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .is_some_and(|(text, cursor)| latin_before_cursor(text, *cursor))
    }

    /// 请应用下发光标周围文本（不发这个信号，大多数应用根本不会调 `SetSurroundingText`）。
    pub(crate) async fn require_surrounding_text(&self) {
        let _ = self
            .conn
            .emit_signal(
                None::<&str>,
                self.path.as_str(),
                ENGINE_IFACE,
                "RequireSurroundingText",
                &(),
            )
            .await;
    }

    /// 刷新界面（按键处理完 / 语音提示变化）。
    pub(crate) async fn refresh_after_handled(&self) {
        let ui = self.snapshot();
        self.update_ui(&ui).await;
    }

    /// 忘记一组学习段（Ctrl+Delete：撤销误学），随后刷新候选窗。
    pub(crate) async fn forget_candidate(&self, learned: Vec<LearnedWord>) {
        if !learned.is_empty() {
            log::debug!("forget: {} segments", learned.len());
            self.decoder.forget(&learned);
        }
        self.refresh_after_handled().await;
    }

    /// 把用户学习数据写盘（`focus_out`/`disable` 时立即落盘，减少丢失窗口）。
    pub(crate) fn flush_user(&self) {
        if let Err(e) = self.decoder.flush_user() {
            log::error!("flush user data failed: {e}");
        }
    }

    /// 提交一段文本并记录学习数据。
    pub(crate) async fn handle_commit(&self, text: String, learned: Vec<LearnedWord>) {
        // 用户通过候选上屏 → 调频 + 新词学习（相邻段拼合成词）
        if !learned.is_empty() {
            log::debug!("learn: {} segments", learned.len());
            self.decoder.learn(&learned);
        }
        self.commit(&text).await;
    }

    /// 若正在组合，把预编辑先上屏（中英切换 / 开始语音输入前调用）。
    pub(crate) async fn commit_composing(&self) {
        let pending = self.lock_state().take_composing();
        if let Some((text, learned)) = pending {
            log::debug!("commit pending composition: {text}");
            if !learned.is_empty() {
                self.decoder.learn(&learned);
            }
            self.commit(&text).await;
        }
    }

    /// Shift 单击切换中英模式；若切换前在组合中，先把预编辑上屏。
    pub(crate) async fn toggle_input_mode(&self) {
        let commit = self.lock_state().toggle_mode();
        if let Some((text, learned)) = commit {
            log::debug!("mode switch commits preedit: {text}");
            if !learned.is_empty() {
                self.decoder.learn(&learned);
            }
            self.commit(&text).await;
        }
        log::info!("input mode toggled");
    }
}

/// 拼出预编辑文本：已确认的汉字 + （浏览候选时的内联预览）+ 未覆盖的分节拼音。
///
/// - 不浏览候选时是 `你好 shi jie` 形态：用户看得见自己打了什么、怎么切分的；
/// - 浏览候选时把选中候选内联进来（整句 → `你好世界`；部分候选 → `你好 shi jie`），
///   空格确认前就能看到结果；
/// - 汉字与拼音之间留一个空格，界限清楚。
///
/// `segment` 是拼音分节函数（由解码器的音节表提供）。纯函数，便于测试边界。
pub(crate) fn render_preedit(
    confirmed: &str,
    preview: Option<(&str, usize)>,
    buffer: &str,
    segment: impl Fn(&str) -> String,
) -> String {
    let mut out = String::with_capacity(confirmed.len() + buffer.len() + 8);
    out.push_str(confirmed);
    let tail = match preview {
        Some((text, consumed)) => {
            out.push_str(text);
            // consumed 来自候选自报的覆盖长度，越界时按全覆盖处理（不 panic）
            buffer.get(consumed.min(buffer.len())..).unwrap_or("")
        }
        None => buffer,
    };
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&segment(tail));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::render_preedit;

    /// 测试用分节函数：按 3 字母一节切（不依赖真实音节表）
    fn seg(py: &str) -> String {
        py.as_bytes()
            .chunks(3)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn preedit_shows_segmented_pinyin_by_default() {
        assert_eq!(render_preedit("", None, "nihaoshi", seg), "nih aos hi");
        assert_eq!(render_preedit("", None, "", seg), "");
    }

    #[test]
    fn preedit_keeps_confirmed_prefix() {
        // 已确认的汉字 + 剩余拼音，中间一个空格
        assert_eq!(render_preedit("你好", None, "shijie", seg), "你好 shi jie");
    }

    #[test]
    fn preedit_inlines_preview_while_browsing() {
        // 整句候选：全覆盖 → 只剩汉字
        assert_eq!(
            render_preedit("", Some(("你好世界", 11)), "nihaoshijie", seg),
            "你好世界"
        );
        // 部分候选：覆盖 nihao，剩余仍是拼音
        assert_eq!(
            render_preedit("", Some(("你好", 5)), "nihaoshijie", seg),
            "你好 shi jie"
        );
        // 已确认段 + 预览 + 剩余
        assert_eq!(
            render_preedit("我说", Some(("你好", 5)), "nihaoshijie", seg),
            "我说你好 shi jie"
        );
    }

    #[test]
    fn preedit_tolerates_out_of_range_coverage() {
        // 覆盖长度越界（不该发生）也不 panic，按全覆盖处理
        assert_eq!(render_preedit("", Some(("你好", 99)), "nihao", seg), "你好");
    }
}
