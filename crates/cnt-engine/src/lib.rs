//! `IBus` 引擎的 D-Bus 服务端实现：
//! - `Factory`：org.freedesktop.IBus.Factory（路径 /org/freedesktop/IBus/Engine/Factory）
//! - `Engine`：org.freedesktop.IBus.Engine（路径 /org/freedesktop/IBus/Engine/N）

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use fastrace::collector::{SpanContext};
use fastrace::Span;
use zbus::connection::Connection;
use zbus::zvariant::OwnedObjectPath;

use cnt_decode::Decoder;
use cnt_input::{Action, EngineState, LearnedWord};

const ENGINE_IFACE: &str = "org.freedesktop.IBus.Engine";
const FACTORY_PATH: &str = "/org/freedesktop/IBus/Factory";

/// `IBus` 的 modifiers 位定义
const RELEASE_MASK: u32 = 1 << 30;
/// Shift 修饰键位（单独处理，用于中英切换）
const SHIFT_MASK: u32 = 1 << 0;
/// 这些组合键按下时不处理（控制键 / Alt / Super / Hyper / Meta；不含 Shift）
const IGNORED_MOD_MASK: u32 = (1 << 2) | (1 << 3) | (1 << 26) | (1 << 27) | (1 << 28);
/// Shift 键（左/右）keysym
const KEY_SHIFT_L: u32 = 0xffe1;
const KEY_SHIFT_R: u32 = 0xffe2;
/// Shift「单击」判定窗口：按下后该时间内释放且期间未打其他键 → 切换中英
const SHIFT_TAP_DURATION: std::time::Duration = std::time::Duration::from_millis(350);

/// Shift 单击检测状态（按下时间 + 是否被「借用」打过其他键）。
#[derive(Default)]
struct ShiftTap {
    pressed_at: Option<std::time::Instant>,
    used: bool,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub struct Factory {
    conn: Connection,
    counter: AtomicU32,
    decoder: Arc<Decoder>,
    page_size: usize,
}

impl Factory {
    #[must_use]
    pub const fn new(conn: Connection, decoder: Arc<Decoder>, page_size: usize) -> Self {
        Self {
            conn,
            counter: AtomicU32::new(1),
            decoder,
            page_size,
        }
    }
}

#[zbus::interface(name = "org.freedesktop.IBus.Factory")]
impl Factory {
    /// 创建一个引擎实例，返回其对象路径。
    async fn create_engine(&self, name: &str) -> zbus::fdo::Result<OwnedObjectPath> {
        log::info!("CreateEngine({name})");
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        let path = format!("/org/freedesktop/IBus/Engine/{n}");
        let engine = Engine::new(self.conn.clone(), path.clone(), self.decoder.clone(), self.page_size);
        self.conn
            .object_server()
            .at(path.as_str(), engine)
            .await
            .map_err(|e| zbus::fdo::Error::Failed(e.to_string()))?;
        OwnedObjectPath::try_from(path).map_err(|e| zbus::fdo::Error::Failed(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    conn: Connection,
    path: String,
    state: Mutex<EngineState>,
    decoder: Arc<Decoder>,
    page_size: usize,
    shift_tap: Mutex<ShiftTap>,
    /// 应用提供的 surrounding text（文本, 光标字节偏移）；应用不支持时为 None。
    surrounding: Mutex<Option<(String, i32)>>,
}

/// 界面状态快照（所有数据均为 owned，可安全跨 await）
struct UiState {
    buffer: String,
    all_cands: Vec<String>,
    /// 光标在全部候选中的绝对位置（面板据此计算当前页）
    cursor_abs: u32,
}

impl Engine {
    /// 锁定并取组合状态（毒锁恢复，内部状态永不让锁失败 panic）。
    fn lock_state(&self) -> std::sync::MutexGuard<'_, EngineState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[must_use]
    pub fn new(conn: Connection, path: String, decoder: Arc<Decoder>, page_size: usize) -> Self {
        Self {
            conn,
            path,
            state: Mutex::new(EngineState::with_page_size(page_size)),
            decoder,
            page_size,
            shift_tap: Mutex::new(ShiftTap::default()),
            surrounding: Mutex::new(None),
        }
    }

    /// 取当前状态快照（不持锁跨 await）
    fn snapshot(&self) -> UiState {
        let st = self.lock_state();
        UiState {
            buffer: st.buffer().to_string(),
            all_cands: st.candidates().iter().map(|c| c.text.clone()).collect(),
            cursor_abs: u32::try_from(st.cursor_abs()).expect("cursor fits u32"),
        }
    }

    /// 更新候选窗口与预编辑文本（上屏前的拼音）。
    async fn update_ui(&self, ui: &UiState) {
        // 预编辑文本（拼音缓冲区）
        let (preedit, cursor, visible) = if ui.buffer.is_empty() {
            (String::new(), 0u32, false)
        } else {
            (ui.buffer.clone(), u32::try_from(ui.buffer.chars().count()).expect("preedit fits u32"), true)
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
    async fn hide_ui(&self) {
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
        let table = cnt_ibus::lookup_table(&[], u32::try_from(self.page_size).expect("page size fits u32"), 0, false);
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
    async fn commit(&self, text: &str) {
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

    async fn refresh_after_handled(&self) {
        let ui = self.snapshot();
        self.update_ui(&ui).await;
    }

    /// 把用户学习数据写盘（`focus_out`/`disable` 时立即落盘，减少丢失窗口）。
    fn flush_user(&self) {
        if let Err(e) = self.decoder.flush_user() {
            log::error!("flush user data failed: {e}");
        }
    }

    /// 提交一段文本并记录学习数据。
    async fn handle_commit(&self, text: String, learned: Vec<LearnedWord>) {
        // 用户通过候选上屏 → 调频 + 新词学习（相邻段拼合成词）
        if !learned.is_empty() {
            log::debug!("learn: {} segments", learned.len());
            self.decoder.learn(&learned);
        }
        self.commit(&text).await;
    }

    /// Shift 单击切换中英模式；若切换前在组合中，先把预编辑上屏。
    async fn toggle_input_mode(&self) {
        let commit = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .toggle_mode();
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

#[zbus::interface(name = "org.freedesktop.IBus.Engine")]
// - used_underscore_binding: zbus 宏生成的分发代码会引用 `_` 前缀参数（协议签名保留）
// - unused_self / missing_const_for_fn: 空协议方法 stub，方法签名由 IBus 协议固定
#[allow(
    clippy::used_underscore_binding,
    clippy::unused_self,
    clippy::missing_const_for_fn
)]
impl Engine {
    /// 核心：处理按键事件。返回 true 表示已消费。
    ///
    /// 每次按键创建一个 root span（fastrace 推荐的短任务模式）：
    /// span 只覆盖同步处理段（状态机 + 词库查询），保证 future Send；
    /// await 的 UI 刷新是 fire-and-forget，不纳入 trace。
    async fn process_key_event(
        &self,
        keyval: u32,
        _keycode: u32,
        state: u32,
    ) -> zbus::fdo::Result<bool> {
        // Shift 键：必须先于通用 release 检查处理 ——
        // 通用检查会把所有 release 事件拦下（含 Shift 释放），导致释放分支永不执行。
        // 用 RELEASE_MASK 区分按下/释放（IBus 的 release 事件带 1<<30 位）。
        if keyval == KEY_SHIFT_L || keyval == KEY_SHIFT_R {
            let mut should_toggle = false;
            {
                let mut tap = self
                    .shift_tap
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if state & RELEASE_MASK != 0 {
                    // 释放：若为「单击」（按下后 TAP 窗口内释放、期间未打其他键）→ 切换
                    if let Some(pressed) = tap.pressed_at.take() {
                        should_toggle = !tap.used && pressed.elapsed() < SHIFT_TAP_DURATION;
                    }
                } else {
                    // 按下
                    tap.pressed_at = Some(std::time::Instant::now());
                }
                tap.used = false; // 每次 shift 事件都重置「借用」标记
            } // 释放锁
            if should_toggle {
                self.toggle_input_mode().await;
            }
            return Ok(false); // Shift 本身转发给应用（无害）
        }

        // 忽略按键释放与组合键（Ctrl/Alt/Super 等；Shift 已在上面处理）
        if state & (RELEASE_MASK | IGNORED_MOD_MASK) != 0 {
            return Ok(false);
        }

        // 带 Shift 的字符键：标记 shift 被「借用」（避免松开时误判为单击切换）
        if state & SHIFT_MASK != 0 {
            self.shift_tap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .used = true;
        }

        let action = {
            let root = Span::root("process_key_event", SpanContext::random());
            let _guard = root.set_local_parent();
            log::debug!("keyval=0x{keyval:x} state=0x{state:x}");

            let mut st = self.lock_state();
            // 半角标点上下文：未组合且前一个字符是数字/英文 → 标点用半角（3，→ 3,）
            let punct_half_width = if st.is_composing() {
                false
            } else {
                let sur = self
                    .surrounding
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                sur.as_ref().is_some_and(|(text, cursor)| {
                    usize::try_from((*cursor - 1).max(0))
                        .ok()
                        .and_then(|idx| text.as_bytes().get(idx))
                        .is_some_and(u8::is_ascii_alphanumeric)
                })
            };
            st.handle_key(keyval, &*self.decoder, punct_half_width)
        };

        match action {
            Action::Forward => {
                log::debug!("forwarded");
                Ok(false)
            }
            Action::Handled => {
                self.refresh_after_handled().await;
                Ok(true)
            }
            Action::Commit { text, learned } => {
                self.handle_commit(text, learned).await;
                Ok(true)
            }
            Action::CommitAndForward { text, learned } => {
                // 提交预编辑后把原按键转发给应用（如 Shift+字母 输出大写）
                self.handle_commit(text, learned).await;
                Ok(false)
            }
        }
    }

    fn focus_in(&self) {}

    fn focus_in_id(&self, _object_path: &str, _client: &str) {}

    async fn focus_out(&self) {
        self.lock_state().clear();
        self.hide_ui().await;
        self.flush_user();
    }

    async fn focus_out_id(&self, _object_path: &str) {
        self.lock_state().clear();
        self.hide_ui().await;
        self.flush_user();
    }

    async fn reset(&self) {
        self.lock_state().clear();
        self.hide_ui().await;
    }

    fn enable(&self) {}

    async fn disable(&self) {
        self.lock_state().clear();
        self.hide_ui().await;
        self.flush_user();
    }

    fn set_capabilities(&self, _caps: u32) {}

    fn set_cursor_location(&self, _x: i32, _y: i32, _w: i32, _h: i32) {}

    fn property_activate(&self, _name: &str, _state: u32) {}

    fn property_show(&self, _name: &str) {}

    fn property_hide(&self, _name: &str) {}

    /// 鼠标点击候选（index 为当前页内从 0 开始的下标）
    async fn candidate_clicked(&self, index: u32, _button: u32, _state: u32) {
        let action = {
            let mut st = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let idx = st.page_size() * st.page() + usize::try_from(index).expect("index fits usize");
            st.candidate_at(idx).map_or(Action::Handled, |cand| {
                st.clear();
                Action::Commit {
                    text: cand.text,
                    learned: cand.learned,
                }
            })
        };
        if let Action::Commit { text, learned } = action {
            if !learned.is_empty() {
                self.decoder.learn(&learned);
            }
            self.commit(&text).await;
        }
    }

    async fn page_up(&self) {
        self.lock_state().page_up();
        self.refresh_after_handled().await;
    }

    async fn page_down(&self) {
        self.lock_state().page_down();
        self.refresh_after_handled().await;
    }

    async fn cursor_up(&self) {
        self.lock_state().cursor_up();
        self.refresh_after_handled().await;
    }

    async fn cursor_down(&self) {
        self.lock_state().cursor_down();
        self.refresh_after_handled().await;
    }

    fn set_surrounding_text(
        &self,
        text: zbus::zvariant::OwnedValue,
        cursor_pos: u32,
        _anchor_pos: u32,
    ) {
        // IBusText 序列化: ('IBusText', a{sv}, s, v) —— 第 3 个字段是文本
        let text_str = text
            .downcast_ref::<zbus::zvariant::Structure>()
            .ok()
            .and_then(|structure| {
                structure
                    .fields()
                    .get(2)
                    .and_then(|v| v.downcast_ref::<String>().ok())
            });
        match text_str {
            Some(t) => {
                *self
                    .surrounding
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((t, i32::try_from(cursor_pos).unwrap_or(0)));
            }
            None => log::debug!("unparseable surrounding text"),
        }
        drop(text); // zbus 接口参数按值传递，已提取完字符串
    }

    fn process_hand_writing_event(&self, _coordinates: Vec<f64>) {}

    fn cancel_hand_writing(&self, _n_strokes: u32) {}

    fn panel_extension_received(&self, _event: zbus::zvariant::OwnedValue) {}

    fn panel_extension_register_keys(&self, _data: zbus::zvariant::OwnedValue) {}
}

/// 供 cnt-daemon 使用的常量
pub const FACTORY_OBJ_PATH: &str = FACTORY_PATH;
