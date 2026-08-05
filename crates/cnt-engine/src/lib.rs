//! `IBus` 引擎的 D-Bus 服务端实现：
//! - `Factory`：org.freedesktop.IBus.Factory（路径 /org/freedesktop/IBus/Engine/Factory）
//! - `Engine`：org.freedesktop.IBus.Engine（路径 /org/freedesktop/IBus/Engine/N）
//!
//! 状态与 UI 发送在 [`core::EngineCore`]（可与语音任务共享），本文件只负责
//! 「协议方法 → 内核调用」的分派，以及按键专属的状态（Shift 单击、语音热键）。

pub mod core;
pub mod voice;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use fastrace::collector::SpanContext;
use fastrace::Span;
use zbus::connection::Connection;
use zbus::zvariant::OwnedObjectPath;

use cnt_decode::Decoder;
use cnt_input::Action;

pub use crate::core::EngineCore;
pub use crate::voice::{KeyOutcome, VoiceRuntime};

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

/// Shift 键单击检测状态（按下时间 + 是否被「借用」打过其他键）。
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
    /// 语音运行时（None = 未启用/不可用；此时输入法照常工作）。
    voice: Option<Arc<VoiceRuntime>>,
}

impl Factory {
    #[must_use]
    pub const fn new(
        conn: Connection,
        decoder: Arc<Decoder>,
        page_size: usize,
        voice: Option<Arc<VoiceRuntime>>,
    ) -> Self {
        Self {
            conn,
            counter: AtomicU32::new(1),
            decoder,
            page_size,
            voice,
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
        let engine = Engine::new(
            self.conn.clone(),
            path.clone(),
            self.decoder.clone(),
            self.page_size,
            self.voice.clone(),
        );
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
    core: Arc<EngineCore>,
    shift_tap: Mutex<ShiftTap>,
    voice: Option<Arc<VoiceRuntime>>,
}

impl Engine {
    #[must_use]
    pub fn new(
        conn: Connection,
        path: String,
        decoder: Arc<Decoder>,
        page_size: usize,
        voice: Option<Arc<VoiceRuntime>>,
    ) -> Self {
        Self {
            core: Arc::new(EngineCore::new(conn, path, decoder, page_size)),
            shift_tap: Mutex::new(ShiftTap::default()),
            voice,
        }
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
        // 语音热键最先判：PTT 用的是修饰键，必须在下面的「忽略修饰键」之前拦下。
        if let Some(voice) = &self.voice {
            match voice.handle_key(&self.core, keyval, state).await {
                KeyOutcome::Consumed => return Ok(true),
                KeyOutcome::Forwarded => return Ok(false),
                KeyOutcome::NotVoice => {}
            }
        }

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
                self.core.toggle_input_mode().await;
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
            // fastrace root span：覆盖按键处理的同步段（含 decode）。
            // 正常按键（<5ms）cancel 掉不上报，只保留慢按键的 span 树
            // 用于闪烁/卡顿诊断（daemon 的 ConsoleReporter 输出到 stderr）。
            let root = Span::root("process_key_event", SpanContext::random());
            let _guard = root.set_local_parent();
            log::debug!("keyval=0x{keyval:x} state=0x{state:x}");

            let act = {
                let mut st = self.core.lock_state();
                // 半角标点上下文：未组合且光标前一个字符是数字/英文 → 半角（3，→ 3,）。
                // 组合中（前面肯定是刚上屏的中文）永远全角。
                let punct_half_width = !st.is_composing() && self.core.punct_half_width();
                st.handle_key(keyval, &**self.core.decoder(), punct_half_width)
            };
            // 同步段结束：释放 thread-local local parent，避免跨 await 污染
            drop(_guard);
            if root.elapsed() < Some(std::time::Duration::from_millis(5)) {
                root.cancel();
            }
            act
        };

        match action {
            Action::Forward => {
                log::debug!("forwarded");
                self.core.note_forwarded_key(keyval);
                Ok(false)
            }
            Action::Handled => {
                self.core.refresh_after_handled().await;
                Ok(true)
            }
            Action::Commit { text, learned } => {
                self.core.handle_commit(text, learned).await;
                Ok(true)
            }
            Action::CommitAndForward { text, learned } => {
                // 提交预编辑后把原按键转发给应用（如 Shift+字母 输出大写）
                self.core.handle_commit(text, learned).await;
                self.core.note_forwarded_key(keyval);
                Ok(false)
            }
        }
    }

    async fn focus_in(&self) {
        // 新的输入上下文：丢掉旧的半角判定依据，并请应用下发光标周围文本
        self.core.forget_context();
        self.core.require_surrounding_text().await;
    }

    async fn focus_in_id(&self, _object_path: &str, _client: &str) {
        self.core.forget_context();
        self.core.require_surrounding_text().await;
    }

    async fn focus_out(&self) {
        self.leave().await;
    }

    async fn focus_out_id(&self, _object_path: &str) {
        self.leave().await;
    }

    async fn reset(&self) {
        self.core.lock_state().clear();
        self.core.forget_context();
        self.core.hide_ui().await;
    }

    async fn enable(&self) {
        self.core.forget_context();
        self.core.require_surrounding_text().await;
    }

    async fn disable(&self) {
        self.leave().await;
    }

    fn set_capabilities(&self, _caps: u32) {}

    fn set_cursor_location(&self, _x: i32, _y: i32, _w: i32, _h: i32) {}

    fn property_activate(&self, _name: &str, _state: u32) {}

    fn property_show(&self, _name: &str) {}

    fn property_hide(&self, _name: &str) {}

    /// 鼠标点击候选（index 为当前页内从 0 开始的下标）
    async fn candidate_clicked(&self, index: u32, _button: u32, _state: u32) {
        let action = {
            let mut st = self.core.lock_state();
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
            self.core.handle_commit(text, learned).await;
        }
    }

    async fn page_up(&self) {
        self.core.lock_state().page_up();
        self.core.refresh_after_handled().await;
    }

    async fn page_down(&self) {
        self.core.lock_state().page_down();
        self.core.refresh_after_handled().await;
    }

    async fn cursor_up(&self) {
        self.core.lock_state().cursor_up();
        self.core.refresh_after_handled().await;
    }

    async fn cursor_down(&self) {
        self.core.lock_state().cursor_down();
        self.core.refresh_after_handled().await;
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
                let cursor = usize::try_from(cursor_pos).unwrap_or(0);
                log::debug!(
                    "surrounding: cursor={cursor} char, prev_is_latin={}",
                    cnt_input::latin_before_cursor(&t, cursor)
                );
                self.core.set_surrounding(t, cursor);
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

impl Engine {
    /// 失去焦点 / 被禁用：清状态、落盘、**并关掉麦克风**。
    ///
    /// 语音会话绝不能跟着焦点漂到别的窗口去：切走了就停录，
    /// 这是隐私底线，也避免识别结果上屏到错误的应用里。
    async fn leave(&self) {
        if let Some(voice) = &self.voice {
            voice.cancel();
        }
        self.core.lock_state().clear();
        self.core.forget_context();
        self.core.hide_ui().await;
        self.core.flush_user();
    }
}

/// 供 cnt-daemon 使用的常量
pub const FACTORY_OBJ_PATH: &str = FACTORY_PATH;
