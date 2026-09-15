//! 语音输入在引擎侧的接线：热键 → 会话 → 事件泵 → 上屏。
//!
//! ```text
//!   按住 ptt_key   ──► Voice::start(PushToTalk) ─┐
//!   松开 ptt_key   ──► Voice::stop()             ├─► 事件泵（tokio 任务）
//!   toggle_key     ──► start/stop(Continuous)   ─┘        │
//!                                                          ▼
//!                                      EngineCore：预编辑提示 / CommitText
//! ```
//!
//! ## 两条硬规则
//!
//! 1. **按键处理路径上不做任何阻塞或等待**：`handle_key` 只发命令、只 spawn，
//!    识别的几百毫秒全在 `cnt-voice` 的线程里。否则一次语音就会把整个输入法
//!    卡住几百毫秒（IBus 的按键是串行的，卡住 = 打字丢键）。
//! 2. **开始语音前先把拼音预编辑上屏**：用户可能打了半句拼音又想说话，
//!    半截拼音既不能丢，也不能和语音结果混在一起。
//!
//! ## PTT 为什么用修饰键
//!
//! 「按住某键说话」要求这个键按住期间对应用无副作用：右 Alt / 右 Ctrl 满足，
//! 字母键会一直往应用灌字符。默认 `Alt_R`（很多 2024 年后的键盘已经把右 Ctrl
//! 换成了 Copilot 键），且**不消费**该事件——照常转发给应用，行为与平时按 Alt 一致。

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use cnt_input::hotkey::{Hotkey, is_release, mask};
use cnt_voice::{Mode, Voice, VoiceError, VoiceEvent, VoiceSession};
use fastrace::Span;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use futures_util::FutureExt as _;

use crate::core::EngineCore;

/// 语音热键的处理结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOutcome {
    /// 与语音无关，继续走正常按键处理。
    NotVoice,
    /// 语音已处理并消费该按键（不转发给应用）。
    Consumed,
    /// 语音已处理，但按键仍要转发给应用（PTT 用修饰键时如此）。
    Forwarded,
}

/// 预编辑里的状态提示（面板上看得见「现在在录音还是在识别」）。
///
/// 非流式模型没有中间结果，所以「它在听」只能靠这一行字 + 音量条表达。
/// 没有这个反馈，用户会反复松手重按（以为没生效）——这是语音输入最常见的
/// 交互失败。
const HINT_LISTENING: &str = "🎤 说话中";
const HINT_LISTENING_CONT: &str = "🎤 常开听写";
const HINT_RECOGNIZING: &str = "🎤 识别中…";
/// 取消提示（Esc 之后短暂显示，随即随会话结束清掉）。
const HINT_CANCELLED: &str = "🎤 已取消";

/// Esc 的 keysym（录音中按 Esc = 放弃，不上屏）。
const KEY_ESCAPE: u32 = 0xff1b;

/// 音量条：把 dBFS 映射成 5 格。
///
/// 说话的正常区间大约 -40~-6 dBFS，所以下限取 -50、上限取 -5；
/// 低于下限全空（等于告诉用户「没听到声音」，可能是麦克风选错了）。
fn meter(db: f32) -> &'static str {
    const BARS: [&str; 6] = ["▁▁▁▁▁", "▂▁▁▁▁", "▂▃▁▁▁", "▂▃▄▁▁", "▂▃▄▅▁", "▂▃▄▅▇"];
    // dBFS → 0..=5 档：clamp 之后转 usize 是安全的（截断即取整，正是要的）
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((db + 50.0) / 45.0 * 5.0).round().clamp(0.0, 5.0) as usize;
    BARS[idx]
}

/// 热键兜底值（配置写错时用；与 `cnt-config` 的默认字符串一致）。
const PTT_FALLBACK: Hotkey = Hotkey {
    keyval: 0xffea, // Alt_R
    mods: 0,
};
const TOGGLE_FALLBACK: Hotkey = Hotkey {
    keyval: 0x20, // space
    mods: mask::CONTROL | mask::SHIFT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoicePhase {
    Idle,
    Starting,
    Recording,
    Stopping,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PttState {
    Released,
    Held,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuousState {
    Off,
    On,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopState {
    None,
    Pending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CancelState {
    Clear,
    Requested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceAction {
    None,
    Start(Mode),
    Stop,
}

#[derive(Debug)]
struct VoiceState {
    phase: VoicePhase,
    ptt: PttState,
    continuous: ContinuousState,
    stop: StopState,
    cancel: CancelState,
    session_id: Option<u64>,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            phase: VoicePhase::Idle,
            ptt: PttState::Released,
            continuous: ContinuousState::Off,
            stop: StopState::None,
            cancel: CancelState::Clear,
            session_id: None,
        }
    }
}

/// 引擎侧的语音运行时（所有输入上下文共享一份：麦克风和模型都只该有一份）。
pub struct VoiceRuntime {
    voice: Arc<Voice>,
    ptt: Hotkey,
    toggle: Hotkey,
    /// 串行化按键、提交和取消，保证会话状态有单一线性顺序。
    lifecycle_gate: tokio::sync::Mutex<()>,
    state: Mutex<VoiceState>,
}

impl VoiceRuntime {
    /// 建立运行时；热键描述非法时退回默认值（不能因为写错一个热键就没了语音）。
    #[must_use]
    pub fn new(voice: Voice, ptt_spec: &str, toggle_spec: &str) -> Self {
        let ptt = Hotkey::parse(ptt_spec).unwrap_or_else(|| {
            log::warn!("invalid voice.ptt_key {ptt_spec:?}, falling back to Alt_R");
            PTT_FALLBACK
        });
        let toggle = Hotkey::parse(toggle_spec).unwrap_or_else(|| {
            log::warn!(
                "invalid voice.toggle_key {toggle_spec:?}, falling back to Control+Shift+space"
            );
            TOGGLE_FALLBACK
        });
        if !ptt.is_modifier_key() {
            log::warn!(
                "voice.ptt_key {ptt_spec:?} is not a modifier key: holding it will also type into the application"
            );
        }
        log::info!(
            "voice ready: backend={}, punct={}, device={}, ptt={ptt_spec}, toggle={toggle_spec}",
            voice.backend(),
            voice.punctuator(),
            voice.device()
        );
        Self {
            voice: Arc::new(voice),
            ptt,
            toggle,
            lifecycle_gate: tokio::sync::Mutex::new(()),
            state: Mutex::new(VoiceState::default()),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, VoiceState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn output_allowed(&self, session_id: u64) -> bool {
        let state = self.lock_state();
        state.session_id == Some(session_id)
            && state.phase != VoicePhase::Finished
            && state.cancel == CancelState::Clear
    }

    fn continuous_for(&self, session_id: u64) -> bool {
        let state = self.lock_state();
        state.session_id == Some(session_id)
            && state.continuous == ContinuousState::On
            && state.cancel == CancelState::Clear
    }

    fn finish_session(&self, session_id: u64) -> bool {
        let mut state = self.lock_state();
        if state.session_id != Some(session_id) {
            return false;
        }
        state.phase = VoicePhase::Finished;
        state.session_id = None;
        state.ptt = PttState::Released;
        state.continuous = ContinuousState::Off;
        state.stop = StopState::None;
        state.cancel = CancelState::Clear;
        state.phase = VoicePhase::Idle;
        true
    }

    fn cancel_requested(&self) -> bool {
        self.lock_state().cancel == CancelState::Requested
    }

    fn has_session(&self) -> bool {
        self.lock_state().phase != VoicePhase::Idle
    }

    fn reset_start(&self) {
        let mut state = self.lock_state();
        if state.session_id.is_some() || state.phase != VoicePhase::Starting {
            return;
        }
        state.phase = VoicePhase::Idle;
        state.ptt = PttState::Released;
        state.continuous = ContinuousState::Off;
        state.stop = StopState::None;
        state.cancel = CancelState::Clear;
        state.session_id = None;
    }

    /// 处理一个按键事件（在任何其他按键逻辑之前调用）。
    pub async fn handle_key(
        self: &Arc<Self>,
        core: &Arc<EngineCore>,
        keyval: u32,
        state: u32,
    ) -> KeyOutcome {
        let release = is_release(state);

        // ---- 录音中按 Esc：放弃（一个字都不上屏）----
        // 这条必须有：说错了、被人打断了、误触了，用户需要一个「作废」出口，
        // 否则唯一选择是让错的内容上屏再删。
        if keyval == KEY_ESCAPE && !release && self.has_session() {
            log::info!("voice: cancelled by Esc");
            self.cancel().await;
            core.set_voice_hint(Some(HINT_CANCELLED.to_owned())).await;
            return KeyOutcome::Consumed;
        }

        if self.toggle.matches(keyval, state) {
            return self.handle_toggle(core, release).await;
        }

        if self.ptt.matches(keyval, state) {
            return self.handle_ptt(core, release).await;
        }

        KeyOutcome::NotVoice
    }

    async fn handle_toggle(self: &Arc<Self>, core: &Arc<EngineCore>, release: bool) -> KeyOutcome {
        if release {
            return KeyOutcome::Consumed;
        }
        let action = {
            let mut state = self.lock_state();
            let action = if state.continuous == ContinuousState::On {
                state.continuous = ContinuousState::Off;
                if state.phase == VoicePhase::Starting {
                    state.stop = StopState::Pending;
                    VoiceAction::None
                } else {
                    state.phase = VoicePhase::Stopping;
                    VoiceAction::Stop
                }
            } else if state.phase == VoicePhase::Idle {
                state.continuous = ContinuousState::On;
                state.cancel = CancelState::Clear;
                state.phase = VoicePhase::Starting;
                VoiceAction::Start(Mode::Continuous)
            } else {
                log::debug!("voice: continuous start ignored while session is active");
                VoiceAction::None
            };
            drop(state);
            action
        };
        match action {
            VoiceAction::None => {}
            VoiceAction::Start(mode) => self.begin(core, mode).await,
            VoiceAction::Stop => {
                log::info!("voice: continuous mode off");
                if let Err(e) = self.voice.stop() {
                    log::error!("voice: stop failed: {e}");
                }
            }
        }
        KeyOutcome::Consumed
    }

    async fn handle_ptt(self: &Arc<Self>, core: &Arc<EngineCore>, release: bool) -> KeyOutcome {
        let action = if release {
            let mut state = self.lock_state();
            match state.ptt {
                PttState::Released => VoiceAction::None,
                PttState::Held => {
                    state.ptt = PttState::Released;
                    match state.phase {
                        VoicePhase::Starting => {
                            state.stop = StopState::Pending;
                            VoiceAction::None
                        }
                        VoicePhase::Recording => {
                            state.phase = VoicePhase::Stopping;
                            VoiceAction::Stop
                        }
                        _ => VoiceAction::None,
                    }
                }
            }
        } else {
            let mut state = self.lock_state();
            match state.ptt {
                PttState::Held => VoiceAction::None,
                PttState::Released => {
                    state.ptt = PttState::Held;
                    if state.continuous == ContinuousState::On {
                        log::debug!("voice: ptt ignored (continuous mode active)");
                        VoiceAction::None
                    } else if state.phase == VoicePhase::Idle {
                        state.cancel = CancelState::Clear;
                        state.phase = VoicePhase::Starting;
                        VoiceAction::Start(Mode::PushToTalk)
                    } else if state.phase == VoicePhase::Starting {
                        state.stop = StopState::None;
                        VoiceAction::None
                    } else {
                        VoiceAction::None
                    }
                }
            }
        };
        match action {
            VoiceAction::Start(mode) => {
                log::debug!("voice: ptt pressed");
                self.begin(core, mode).await;
            }
            VoiceAction::Stop => {
                log::debug!("voice: ptt released");
                if let Err(e) = self.voice.stop() {
                    log::error!("voice: stop failed: {e}");
                }
            }
            VoiceAction::None => {}
        }
        if self.ptt.is_modifier_key() {
            KeyOutcome::Forwarded
        } else {
            KeyOutcome::Consumed
        }
    }

    /// 会话结束时（引擎失去焦点 / 被禁用）收尾：麦克风不能跟着焦点漂。
    pub async fn cancel(&self) {
        let _lifecycle_guard = self.lifecycle_gate.lock().await;
        let should_cancel = {
            let mut state = self.lock_state();
            if state.phase == VoicePhase::Idle {
                false
            } else {
                state.cancel = CancelState::Requested;
                state.phase = VoicePhase::Stopping;
                state.continuous = ContinuousState::Off;
                state.ptt = PttState::Released;
                true
            }
        };
        if should_cancel {
            log::info!("voice: cancelling session (focus lost)");
            if let Err(e) = self.voice.cancel() {
                log::error!("voice: cancel failed: {e}");
            }
        }
    }

    /// 开一个会话：先把拼音预编辑上屏，再起事件泵。
    async fn begin(self: &Arc<Self>, core: &Arc<EngineCore>, mode: Mode) {
        core.commit_composing().await;
        if self.cancel_requested() {
            self.reset_start();
            return;
        }
        let session = match self.voice.start(mode) {
            Ok(s) => s,
            Err(VoiceError::Busy) => {
                log::debug!("voice: session already active");
                self.reset_start();
                return;
            }
            Err(e) => {
                log::error!("voice: cannot start session: {e}");
                self.reset_start();
                core.set_voice_hint(None).await;
                return;
            }
        };
        let session_id = session.session_id();
        let stop_after_start = {
            let mut state = self.lock_state();
            state.session_id = Some(session_id);
            let cancel = state.cancel == CancelState::Requested;
            let stop = cancel
                || state.stop == StopState::Pending
                || (mode == Mode::PushToTalk && state.ptt == PttState::Released);
            state.stop = StopState::None;
            state.phase = if stop {
                VoicePhase::Stopping
            } else {
                VoicePhase::Recording
            };
            stop
        };
        let core = Arc::clone(core);
        let runtime = Arc::clone(self);
        tokio::spawn(async move {
            let cleanup_runtime = Arc::clone(&runtime);
            let result = AssertUnwindSafe(runtime.run_session(core.clone(), session))
                .catch_unwind()
                .await;
            if result.is_err() {
                log::error!("voice: event pump panicked; closing session {session_id}");
                if cleanup_runtime.finish_session(session_id) {
                    core.set_voice_hint(None).await;
                }
            }
        });
        if stop_after_start {
            let cancelled = self.cancel_requested();
            let result = if cancelled {
                self.voice.cancel()
            } else {
                self.voice.stop()
            };
            if let Err(e) = result {
                log::error!("voice: stop-after-start failed: {e}");
            }
        }
    }

    async fn run_session(self: Arc<Self>, core: Arc<EngineCore>, mut session: VoiceSession) {
        let session_id = session.session_id();
        while let Some(event) = session.next().await {
            if event.session_id() != session_id {
                log::warn!(
                    "voice: ignoring event from unexpected session {} (current {})",
                    event.session_id(),
                    session_id
                );
                continue;
            }
            match event {
                VoiceEvent::Started { mode, .. } if self.output_allowed(session_id) => {
                    let hint = match mode {
                        Mode::PushToTalk => HINT_LISTENING,
                        Mode::Continuous => HINT_LISTENING_CONT,
                    };
                    core.set_voice_hint(Some(hint.to_owned())).await;
                }
                VoiceEvent::Level { secs, db, .. } if self.output_allowed(session_id) => {
                    let base = if self.continuous_for(session_id) {
                        HINT_LISTENING_CONT
                    } else {
                        HINT_LISTENING
                    };
                    core.set_voice_hint(Some(format!("{base} {secs:.1}s {}", meter(db))))
                        .await;
                }
                VoiceEvent::Recognizing { .. } if self.output_allowed(session_id) => {
                    core.set_voice_hint(Some(HINT_RECOGNIZING.to_owned())).await;
                }
                VoiceEvent::Text { text, .. } => {
                    let _operation_guard = core.lock_operations().await;
                    let _lifecycle_guard = self.lifecycle_gate.lock().await;
                    if !self.output_allowed(session_id) {
                        continue;
                    }
                    let chars = text.chars().count();
                    core.commit_voice_locked(&text)
                        .in_span(
                            Span::root("voice_commit", SpanContext::random())
                                .with_property(|| ("chars", chars.to_string())),
                        )
                        .await;
                    if self.continuous_for(session_id) {
                        core.set_voice_hint(Some(HINT_LISTENING_CONT.to_owned()))
                            .await;
                    }
                }
                VoiceEvent::Empty { .. } => log::debug!("voice: nothing recognized"),
                VoiceEvent::Error { error, .. } => log::error!("voice: {error}"),
                VoiceEvent::Stopped { .. } => break,
                VoiceEvent::Started { .. }
                | VoiceEvent::Level { .. }
                | VoiceEvent::Recognizing { .. } => {}
            }
        }
        if self.finish_session(session_id) {
            core.set_voice_hint(None).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KeyOutcome, meter};

    #[test]
    fn meter_maps_db_to_bars() {
        // 静音 → 全空（等于提示「没听到声音」）
        assert_eq!(meter(-90.0), "▁▁▁▁▁");
        assert_eq!(meter(-50.0), "▁▁▁▁▁");
        // 正常说话 → 中段
        let normal = meter(-25.0);
        assert!(normal.contains('▃'), "unexpected {normal}");
        // 大声 / 越界都不 panic
        assert_eq!(meter(-5.0), "▂▃▄▅▇");
        assert_eq!(meter(10.0), "▂▃▄▅▇");
    }

    #[test]
    fn outcomes_are_distinct() {
        // 三种结果必须能区分：Forwarded 与 Consumed 混淆会让 Ctrl 键失灵
        assert_ne!(KeyOutcome::NotVoice, KeyOutcome::Consumed);
        assert_ne!(KeyOutcome::Consumed, KeyOutcome::Forwarded);
    }
}
