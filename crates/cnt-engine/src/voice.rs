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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cnt_input::hotkey::{is_release, mask, Hotkey};
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;
use fastrace::Span;
use cnt_voice::{Mode, Voice, VoiceError, VoiceEvent};

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

/// 引擎侧的语音运行时（所有输入上下文共享一份：麦克风和模型都只该有一份）。
pub struct VoiceRuntime {
    voice: Arc<Voice>,
    ptt: Hotkey,
    toggle: Hotkey,
    /// PTT 是否处于「按住」状态（按键会重复上报，必须去抖）。
    ptt_held: AtomicBool,
    /// 常开模式是否开启。
    continuous: AtomicBool,
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
            ptt_held: AtomicBool::new(false),
            continuous: AtomicBool::new(false),
        }
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
        if keyval == KEY_ESCAPE && !release && self.voice.is_active() {
            log::info!("voice: cancelled by Esc");
            self.continuous.store(false, Ordering::SeqCst);
            self.ptt_held.store(false, Ordering::SeqCst);
            if let Err(e) = self.voice.cancel() {
                log::error!("voice: cancel failed: {e}");
            }
            core.set_voice_hint(Some(HINT_CANCELLED.to_owned())).await;
            return KeyOutcome::Consumed;
        }

        // ---- 切换式常开 ----
        if self.toggle.matches(keyval, state) {
            if release {
                return KeyOutcome::Consumed; // 组合键的释放事件不给应用
            }
            // fetch_xor 一次原子翻转并拿到旧值（先 load 再 swap 会有竞态）
            if self.continuous.fetch_xor(true, Ordering::SeqCst) {
                log::info!("voice: continuous mode off");
                if let Err(e) = self.voice.stop() {
                    log::error!("voice: stop failed: {e}");
                }
            } else {
                self.begin(core, Mode::Continuous).await;
            }
            return KeyOutcome::Consumed;
        }

        // ---- 按住说话 ----
        if self.ptt.matches(keyval, state) {
            if release {
                if self.ptt_held.swap(false, Ordering::SeqCst) {
                    log::debug!("voice: ptt released");
                    if let Err(e) = self.voice.stop() {
                        log::error!("voice: stop failed: {e}");
                    }
                }
            } else if !self.ptt_held.swap(true, Ordering::SeqCst) {
                if self.continuous.load(Ordering::SeqCst) {
                    log::debug!("voice: ptt ignored (continuous mode active)");
                } else {
                    log::debug!("voice: ptt pressed");
                    self.begin(core, Mode::PushToTalk).await;
                }
            }
            // 修饰键要照常转发，应用侧的 Ctrl 行为不能被输入法吞掉
            return if self.ptt.is_modifier_key() {
                KeyOutcome::Forwarded
            } else {
                KeyOutcome::Consumed
            };
        }

        KeyOutcome::NotVoice
    }

    /// 会话结束时（引擎失去焦点 / 被禁用）收尾：麦克风不能跟着焦点漂。
    pub fn cancel(&self) {
        if self.voice.is_active() {
            log::info!("voice: cancelling session (focus lost)");
            self.continuous.store(false, Ordering::SeqCst);
            self.ptt_held.store(false, Ordering::SeqCst);
            if let Err(e) = self.voice.stop() {
                log::error!("voice: stop failed: {e}");
            }
        }
    }

    /// 开一个会话：先把拼音预编辑上屏，再起事件泵。
    async fn begin(self: &Arc<Self>, core: &Arc<EngineCore>, mode: Mode) {
        core.commit_composing().await;
        let session = match self.voice.start(mode) {
            Ok(s) => s,
            Err(VoiceError::Busy) => {
                log::debug!("voice: session already active");
                return;
            }
            Err(e) => {
                log::error!("voice: cannot start session: {e}");
                core.set_voice_hint(None).await;
                return;
            }
        };
        let core = Arc::clone(core);
        let runtime = Arc::clone(self);
        // 事件泵：识别在别的线程，这里只把结果搬到 IBus 上。
        // spawn 而不是 await：按键处理必须立刻返回。
        tokio::spawn(async move {
            let mut session = session;
            while let Some(event) = session.next().await {
                match event {
                    VoiceEvent::Started(Mode::PushToTalk) => {
                        core.set_voice_hint(Some(HINT_LISTENING.to_owned())).await;
                    }
                    VoiceEvent::Started(Mode::Continuous) => {
                        core.set_voice_hint(Some(HINT_LISTENING_CONT.to_owned()))
                            .await;
                    }
                    VoiceEvent::Level { secs, db } => {
                        let base = if runtime.continuous.load(Ordering::SeqCst) {
                            HINT_LISTENING_CONT
                        } else {
                            HINT_LISTENING
                        };
                        core.set_voice_hint(Some(format!("{base} {secs:.1}s {}", meter(db))))
                            .await;
                    }
                    VoiceEvent::Recognizing => {
                        core.set_voice_hint(Some(HINT_RECOGNIZING.to_owned())).await;
                    }
                    VoiceEvent::Text(text) => {
                        // 常开模式一句一次上屏；提示随后重画（会话还在继续）。
                        //
                        // 这里独立成一棵 root span 而不是接到 cnt-voice 的
                        // voice_release_to_commit 下面：上屏发生在**引擎的 tokio 任务**里，
                        // 跨任务连成一棵树需要把 SpanContext 随事件传过来。
                        // 先量出 D-Bus 这一段到底有多贵，再决定值不值得做那层传递。
                        let chars = text.chars().count();
                        core.commit_voice(&text)
                            .in_span(
                                Span::root("voice_commit", SpanContext::random())
                                    .with_property(|| ("chars", chars.to_string())),
                            )
                            .await;
                        if runtime.continuous.load(Ordering::SeqCst) {
                            core.set_voice_hint(Some(HINT_LISTENING_CONT.to_owned()))
                                .await;
                        }
                    }
                    VoiceEvent::Empty => {
                        log::debug!("voice: nothing recognized");
                    }
                    VoiceEvent::Error(e) => {
                        log::error!("voice: {e}");
                    }
                    VoiceEvent::Stopped => break,
                }
            }
            runtime.continuous.store(false, Ordering::SeqCst);
            runtime.ptt_held.store(false, Ordering::SeqCst);
            core.set_voice_hint(None).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{meter, KeyOutcome};

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
