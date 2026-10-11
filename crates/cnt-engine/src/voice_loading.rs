//! 后台语音加载结果在所有输入上下文间共享，按键路径只读、不等待加载。

use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use cnt_input::hotkey::{Hotkey, is_release};
use fastrace::Span;
use fastrace::collector::SpanContext;
use fastrace::future::FutureExt;

use crate::voice::{PTT_FALLBACK, TOGGLE_FALLBACK};
use crate::{EngineCore, KeyOutcome, VoiceRuntime};

/// 一次性发布语音运行时。空槽 = 加载中，Some(None) = 加载失败。
/// 工厂和已存在的引擎共享此对象，不能在 `CreateEngine` 时复制空运行时快照。
pub struct VoiceAvailability {
    enabled: bool,
    ptt: Hotkey,
    toggle: Hotkey,
    runtime: OnceLock<Option<Arc<VoiceRuntime>>>,
}

impl VoiceAvailability {
    #[must_use]
    pub fn new(enabled: bool, ptt: &str, toggle: &str) -> Self {
        Self {
            enabled,
            ptt: Hotkey::parse(ptt).unwrap_or(PTT_FALLBACK),
            toggle: Hotkey::parse(toggle).unwrap_or(TOGGLE_FALLBACK),
            runtime: OnceLock::new(),
        }
    }

    /// 发布加载结果；重复发布不会替换现有运行时。
    pub fn complete(&self, runtime: Option<Arc<VoiceRuntime>>) -> bool {
        self.runtime.set(runtime).is_ok()
    }

    /// 立即读取就绪运行时；加载中/失败/禁用均返回 None。
    #[must_use]
    pub fn runtime(&self) -> Option<&Arc<VoiceRuntime>> {
        self.enabled
            .then(|| self.runtime.get().and_then(Option::as_ref))
            .flatten()
    }

    pub(crate) async fn handle_key(
        &self,
        core: &Arc<EngineCore>,
        pending: &Mutex<PendingVoiceKeys>,
        keyval: u32,
        state: u32,
    ) -> KeyOutcome {
        if !self.enabled {
            return KeyOutcome::NotVoice;
        }
        // 本次事件使用同一个快照；发布若发生在此后，当前按键仍走未就绪分支。
        let runtime = self.runtime();
        let release = is_release(state);
        let decision = {
            let mut keys = pending.lock().unwrap_or_else(PoisonError::into_inner);
            // 松开其他修饰键后收到的 release 仍须清除被抑制的原始按键。
            let toggle = self.toggle.matches(keyval, state)
                || (keys.toggle.held && keyval == self.toggle.keyval);
            let ptt =
                self.ptt.matches(keyval, state) || (keys.ptt.held && keyval == self.ptt.keyval);
            if toggle {
                keys.toggle
                    .intercept(release, runtime.is_some())
                    .map(|first| (first, false))
            } else if ptt {
                keys.ptt
                    .intercept(release, runtime.is_some())
                    .map(|first| (first, self.ptt.is_modifier_key()))
            } else {
                None
            }
        };
        if let Some((first, forward)) = decision {
            if first || release {
                let hint = if release {
                    None
                } else if self.runtime.get().is_some_and(Option::is_none) {
                    Some("语音不可用".to_owned())
                } else {
                    Some("语音准备中，请稍后重试".to_owned())
                };
                // 用户按键到未就绪提示 IO 完成，独立记录这次操作。
                async {
                    let _operation = core.lock_operations().await;
                    core.set_voice_hint(hint).await;
                }
                .in_span(Span::root("voice_pending_key", SpanContext::random()))
                .await;
            }
            return if forward {
                KeyOutcome::Forwarded
            } else {
                KeyOutcome::Consumed
            };
        }
        if let Some(runtime) = runtime {
            runtime.handle_key(core, keyval, state).await
        } else {
            KeyOutcome::NotVoice
        }
    }
}

/// 未就绪期间按下的键必须先释放；就绪本身不触发录音，也不重放按键。
#[derive(Default)]
struct PendingKey {
    held: bool,
}

impl PendingKey {
    /// Some(first) 表示拦下事件；first 表示首次按下，需要显示提示。
    const fn intercept(&mut self, release: bool, ready: bool) -> Option<bool> {
        if !self.held && ready {
            return None;
        }
        let first = !self.held && !release;
        self.held = !release;
        Some(first)
    }
}

#[derive(Default)]
pub struct PendingVoiceKeys {
    ptt: PendingKey,
    toggle: PendingKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_during_held_key_requires_release_and_fresh_press() {
        let mut key = PendingKey::default();
        assert_eq!(key.intercept(false, false), Some(true));
        assert_eq!(key.intercept(false, true), Some(false)); // autorepeat after readiness
        assert_eq!(key.intercept(true, true), Some(false));
        assert_eq!(key.intercept(false, true), None);
    }

    #[test]
    fn release_before_readiness_does_not_replay_press() {
        let mut key = PendingKey::default();
        assert_eq!(key.intercept(false, false), Some(true));
        assert_eq!(key.intercept(true, false), Some(false));
        assert_eq!(key.intercept(true, true), None);
        assert_eq!(key.intercept(false, true), None);
    }

    #[test]
    fn failed_load_is_visible_to_existing_contexts() {
        let shared = Arc::new(VoiceAvailability::new(true, "Alt_R", "Control+Shift+space"));
        let existing_context = Arc::clone(&shared);
        assert!(existing_context.runtime.get().is_none());
        shared.complete(None);
        assert!(existing_context.runtime.get().is_some());
        assert!(existing_context.runtime().is_none());
    }
}
