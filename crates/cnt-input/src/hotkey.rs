//! 热键解析与匹配：把配置里的 `"Control+Shift+space"` 变成可比较的键位。
//!
//! 放在 `cnt-input` 而不是 `cnt-config`，因为**键位是输入逻辑的知识**
//! （keysym、修饰键掩码都是 IBus/X11 的口径），配置层只该搬字符串。
//!
//! 支持的写法：
//!
//! ```text
//!   Control_R              单键（可以是修饰键本身 —— 按住说话最顺手的就是它）
//!   Control+Shift+space    组合键
//!   Alt+v / Super+F9 / Menu
//! ```
//!
//! 为什么允许「修饰键本身」当热键：按住说话（push-to-talk）要求这个键
//! **按着不放时不产生任何副作用**，右 Ctrl / 右 Alt 恰好满足；换成字母键
//! 会一直往应用里灌字符。

/// `IBus` 的修饰键掩码（与 `ibus` 的 `IBusModifierType` 一致）。
pub mod mask {
    /// Shift。
    pub const SHIFT: u32 = 1 << 0;
    /// Control。
    pub const CONTROL: u32 = 1 << 2;
    /// Alt（Mod1）。
    pub const ALT: u32 = 1 << 3;
    /// Super（Mod4）。
    pub const MOD4: u32 = 1 << 6;
    /// `IBus` 自己的 Super 掩码（部分桌面只置这一位）。
    pub const SUPER: u32 = 1 << 26;
    /// 按键释放。
    pub const RELEASE: u32 = 1 << 30;
}

/// 一个热键：keysym + 需要按住的修饰键。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hotkey {
    /// keysym（X11 keysym 值）。
    pub keyval: u32,
    /// 需要同时按住的修饰键掩码（Shift/Control/Alt/Super 的按位或）。
    pub mods: u32,
}

impl Hotkey {
    /// 解析 `"Control+Shift+space"` 形式的描述。
    ///
    /// 大小写不敏感（`ctrl+shift+space` 也可以），未知键名返回 None。
    #[must_use]
    pub fn parse(spec: &str) -> Option<Self> {
        let mut mods = 0;
        let mut keyval = None;
        for part in spec.split('+').map(str::trim).filter(|p| !p.is_empty()) {
            match part.to_ascii_lowercase().as_str() {
                "control" | "ctrl" => mods |= mask::CONTROL,
                "shift" => mods |= mask::SHIFT,
                "alt" | "meta" => mods |= mask::ALT,
                "super" | "win" | "cmd" => mods |= mask::MOD4,
                _ => keyval = Some(keysym(part)?),
            }
        }
        Some(Self {
            keyval: keyval?,
            mods,
        })
    }

    /// 这个热键本身是不是修饰键（决定「按住」语义能否成立）。
    #[must_use]
    pub const fn is_modifier_key(&self) -> bool {
        matches!(self.keyval, 0xffe1..=0xffee)
    }

    /// 事件是否匹配本热键（忽略 Lock 等无关位；释放事件由调用方区分）。
    ///
    /// 修饰键自身作为热键时不校验修饰位——按下 `Control_R` 的那一刻，
    /// `IBus` 给的 state 里 Control 位可能已置也可能未置（取决于桌面实现）。
    #[must_use]
    pub const fn matches(&self, keyval: u32, state: u32) -> bool {
        if keyval != self.keyval {
            return false;
        }
        if self.is_modifier_key() {
            return true;
        }
        let want = self.mods;
        let have = normalize_mods(state);
        want == have
    }
}

/// 把 state 里我们关心的修饰位抽出来（Super 的两种表示合并）。
#[must_use]
pub const fn normalize_mods(state: u32) -> u32 {
    let mut out = state & (mask::SHIFT | mask::CONTROL | mask::ALT);
    if state & (mask::MOD4 | mask::SUPER) != 0 {
        out |= mask::MOD4;
    }
    out
}

/// 事件是否是「释放」。
#[must_use]
pub const fn is_release(state: u32) -> bool {
    state & mask::RELEASE != 0
}

/// 键名 → keysym。覆盖热键场景够用的范围（字母数字、空格、功能键、修饰键、Menu）。
#[must_use]
pub fn keysym(name: &str) -> Option<u32> {
    // 单个 ASCII 可打印字符：keysym 等于其 ASCII 码
    if name.chars().count() == 1 {
        let c = name.chars().next()?;
        if c.is_ascii_graphic() {
            return Some(u32::from(c.to_ascii_lowercase()));
        }
    }
    let lower = name.to_ascii_lowercase();
    // F1..F24
    if let Some(n) = lower.strip_prefix('f')
        && let Ok(n) = n.parse::<u32>()
        && (1..=24).contains(&n)
    {
        return Some(0xffbe + n - 1);
    }
    Some(match lower.as_str() {
        "space" => 0x0020,
        "tab" => 0xff09,
        "return" | "enter" => 0xff0d,
        "escape" | "esc" => 0xff1b,
        "backspace" => 0xff08,
        "menu" => 0xff67,
        "shift_l" => 0xffe1,
        "shift_r" => 0xffe2,
        "control_l" => 0xffe3,
        "control_r" => 0xffe4,
        "caps_lock" => 0xffe5,
        "alt_l" => 0xffe9,
        "alt_r" => 0xffea,
        "super_l" => 0xffeb,
        "super_r" => 0xffec,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{is_release, mask, normalize_mods, Hotkey};

    #[test]
    fn parses_single_modifier_key() {
        let hk = Hotkey::parse("Control_R").expect("valid");
        assert_eq!(hk.keyval, 0xffe4);
        assert_eq!(hk.mods, 0);
        assert!(hk.is_modifier_key());
    }

    #[test]
    fn parses_combination() {
        let hk = Hotkey::parse("Control+Shift+space").expect("valid");
        assert_eq!(hk.keyval, 0x20);
        assert_eq!(hk.mods, mask::CONTROL | mask::SHIFT);
        assert!(!hk.is_modifier_key());
    }

    #[test]
    fn parsing_is_case_insensitive_and_tolerates_spaces() {
        assert_eq!(
            Hotkey::parse("ctrl + shift + SPACE"),
            Hotkey::parse("Control+Shift+space")
        );
        assert_eq!(Hotkey::parse("alt+v"), Hotkey::parse("Alt+V"));
    }

    #[test]
    fn rejects_unknown_keys() {
        assert!(Hotkey::parse("Control+nosuchkey").is_none());
        assert!(Hotkey::parse("Control").is_none()); // 只有修饰键，没有主键
        assert!(Hotkey::parse("").is_none());
    }

    #[test]
    fn function_keys_are_supported() {
        assert_eq!(Hotkey::parse("F9").expect("valid").keyval, 0xffc6);
        assert_eq!(Hotkey::parse("f1").expect("valid").keyval, 0xffbe);
        assert!(Hotkey::parse("F25").is_none());
    }

    #[test]
    fn combination_requires_exact_modifiers() {
        let hk = Hotkey::parse("Control+Shift+space").expect("valid");
        assert!(hk.matches(0x20, mask::CONTROL | mask::SHIFT));
        // 少一个修饰键 / 多一个修饰键都不算（避免和 Ctrl+空格 之类抢键）
        assert!(!hk.matches(0x20, mask::CONTROL));
        assert!(!hk.matches(0x20, mask::CONTROL | mask::SHIFT | mask::ALT));
        // 无关位（Lock）不影响匹配
        assert!(hk.matches(0x20, mask::CONTROL | mask::SHIFT | (1 << 1)));
        // 别的键不匹配
        assert!(!hk.matches(0x21, mask::CONTROL | mask::SHIFT));
    }

    #[test]
    fn modifier_hotkey_ignores_modifier_state() {
        // 按住 Control_R 时 state 是否已含 Control 位取决于桌面实现，不能作为判据
        let hk = Hotkey::parse("Control_R").expect("valid");
        assert!(hk.matches(0xffe4, 0));
        assert!(hk.matches(0xffe4, mask::CONTROL));
    }

    #[test]
    fn super_masks_are_unified() {
        assert_eq!(normalize_mods(mask::SUPER), mask::MOD4);
        assert_eq!(normalize_mods(mask::MOD4), mask::MOD4);
    }

    #[test]
    fn release_is_detected() {
        assert!(is_release(mask::RELEASE | mask::CONTROL));
        assert!(!is_release(mask::CONTROL));
    }
}
