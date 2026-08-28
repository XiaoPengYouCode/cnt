//! 与 `IBus` 协议相关的辅助函数：
//! - 查找 `IBus` 私有总线的地址
//! - 按照 `IBus` 的 D-Bus 序列化格式构造 `IBusComponent` / `IBusEngineDesc` /
//!   `IBusText` / `IBusLookupTable` 等对象（参见 ibus 源码中的 ibusserializable.c）

use std::collections::HashMap;

use zbus::zvariant::{StructureBuilder, Value};

/// `IBus` 相关错误（显式，thiserror）。
#[derive(Debug, thiserror::Error)]
pub enum IbusError {
    /// 找不到 `IBus` 私有总线地址文件
    #[error("no IBus address file found in {0}")]
    NoAddress(String),
    /// 需要 HOME 环境变量
    #[error("HOME not set")]
    NoHome,
    /// 无法确定 machine-id
    #[error("could not determine machine-id")]
    NoMachineId,
    /// IO 错误
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// 空的 `a{sv}`（`IBusSerializable` 的 attachments 字典，序列化时恒为空）。
fn empty_props() -> HashMap<String, Value<'static>> {
    HashMap::new()
}

/// 构造一个 `IBusAttrList` 对象值（`('IBusAttrList', a{sv}, av)`，此处恒为空列表）。
fn attr_list() -> Value<'static> {
    let attrs: Vec<Value<'static>> = Vec::new();
    let sb = StructureBuilder::new()
        .add_field("IBusAttrList")
        .add_field(empty_props())
        .add_field(attrs);
    Value::Structure(sb.build().expect("build attr list"))
}

/// 构造一个 `IBusText` 对象值（`('IBusText', a{sv}, s, v)`）。
///
/// # Panics
/// 构造 D-Bus 结构时若序列化字段非法会 panic（对固定合法字段不会发生）。
#[must_use]
pub fn text(s: &str) -> Value<'_> {
    let sb = StructureBuilder::new()
        .add_field("IBusText")
        .add_field(empty_props())
        .add_field(s)
        .add_field(attr_list());
    Value::Structure(sb.build().expect("build text"))
}

/// 构造一个 `IBusLookupTable` 对象值。
///
/// # Panics
/// 构造 D-Bus 结构时若序列化字段非法会 panic（对固定合法字段不会发生）。
///
/// 序列化格式（见 ibuslookuptable.c）：
/// `('IBusLookupTable', a{sv}, u page_size, u cursor_pos, b cursor_visible,
///  b round, i orientation, av candidates, av labels)`
#[must_use]
pub fn lookup_table<'a>(
    candidates: &'a [String],
    page_size: u32,
    cursor_pos: u32,
    cursor_visible: bool,
) -> Value<'a> {
    let cands: Vec<Value<'a>> = candidates.iter().map(|c| text(c)).collect();
    let labels: Vec<Value<'static>> = Vec::new();
    let sb = StructureBuilder::new()
        .add_field("IBusLookupTable")
        .add_field(empty_props())
        .add_field(page_size)
        .add_field(cursor_pos)
        .add_field(cursor_visible)
        .add_field(true) // round
        .add_field(0i32) // orientation = IBUS_ORIENTATION_SYSTEM
        .add_field(cands)
        .add_field(labels);
    Value::Structure(sb.build().expect("build lookup table"))
}

/// 构造一个 `IBusEngineDesc` 对象值。
///
/// # Panics
/// 构造 D-Bus 结构时若序列化字段非法会 panic（对固定合法字段不会发生）。
///
/// 序列化格式（见 ibusenginedesc.c，顺序不可改变）：
/// `('IBusEngineDesc', a{sv}, s name, s longname, s description, s language,
///  s license, s author, s icon, s layout, u rank, s hotkeys, s symbol,
///  s setup, s layout_variant, s layout_option, s version, s textdomain,
///  s icon_prop_key)`
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn engine_desc<'a>(
    name: &'a str,
    longname: &'a str,
    description: &'a str,
    language: &'a str,
    license: &'a str,
    author: &'a str,
    icon: &'a str,
    layout: &'a str,
    rank: u32,
    hotkeys: &'a str,
    symbol: &'a str,
    setup: &'a str,
    layout_variant: &'a str,
    layout_option: &'a str,
    version: &'a str,
    textdomain: &'a str,
    icon_prop_key: &'a str,
) -> Value<'a> {
    let sb = StructureBuilder::new()
        .add_field("IBusEngineDesc")
        .add_field(empty_props())
        .add_field(name)
        .add_field(longname)
        .add_field(description)
        .add_field(language)
        .add_field(license)
        .add_field(author)
        .add_field(icon)
        .add_field(layout)
        .add_field(rank)
        .add_field(hotkeys)
        .add_field(symbol)
        .add_field(setup)
        .add_field(layout_variant)
        .add_field(layout_option)
        .add_field(version)
        .add_field(textdomain)
        .add_field(icon_prop_key);
    Value::Structure(sb.build().expect("build engine desc"))
}

/// 构造一个 `IBusComponent` 对象值（用于 `RegisterComponent`）。
///
/// # Panics
/// 构造 D-Bus 结构时若序列化字段非法会 panic（对固定合法字段不会发生）。
///
/// 序列化格式（见 ibuscomponent.c）：
/// `('IBusComponent', a{sv}, s name, s description, s version, s license,
///  s author, s homepage, s exec, s textdomain, av observed_paths, av engines)`
///
/// 参数个数固定对应 `IBus` 协议字段，无法精简。
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn component<'a>(
    name: &'a str,
    description: &'a str,
    version: &'a str,
    license: &'a str,
    author: &'a str,
    homepage: &'a str,
    exec: &'a str,
    textdomain: &'a str,
    observed_paths: Vec<Value<'static>>,
    engines: Vec<Value<'static>>,
) -> Value<'a> {
    let sb = StructureBuilder::new()
        .add_field("IBusComponent")
        .add_field(empty_props())
        .add_field(name)
        .add_field(description)
        .add_field(version)
        .add_field(license)
        .add_field(author)
        .add_field(homepage)
        .add_field(exec)
        .add_field(textdomain)
        .add_field(observed_paths)
        .add_field(engines);
    Value::Structure(sb.build().expect("build component"))
}

/// 找到 `IBus` 私有总线的地址：
/// 1. 优先使用 `IBUS_ADDRESS` 环境变量；
/// 2. 否则在 `$XDG_CONFIG_HOME/ibus/bus/` 中按
///    `DISPLAY` / `WAYLAND_DISPLAY` 匹配地址文件。
///
/// # Errors
/// 找不到地址文件/缺少必要环境时返回 [`IbusError`]。
pub fn find_address() -> Result<String, IbusError> {
    if let Ok(addr) = std::env::var("IBUS_ADDRESS") {
        return Ok(strip_guid(&addr));
    }

    let config_home = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME")
            .map_err(|_| IbusError::NoHome)
            .unwrap_or_default();
        format!("{home}/.config")
    });
    let bus_dir = format!("{config_home}/ibus/bus");

    // 候选文件名：Wayland 优先，其次 DISPLAY
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(wayland) = std::env::var("WAYLAND_DISPLAY")
        && let Ok(machine_id) = read_machine_id()
    {
        candidates.push(format!("{bus_dir}/{machine_id}-unix-{wayland}"));
    }
    if let Ok(display) = std::env::var("DISPLAY")
        && let Ok(machine_id) = read_machine_id()
    {
        let mut parts = display.split(':');
        let host_part = parts.next().unwrap_or("");
        let disp_part = parts.next().unwrap_or("0.0");
        let host = if host_part.is_empty() {
            "unix"
        } else {
            host_part
        };
        let disp_num = disp_part.split('.').next().unwrap_or("0");
        candidates.push(format!("{bus_dir}/{machine_id}-{host}-{disp_num}"));
    }

    // 逐个尝试候选文件
    for f in &candidates {
        if let Some(addr) = read_address_file(f) {
            return Ok(addr);
        }
    }

    // 兑底：扫描目录中任意带 IBUS_ADDRESS 的文件
    if let Ok(entries) = std::fs::read_dir(&bus_dir) {
        for entry in entries.flatten() {
            if let Some(addr) = read_address_file(&entry.path().to_string_lossy()) {
                return Ok(addr);
            }
        }
    }

    Err(IbusError::NoAddress(bus_dir))
}

fn read_address_file(path: &str) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    for line in content.lines() {
        if let Some(addr) = line.strip_prefix("IBUS_ADDRESS=") {
            return Some(strip_guid(addr.trim()));
        }
    }
    None
}

/// zbus 的地址解析可能不识别 `,guid=...` 后缀，这里去掉它。
fn strip_guid(addr: &str) -> String {
    addr.split(',').next().unwrap_or(addr).to_string()
}

fn read_machine_id() -> Result<String, IbusError> {
    for p in ["/etc/machine-id", "/var/lib/dbus/machine-id"] {
        if let Ok(s) = std::fs::read_to_string(p) {
            let s = s.trim();
            if !s.is_empty() {
                return Ok(s.to_string());
            }
        }
    }
    Err(IbusError::NoMachineId)
}
