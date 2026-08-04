//! 开发用测试客户端：
//! 模拟一个使用 `IBus` 输入法的应用程序 —— 创建输入上下文、聚焦、
//! 发送按键，并打印收到的信号（`CommitText` / `UpdatePreeditText` / `UpdateLookupTable`）。
//!
//! 用法：
//!   cargo run -p cnt-test-client
//! 然后在 stdin 输入按键名（a-z、space、enter、backspace、esc、1-9、up、down、pageup、pagedown），
//! 每行一个按键，空行退出。

use std::io::{self, BufRead};

use futures_util::StreamExt;
use zbus::connection::Builder;
use zbus::message::Type;

const DAEMON: &str = "org.freedesktop.IBus";
const IC_IFACE: &str = "org.freedesktop.IBus.InputContext";

fn keysym(name: &str) -> Option<u32> {
    let b = name.as_bytes();
    // 数字键 1-9
    if b.len() == 1 && b[0].is_ascii_digit() && b[0] != b'0' {
        return Some(u32::from(b[0] - b'1' + 0x31));
    }
    match name {
        "space" => Some(0x20),
        "enter" | "return" => Some(0xff0d),
        "backspace" => Some(0xff08),
        "esc" | "escape" => Some(0xff1b),
        "up" => Some(0xff52),
        "down" => Some(0xff54),
        "pageup" => Some(0xff55),
        "pagedown" => Some(0xff56),
        _ => {
            if b.len() == 1 && b[0].is_ascii_alphabetic() {
                Some(u32::from(b[0].to_ascii_lowercase()))
            } else {
                None
            }
        }
    }
}

/// 后台任务：打印引擎发出的信号（`CommitText` / `UpdatePreeditText` / `UpdateLookupTable`）。
///
/// `member()` 返回 `zbus::zbus_names::MemberName` 而非 `&str`，
/// 闭包写法是正确类型（clippy 的简化建议在此不适用）。
#[allow(clippy::redundant_closure_for_method_calls)]
fn spawn_signal_printer(conn: zbus::Connection) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut stream = zbus::MessageStream::from(conn);
        while let Some(msg) = stream.next().await {
            let Ok(msg) = msg else { continue };
            if msg.message_type() != Type::Signal {
                continue;
            }
            let member = msg
                .header()
                .member()
                // `MemberName` 不是 `str`，clippy 的简化建议在此不适用
                .map(|m| m.to_string())
                .unwrap_or_default();
            if !matches!(
                member.as_str(),
                "CommitText" | "UpdatePreeditText" | "UpdateLookupTable" | "UpdateAuxiliaryText"
            ) {
                continue;
            }
            match member.as_str() {
                "CommitText" => {
                    match msg.body().deserialize::<zbus::zvariant::Value>() {
                        Ok(text) => println!("<< CommitText: {text:?}"),
                        Err(e) => println!("<< CommitText (deserialize err: {e})"),
                    }
                }
                "UpdatePreeditText" => {
                    let body = msg.body();
                    let r = body.deserialize::<(
                        zbus::zvariant::Value,
                        u32,
                        bool,
                        u32,
                    )>();
                    match r {
                        Ok((text, cursor, visible, mode)) => println!(
                            "<< UpdatePreeditText: text={text:?} cursor={cursor} visible={visible} mode={mode}"
                        ),
                        Err(e) => println!("<< UpdatePreeditText (deserialize err: {e})"),
                    }
                }
                "UpdateLookupTable" => {
                    match msg
                        .body()
                        .deserialize::<(zbus::zvariant::Value, bool)>()
                    {
                        Ok((table, visible)) => {
                            println!("<< UpdateLookupTable: visible={visible} table={table:?}");
                        }
                        Err(e) => println!("<< UpdateLookupTable (deserialize err: {e})"),
                    }
                }
                _ => println!("<< {member}"),
            }
        }
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = cnt_ibus::find_address()?;
    let conn = Builder::address(addr.as_str())?.build().await?;

    // 创建输入上下文（保持连接以维持 IC 存活）
    let ic_path: zbus::zvariant::OwnedObjectPath = conn
        .call_method(
            Some(DAEMON),
            "/org/freedesktop/IBus",
            Some("org.freedesktop.IBus"),
            "CreateInputContext",
            &("cnt-test-client",),
        )
        .await?
        .body()
        .deserialize()?;
    println!("input context: {ic_path}");

    // use-global-engine 开启时，全局引擎会自动应用到聚焦的上下文；
    // 显式 SetEngine 会报错，因此忽略失败。
    if let Err(e) = conn
        .call_method(Some(DAEMON), ic_path.as_str(), Some(IC_IFACE), "SetEngine", &("cnt",))
        .await
    {
        println!("(SetEngine skipped: {e})");
    } else {
        println!("engine set: cnt");
    }

    conn.call_method(Some(DAEMON), ic_path.as_str(), Some(IC_IFACE), "FocusIn", &())
        .await?;
    println!("focused");

    let sig_task = spawn_signal_printer(conn.clone());

    // 从 stdin 读取按键并发送
    println!("type keys (a-z, space, enter, backspace, esc, 1-9, up/down/pageup/pagedown); empty line to quit");
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        let Some(kv) = keysym(line) else {
            println!("unknown key: {line}");
            continue;
        };
        let handled: bool = conn
            .call_method(
                Some(DAEMON),
                ic_path.as_str(),
                Some(IC_IFACE),
                "ProcessKeyEvent",
                &(kv, 0u32, 0u32),
            )
            .await?
            .body()
            .deserialize()?;
        println!(">> {line} (keyval=0x{kv:x}) handled={handled}");
    }

    sig_task.abort();
    println!("done");
    Ok(())
}
