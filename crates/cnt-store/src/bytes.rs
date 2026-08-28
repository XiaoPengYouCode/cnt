//! 小端字节读取与窄类型转换。
//!
//! 全部用手写的 `from_le_bytes` 切片读取，无对齐要求（mmap 友好）。

/// 读小端 `u16`。
#[must_use]
pub const fn read_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// 读小端 `u32`。
#[must_use]
pub const fn read_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// 读小端 `u64`。
#[must_use]
pub const fn read_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// 读小端 `f32`（字节序无关，非对齐安全）。
#[must_use]
pub const fn read_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// 把 `usize` 安全转成格式允许的窄类型。
///
/// # Errors
/// 数值超出 `T` 的表示范围时返回 IO 错误。
pub fn narrow<T>(v: usize, what: &str) -> std::io::Result<T>
where
    T: TryFrom<usize>,
{
    T::try_from(v).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{what} too large for the binary format ({v} bytes)"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut b = [0u8; 16];
        b[..2].copy_from_slice(&0x_1234u16.to_le_bytes());
        b[2..6].copy_from_slice(&0x_dead_beefu32.to_le_bytes());
        b[6..14].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        assert_eq!(read_u16(&b[..2]), 0x_1234);
        assert_eq!(read_u32(&b[2..6]), 0x_dead_beef);
        assert_eq!(read_u64(&b[6..14]), 0x1234_5678_9abc_def0);
        assert_eq!(
            read_f32(&(-3.0f32).to_le_bytes()).to_bits(),
            (-3.0f32).to_bits()
        );
    }

    #[test]
    fn narrow_converts_and_rejects() {
        assert_eq!(narrow::<u16>(3, "len").unwrap(), 3u16);
        assert!(narrow::<u16>(70_000, "len").is_err());
    }
}
