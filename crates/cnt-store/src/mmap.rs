//! mmap 文件包装与区域校验。

use std::fs::File;
use std::ops::Deref;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::StoreError;

/// 只读 mmap 文件（Deref 到 `[u8]`）。
pub struct MmapFile {
    mmap: Mmap,
}

impl MmapFile {
    /// 打开并 mmap 一个文件。
    ///
    /// # Errors
    /// 文件不存在/无法 mmap 时返回 [`StoreError`]。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self { mmap })
    }

    /// 文件长度（字节）。
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// 是否为空文件。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

impl Deref for MmapFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.mmap
    }
}

/// 校验一组区域边界都在文件长度内（checked 加法，防溢出/越界 panic）。
///
/// # Errors
/// 任一区域越界时返回 [`StoreError::Region`]。
pub fn validate_regions(
    len: usize,
    regions: &[(usize, usize, &'static str)],
) -> Result<(), StoreError> {
    for (off, size, what) in regions {
        off.checked_add(*size)
            .filter(|e| *e <= len)
            .ok_or(StoreError::Region(what))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_validated() {
        assert!(validate_regions(100, &[(0, 100, "a"), (10, 5, "b")]).is_ok());
        assert!(validate_regions(100, &[(0, 101, "a")]).is_err());
        assert!(validate_regions(usize::MAX, &[(usize::MAX, 1, "overflow")]).is_err());
    }
}
