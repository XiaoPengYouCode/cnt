//! cnt-dict —— 拼音词库与用户学习模型。
//!
//! 两层结构：
//! 1. **静态基础词库**：`format.rs` 定义二进制布局，`writer.rs` 负责生成，
//!    `mmap_dict.rs` 用 mmap 零拷贝读取 + 二分查找。
//! 2. **用户个性化模型**：`user.rs`（动态调频 + 用户词库，明文 tsv 持久化）
//!    与 `model.rs`（把两者按分数混合排序）。

pub mod format;
pub mod mmap_dict;
pub mod model;
pub mod user;
pub mod writer;

pub use cnt_store::StoreError;
pub use mmap_dict::{Candidate, MmapDict, PrefixHit};
pub use model::{DictQuery, PinyinModel, DEFAULT_DICT_FILE, DEFAULT_USER_FILE, USER_BOOST};
pub use user::UserDb;
