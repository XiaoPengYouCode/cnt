//! 模型契约探查：把 ONNX 的输入/输出/metadata 原样打印出来。
//!
//! 这个模块的存在是一次教训的产物：换模型时最容易踩的不是推理代码，而是
//! **契约假设**——输入叫什么、dtype 是 int32 还是 int64、CMVN 参数在不在
//! metadata 里、词表是明文还是 base64、blank 是 0 还是最后一个 token。
//! 靠猜就会得到「能跑但输出乱码」这种最费时间的失败。
//!
//! 所以先探查、再适配：`cnt-asr-tools info <model-dir>` 直接打出这些事实。

use std::path::Path;

use cnt_asr::AsrError;
use ort::session::Session;
use ort::value::ValueType;

/// 一个输入/输出的声明。
#[derive(Debug, Clone)]
pub struct Outlet {
    /// 名字（`session.run` 里要用的键）。
    pub name: String,
    /// 元素类型（`Float32`/`Int32`/`Int64`…）。
    pub dtype: String,
    /// 形状（-1 表示动态维）。
    pub shape: Vec<i64>,
}

/// 模型契约快照。
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub inputs: Vec<Outlet>,
    pub outputs: Vec<Outlet>,
    /// ONNX 自定义 metadata（`neg_mean`/`inv_stddev`/`lfr_window_size`…）。
    pub metadata: Vec<(String, String)>,
}

/// 打开模型并读出契约（不做推理，加载即返回）。
///
/// # Errors
/// 模型打不开时返回错误。
pub fn inspect(model: impl AsRef<Path>) -> Result<ModelInfo, AsrError> {
    let model = model.as_ref();
    let session = Session::builder()
        .map_err(|e| AsrError::Backend(e.to_string()))?
        .commit_from_file(model)
        .map_err(|e| AsrError::Model(format!("{}: {e}", model.display())))?;

    let outlets = |list: &[ort::value::Outlet]| -> Vec<Outlet> {
        list.iter()
            .map(|o| {
                let (dtype, shape) = match o.dtype() {
                    ValueType::Tensor { ty, shape, .. } => {
                        (format!("{ty:?}"), shape.iter().copied().collect())
                    }
                    other => (format!("{other:?}"), Vec::new()),
                };
                Outlet {
                    name: o.name().to_owned(),
                    dtype,
                    shape,
                }
            })
            .collect()
    };

    let inputs = outlets(session.inputs());
    let outputs = outlets(session.outputs());

    let mut metadata = Vec::new();
    if let Ok(meta) = session.metadata() {
        if let Ok(keys) = meta.custom_keys() {
            for key in keys {
                let value = meta.custom(&key).unwrap_or_default();
                metadata.push((key, value));
            }
        }
        drop(meta);
    }
    metadata.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(ModelInfo {
        inputs,
        outputs,
        metadata,
    })
}

impl ModelInfo {
    /// metadata 取值。
    #[must_use]
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// 是否声明了这个输入。
    #[must_use]
    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|i| i.name == name)
    }
}
