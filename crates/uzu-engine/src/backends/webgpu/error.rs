#[derive(Debug, thiserror::Error)]
pub enum WebGpuError {
    #[error("WebGPU: {0}")]
    Runtime(String),
    #[error("WebGPU does not support this operation")]
    NotSupported,
    #[error("Unsupported WebGPU operation: {0}")]
    Unsupported(String),
    #[error("WebGPU buffer size {requested} exceeds the supported per-buffer limit {limit}")]
    BufferTooLarge {
        requested: usize,
        limit: usize,
    },
}

impl From<wasm_bindgen::JsValue> for WebGpuError {
    fn from(value: wasm_bindgen::JsValue) -> Self {
        Self::Runtime(value.as_string().unwrap_or_else(|| format!("{value:?}")))
    }
}
