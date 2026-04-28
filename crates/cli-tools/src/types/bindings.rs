use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bindings {
    Uniffi,
    Napi,
    Pyo3,
    Wasm,
}

impl Bindings {
    pub fn feature(self) -> String {
        match self {
            Bindings::Uniffi => "bindings-uniffi".to_string(),
            Bindings::Napi => "bindings-napi".to_string(),
            Bindings::Pyo3 => "bindings-pyo3".to_string(),
            Bindings::Wasm => "bindings-wasm".to_string(),
        }
    }
}
