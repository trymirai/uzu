use std::path::PathBuf;

use crate::common::Value;

#[derive(Default)]
pub struct ModelSource {
    pub identifier: Option<String>,
    pub local_path: Option<PathBuf>,
    #[serde(default)]
    pub encodings: Vec<Value>,
}

impl ModelSource {
    pub fn from_local_path(path: PathBuf) -> ModelSource {
        ModelSource {
            local_path: Some(path),
            ..Self::default()
        }
    }
}
