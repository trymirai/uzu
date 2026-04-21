use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub identifier: String,
    pub backend_identifier: String,
    pub name: String,
    pub path: String,
}

impl Config {
    pub fn new(
        identifier: String,
        backend_identifier: String,
        name: String,
        path: String,
    ) -> Self {
        Self {
            identifier,
            backend_identifier,
            name,
            path,
        }
    }

    pub fn lalamo(
        backend_identifier: String,
        path: String,
    ) -> Self {
        let models_path = PathBuf::from(&path).join("models");
        Self::new(
            "lalamo".to_string(),
            backend_identifier,
            "Lalamo".to_string(),
            models_path.to_string_lossy().to_string(),
        )
    }
}
