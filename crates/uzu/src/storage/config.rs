use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use shoji::types::model::Model;

use crate::device::Device;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub device: Device,
    pub base_path: Option<PathBuf>,
    pub name: String,
}

impl Config {
    pub fn new(
        device: Device,
        base_path: Option<PathBuf>,
        name: String,
    ) -> Self {
        Self {
            device,
            base_path,
            name,
        }
    }

    pub fn cache_path(&self) -> PathBuf {
        self.base_path.clone().unwrap_or(PathBuf::from(self.device.home_path.clone())).join(".cache").join(&self.name)
    }

    pub fn cache_models_path(&self) -> PathBuf {
        self.cache_path().join("models")
    }

    pub fn cache_model_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        let checkpoint_version = model.checkpoint_version()?;
        Some(self.cache_models_path().join(reference_name).join(model.cache_identifier()).join(checkpoint_version))
    }

    pub fn log_name(&self) -> String {
        format!("{}.log", self.name)
    }
}
