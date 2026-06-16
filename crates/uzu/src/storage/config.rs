use std::{
    fs::read_dir,
    path::{Path, PathBuf},
};

use download_manager::FileDownloadManagerType;
use serde::{Deserialize, Serialize};
use shoji::types::model::Model;

use crate::device::Device;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub device: Device,
    pub base_path: Option<PathBuf>,
    pub name: String,
    #[serde(default)]
    pub download_manager_type: FileDownloadManagerType,
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
            download_manager_type: FileDownloadManagerType::default(),
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

    pub fn cache_model_metadata_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        Some(self.cache_models_path().join(reference_name).join(model.cache_identifier()).join("model.json"))
    }

    pub fn cache_path_has_model_files(path: &Path) -> bool {
        path.exists()
            && read_dir(path).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| !name.ends_with(".resume_data") && !name.starts_with('.'))
                })
            })
    }

    pub fn log_name(&self) -> String {
        format!("{}.log", self.name)
    }

    pub fn with_download_manager_type(
        &self,
        download_manager_type: FileDownloadManagerType,
    ) -> Self {
        Self {
            download_manager_type,
            ..self.clone()
        }
    }
}
