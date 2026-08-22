use std::path::{Component, Path, PathBuf};

use download_manager::FileDownloadManagerType;
use serde::{Deserialize, Serialize};
use shoji::types::model::Model;

use super::download_contents::DownloadContents;
use crate::device::Device;

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub device: Device,
    pub base_path: Option<PathBuf>,
    pub name: String,
    #[serde(default)]
    pub download_manager_type: FileDownloadManagerType,
    #[serde(skip)]
    pub download_contents: DownloadContents,
    #[serde(skip)]
    huggingface_api_key: Option<String>,
}

impl std::fmt::Debug for Config {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("Config")
            .field("device", &self.device)
            .field("base_path", &self.base_path)
            .field("name", &self.name)
            .field("download_manager_type", &self.download_manager_type)
            .field("download_contents", &self.download_contents)
            .field("huggingface_api_key", &self.huggingface_api_key.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
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
            download_contents: DownloadContents::default(),
            huggingface_api_key: None,
        }
    }

    pub fn cache_path(&self) -> PathBuf {
        self.base_path.clone().unwrap_or(PathBuf::from(self.device.home_path.clone())).join(".cache").join(&self.name)
    }

    pub fn cache_models_path(&self) -> PathBuf {
        self.cache_path().join("models")
    }

    /// The on-disk root for one revision of a model.
    ///
    /// `revision` is the Mirai checkpoint version or the Hugging Face commit;
    /// both are safe single path components.
    pub(crate) fn cache_model_path(
        &self,
        model: &Model,
        revision: &str,
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        let model_identifier = model.cache_identifier();
        if !is_safe_component(&model_identifier) || !is_safe_component(revision) {
            return None;
        }
        Some(self.cache_models_path().join(reference_name).join(model_identifier).join(revision))
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

    pub fn with_download_contents(
        &self,
        download_contents: DownloadContents,
    ) -> Self {
        Self {
            download_contents,
            ..self.clone()
        }
    }

    pub fn with_huggingface_api_key(
        &self,
        huggingface_api_key: Option<String>,
    ) -> Self {
        Self {
            huggingface_api_key,
            ..self.clone()
        }
    }

    pub(crate) fn huggingface_api_key(&self) -> Option<&str> {
        self.huggingface_api_key.as_deref()
    }
}

fn is_safe_component(value: &str) -> bool {
    let mut components = Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}
