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

    pub fn cache_model_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        self.cache_model_path_at_revision(model, &model.checkpoint_version()?)
    }

    pub(crate) fn cache_model_path_at_revision(
        &self,
        model: &Model,
        revision: &str,
    ) -> Option<PathBuf> {
        self.cache_model_path_for_source(model, revision, &[])
    }

    pub(crate) fn cache_model_path_for_source(
        &self,
        model: &Model,
        revision: &str,
        canonical_source_identity: &[u8],
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        let repositories = model.repo_ids().join("\n");
        let model_identity = format!("{reference_name}\n{}\n{repositories}", model.identifier);
        let model_key = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, model_identity.as_bytes());
        let mut source_identity = Vec::with_capacity(revision.len() + canonical_source_identity.len() + 1);
        source_identity.extend_from_slice(revision.as_bytes());
        source_identity.push(0);
        source_identity.extend_from_slice(canonical_source_identity);
        let revision_key = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, &source_identity);
        Some(self.cache_models_path().join(reference_name).join(model_key.to_string()).join(revision_key.to_string()))
    }

    pub(crate) fn legacy_cache_model_path(
        &self,
        model: &Model,
    ) -> Option<PathBuf> {
        let reference_name = model.reference_name()?;
        let model_identifier = model.cache_identifier();
        let checkpoint_version = model.checkpoint_version()?;
        if !is_safe_component(&model_identifier) || !is_safe_component(&checkpoint_version) {
            return None;
        }
        Some(self.cache_models_path().join(reference_name).join(model_identifier).join(checkpoint_version))
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
