use std::path::PathBuf;

use download_manager::{BearerToken, DownloadManagerType};
use serde::{Deserialize, Serialize};

use crate::device::Device;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Config {
    pub device: Device,
    pub base_path: Option<PathBuf>,
    pub name: String,
    #[serde(default)]
    pub download_manager_type: DownloadManagerType,
    pub huggingface_url: String,
    #[serde(skip)]
    pub huggingface_api_key: Option<BearerToken>,
}

impl Config {
    pub fn new(
        device: Device,
        base_path: Option<PathBuf>,
        name: String,
        download_manager_type: DownloadManagerType,
        huggingface_url: String,
        huggingface_api_key: Option<BearerToken>,
    ) -> Self {
        Self {
            device,
            base_path,
            name,
            download_manager_type,
            huggingface_url,
            huggingface_api_key,
        }
    }
}
