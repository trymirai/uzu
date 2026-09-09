use std::path::PathBuf;

use download_manager::DownloadManagerType;
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
}

impl Config {
    pub fn new(
        device: Device,
        base_path: Option<PathBuf>,
        name: String,
        download_manager_type: DownloadManagerType,
    ) -> Self {
        Self {
            device,
            base_path,
            name,
            download_manager_type,
        }
    }
}
