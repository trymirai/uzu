use std::path::{Path, PathBuf};

use uuid::Uuid;

use crate::{DownloadId, FileCheck, HttpDownloadRequest, lock_manager::lock_path_for_destination};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadConfig {
    pub download_id: DownloadId,
    pub request: HttpDownloadRequest,
    pub destination: PathBuf,
    pub artifact_root: PathBuf,
    pub file_check: FileCheck,
    pub expected_bytes: Option<u64>,
    pub manager_id: String,
    pub manager_instance_id: Uuid,
}

impl DownloadConfig {
    pub(crate) fn resume_artifact_path(
        &self,
        extension: &str,
    ) -> PathBuf {
        self.artifact_root.join(format!("download.{extension}"))
    }

    pub(crate) fn installation_artifact_path(&self) -> PathBuf {
        self.artifact_root.join("installing")
    }

    pub(crate) fn integrity_receipt_path(&self) -> PathBuf {
        self.artifact_root.join("integrity.json")
    }

    pub(crate) fn recovery_metadata_path(&self) -> PathBuf {
        self.artifact_root.join("recovery.json")
    }

    pub(crate) fn recovery_metadata_staging_path(&self) -> PathBuf {
        self.artifact_root.join("recovery.tmp")
    }

    pub(crate) fn lock_path(&self) -> PathBuf {
        lock_path_for_destination(&self.destination)
    }

    pub(crate) fn default_artifact_root(
        destination: &Path,
        download_id: DownloadId,
    ) -> PathBuf {
        destination
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(".uzu-download-manager")
            .join(download_id.to_string())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/traits/download_config_test.rs"]
mod tests;
