use std::{fmt, path::{Path, PathBuf}};

use kiban::fs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    DownloadError, DownloadId, FileCheck, FileState, backends::common::reject_symlink_components,
    lock_manager::DestinationLockLease, traits::DownloadConfig,
};

const CURRENT_SCHEMA_VERSION: u8 = 1;

/// Private recovery identity used in manager artifacts and Apple task descriptions.
///
/// Version zero is the former public [`crate::DownloadInfo`] JSON shape. Keeping
/// its fields here lets downloads created by older releases be reclaimed without
/// changing that public type.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RecoveryMetadata {
    #[serde(default, skip_serializing_if = "is_legacy_schema")]
    schema_version: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    download_id: Option<DownloadId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    source_url: String,
    destination_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    crc32c: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_check: Option<FileCheck>,
}

impl RecoveryMetadata {
    pub(crate) fn new(
        download_id: DownloadId,
        source_url: &str,
        destination_path: &Path,
        expected_bytes: Option<u64>,
        file_check: FileCheck,
    ) -> Self {
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            download_id: Some(download_id),
            source_fingerprint: Some(source_fingerprint(source_url)),
            source_url: String::new(),
            destination_path: destination_path.to_string_lossy().into_owned(),
            expected_bytes,
            crc32c: None,
            file_check: Some(file_check),
        }
    }

    pub(crate) fn download_id(&self) -> Option<DownloadId> {
        match self.schema_version {
            0 => Some(crate::compute_download_id(Path::new(&self.destination_path))),
            CURRENT_SCHEMA_VERSION => self.download_id,
            _ => None,
        }
    }

    pub(crate) fn matches_request(
        &self,
        source_url: &str,
        expected_bytes: Option<u64>,
        file_check: &FileCheck,
    ) -> bool {
        let source_matches = match self.schema_version {
            0 => self.source_url == source_url,
            CURRENT_SCHEMA_VERSION => {
                self.source_fingerprint.as_deref() == Some(source_fingerprint(source_url).as_str())
            },
            _ => false,
        };
        let size_matches = self.schema_version == 0 || self.expected_bytes == expected_bytes;
        source_matches && size_matches && self.resolved_file_check() == *file_check
    }

    fn matches_recovery_identity(
        &self,
        download_id: DownloadId,
        destination_path: &Path,
        source_url: &str,
        expected_bytes: Option<u64>,
        file_check: &FileCheck,
    ) -> bool {
        let destination_matches = self.destination_path == destination_path.to_string_lossy();
        match self.schema_version {
            0 => destination_matches && self.matches_request(source_url, expected_bytes, file_check),
            CURRENT_SCHEMA_VERSION => {
                self.download_id == Some(download_id)
                    && destination_matches
                    && self.matches_request(source_url, expected_bytes, file_check)
            },
            _ => false,
        }
    }

    fn resolved_file_check(&self) -> FileCheck {
        self.file_check.clone().unwrap_or_else(|| self.crc32c.clone().map(FileCheck::CRC).unwrap_or(FileCheck::None))
    }

    pub(crate) fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub(crate) fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Debug for RecoveryMetadata {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter
            .debug_struct("RecoveryMetadata")
            .field("schema_version", &self.schema_version)
            .field("download_id", &self.download_id)
            .field("destination_path", &self.destination_path)
            .field("expected_bytes", &self.expected_bytes)
            .field("file_check", &self.resolved_file_check())
            .finish()
    }
}

fn source_fingerprint(source_url: &str) -> String {
    format!("{:x}", Sha256::digest(source_url.as_bytes()))
}

const fn is_legacy_schema(version: &u8) -> bool {
    *version == 0
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResumeRecoveryObservation {
    pub resume_state: FileState,
    pub cleanup_paths: Box<[PathBuf]>,
}

pub(crate) async fn observe_resume_recovery(
    config: &DownloadConfig,
    resume_artifact_path: &Path,
) -> Result<ResumeRecoveryObservation, DownloadError> {
    validate_recovery_root(config).await?;
    let resume_file_state = owned_file_state(resume_artifact_path).await?;
    let metadata_path = config.recovery_metadata_path();
    let metadata_state = read_metadata_state(config, &metadata_path).await?;
    let staging_path = config.recovery_metadata_staging_path();
    let staging_state = owned_file_state(&staging_path).await?;

    let recoverable = resume_file_state == OwnedFileState::Regular && metadata_state == MetadataState::Matching;
    let has_stale_pair =
        !recoverable && (resume_file_state != OwnedFileState::Missing || metadata_state != MetadataState::Missing);
    let mut cleanup_paths = Vec::new();
    if has_stale_pair {
        cleanup_paths.push(resume_artifact_path.to_path_buf());
        cleanup_paths.push(metadata_path);
    }
    if staging_state != OwnedFileState::Missing {
        cleanup_paths.push(staging_path);
    }
    validate_recovery_root(config).await?;

    Ok(ResumeRecoveryObservation {
        resume_state: if recoverable {
            FileState::Exists
        } else {
            FileState::Missing
        },
        cleanup_paths: cleanup_paths.into_boxed_slice(),
    })
}

pub(crate) async fn prepare_fresh_recovery(
    config: &DownloadConfig,
    resume_artifact_path: &Path,
    _destination_lease: &DestinationLockLease,
) -> Result<(), DownloadError> {
    remove_owned_file_if_present(config, resume_artifact_path).await?;
    remove_owned_file_if_present(config, &config.recovery_metadata_path()).await?;
    remove_owned_file_if_present(config, &config.recovery_metadata_staging_path()).await?;
    write_recovery_metadata(config).await
}

pub(crate) async fn prepare_resume_recovery(
    config: &DownloadConfig,
    resume_artifact_path: &Path,
    destination_lease: &DestinationLockLease,
) -> Result<bool, DownloadError> {
    let observation = observe_resume_recovery(config, resume_artifact_path).await?;
    if observation.resume_state == FileState::Exists {
        for path in &observation.cleanup_paths {
            remove_owned_file_if_present(config, path).await?;
        }
        return Ok(true);
    }

    prepare_fresh_recovery(config, resume_artifact_path, destination_lease).await?;
    Ok(false)
}

pub(crate) async fn write_recovery_metadata(config: &DownloadConfig) -> Result<(), DownloadError> {
    validate_recovery_root(config).await?;
    let metadata = RecoveryMetadata::new(
        config.download_id,
        &config.request.url,
        &config.destination,
        config.expected_bytes,
        config.file_check.clone(),
    );
    let bytes = metadata.to_json()?.into_bytes();
    let metadata_path = config.recovery_metadata_path();
    let staging_path = config.recovery_metadata_staging_path();

    remove_owned_file_if_present(config, &staging_path).await?;
    if let Err(error) = fs::asyn::write_with_sync_all(&staging_path, &bytes).await {
        let _ = fs::asyn::remove_file(&staging_path).await;
        return Err(DownloadError::from(error));
    }
    validate_recovery_root(config).await?;
    if let Err(error) = fs::asyn::rename(&staging_path, &metadata_path).await {
        let _ = fs::asyn::remove_file(&staging_path).await;
        return Err(DownloadError::from(error));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetadataState {
    Missing,
    Matching,
    Invalid,
}

async fn read_metadata_state(
    config: &DownloadConfig,
    metadata_path: &Path,
) -> Result<MetadataState, DownloadError> {
    match owned_file_state(metadata_path).await? {
        OwnedFileState::Missing => Ok(MetadataState::Missing),
        OwnedFileState::Other => Ok(MetadataState::Invalid),
        OwnedFileState::Regular => {
            let bytes = fs::asyn::read(metadata_path).await?;
            let Ok(info) = serde_json::from_slice::<RecoveryMetadata>(&bytes) else {
                return Ok(MetadataState::Invalid);
            };
            if info.matches_recovery_identity(
                config.download_id,
                &config.destination,
                &config.request.url,
                config.expected_bytes,
                &config.file_check,
            ) {
                Ok(MetadataState::Matching)
            } else {
                Ok(MetadataState::Invalid)
            }
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnedFileState {
    Missing,
    Regular,
    Other,
}

async fn owned_file_state(path: &Path) -> Result<OwnedFileState, DownloadError> {
    #[cfg(not(target_family = "wasm"))]
    {
        match tokio::fs::symlink_metadata(path).await {
            Ok(metadata) if metadata.file_type().is_file() => Ok(OwnedFileState::Regular),
            Ok(_) => Ok(OwnedFileState::Other),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(OwnedFileState::Missing),
            Err(error) => Err(DownloadError::from(error)),
        }
    }
    #[cfg(target_family = "wasm")]
    {
        if !fs::asyn::try_exists(path).await? {
            Ok(OwnedFileState::Missing)
        } else if fs::asyn::is_file(path).await {
            Ok(OwnedFileState::Regular)
        } else {
            Ok(OwnedFileState::Other)
        }
    }
}

async fn remove_owned_file_if_present(
    config: &DownloadConfig,
    path: &Path,
) -> Result<(), DownloadError> {
    validate_recovery_root(config).await?;
    match fs::asyn::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(DownloadError::from(error)),
    }
}

async fn validate_recovery_root(config: &DownloadConfig) -> Result<(), DownloadError> {
    reject_symlink_components(&config.artifact_root).await
}

#[cfg(test)]
#[path = "../tests/unit/recovery_metadata_test.rs"]
mod tests;
