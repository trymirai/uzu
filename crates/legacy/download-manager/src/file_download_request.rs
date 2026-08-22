use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use unicode_normalization::UnicodeNormalization;

use crate::{FileCheck, HttpDownloadRequest, RelativeFilePath};

/// Everything needed to download one member of a file group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDownloadRequest {
    pub relative_path: RelativeFilePath,
    pub source: HttpDownloadRequest,
    pub expected_bytes: Option<u64>,
    pub check: FileCheck,
}

impl FileDownloadRequest {
    pub fn new(
        source: impl Into<HttpDownloadRequest>,
        relative_path: RelativeFilePath,
        check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Self {
        Self {
            relative_path,
            source: source.into(),
            expected_bytes,
            check,
        }
    }
}

/// A validated set of files sharing one destination root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDownloadGroupSpec {
    destination_root: PathBuf,
    files: Arc<[FileDownloadRequest]>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FileDownloadGroupSpecError {
    #[error("file download group cannot be empty")]
    Empty,
    #[error("file download group destination root cannot be empty")]
    EmptyDestinationRoot,
    #[error("file download group contains duplicate destination: {0}")]
    DuplicateDestination(RelativeFilePath),
    #[error("file and directory destinations conflict: {ancestor} and {descendant}")]
    ConflictingDestinations {
        ancestor: RelativeFilePath,
        descendant: RelativeFilePath,
    },
    #[error("declared file sizes overflow u64")]
    TotalBytesOverflow,
}

impl FileDownloadGroupSpec {
    pub fn new(
        destination_root: impl Into<PathBuf>,
        files: impl IntoIterator<Item = FileDownloadRequest>,
    ) -> Result<Self, FileDownloadGroupSpecError> {
        let destination_root = destination_root.into();
        if destination_root.as_os_str().is_empty() {
            return Err(FileDownloadGroupSpecError::EmptyDestinationRoot);
        }

        let mut files: Vec<_> = files.into_iter().collect();
        if files.is_empty() {
            return Err(FileDownloadGroupSpecError::Empty);
        }

        let mut destinations = HashMap::with_capacity(files.len());
        let mut expected_total = 0_u64;
        for file in &files {
            if destinations
                .insert(portable_path_key(file.relative_path.as_path()), file.relative_path.clone())
                .is_some()
            {
                return Err(FileDownloadGroupSpecError::DuplicateDestination(file.relative_path.clone()));
            }
            if let Some(expected_bytes) = file.expected_bytes {
                expected_total =
                    expected_total.checked_add(expected_bytes).ok_or(FileDownloadGroupSpecError::TotalBytesOverflow)?;
            }
        }

        for descendant in &files {
            let mut ancestor_path = descendant.relative_path.as_path().parent();
            while let Some(path) = ancestor_path.filter(|path| !path.as_os_str().is_empty()) {
                if let Some(ancestor) = destinations.get(&portable_path_key(path)) {
                    return Err(FileDownloadGroupSpecError::ConflictingDestinations {
                        ancestor: ancestor.clone(),
                        descendant: descendant.relative_path.clone(),
                    });
                }
                ancestor_path = path.parent();
            }
        }

        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

        Ok(Self {
            destination_root,
            files: files.into(),
        })
    }

    pub fn destination_root(&self) -> &std::path::Path {
        &self.destination_root
    }

    pub fn files(&self) -> &[FileDownloadRequest] {
        &self.files
    }

    pub(crate) fn with_destination_root(
        mut self,
        destination_root: PathBuf,
    ) -> Self {
        self.destination_root = destination_root;
        self
    }

    pub fn destination_for(
        &self,
        file: &FileDownloadRequest,
    ) -> PathBuf {
        self.destination_root.join(file.relative_path.as_path())
    }
}

pub(crate) fn portable_path_key(path: &Path) -> String {
    path.to_string_lossy().nfd().flat_map(char::to_lowercase).collect()
}

#[cfg(test)]
#[path = "../tests/unit/file_download_request_test.rs"]
mod tests;
