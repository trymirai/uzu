use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    DeleteFile {
        path: PathBuf,
    },
    DeleteCrcCache {
        path: PathBuf,
    },
    DeleteResumeArtifact {
        path: PathBuf,
    },
    SaveIntegrityCache {
        destination: PathBuf,
        receipt_path: PathBuf,
        file_check: crate::FileCheck,
    },
}
