use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Root path not found")]
    RootPathNotFound,
    #[error("Path not found: {path}")]
    PathNotFound {
        path: PathBuf,
    },
    #[error("Unable to clone directory")]
    UnableToCloneDirectory,
    #[error("Unable to remove ignored entities")]
    UnableToRemoveIgnoredEntities,
    #[error("Unable to update JSON")]
    UnableToUpdateJSON,
    #[error("Unable to extract snippets")]
    UnableToExtractSnippets,
    #[error("Unable to extract examples")]
    UnableToExtractExamples,
    #[error("Unable to update README")]
    UnableToUpdateREADME,
    #[error("Unable to update Package.swift")]
    UnableToUpdatePackageSwift,
    #[error("Unable to prepare workspace")]
    UnableToPrepareWorkspace,
    #[error("Unable to prepare workspace/ts-napi")]
    UnableToPrepareWorkspaceTSNAPI,
    #[error("Unable to prepare bindings/swift")]
    UnableToPrepareBindingsSwift,
    #[error("Unable to prepare bindings/ts")]
    UnableToPrepareBindingsTS,
    #[error("Unable to prepare bindings/python")]
    UnableToPrepareBindingsPython,
    #[error("Unable to prepare workspace/ts")]
    UnableToPrepareWorkspaceTS,
    #[error("Unable to prepare workspace/ts-npm")]
    UnableToPrepareWorkspaceTSNPM,
    #[error("Unable to prepare workspace/swift")]
    UnableToPrepareWorkspaceSwift,
    #[error("Unable to prepare workspace/swift-spm")]
    UnableToPrepareWorkspaceSwiftSPM,
    #[error("Unable to prepare docs")]
    UnableToPrepareDocs,
    #[error("Unable to prepare platform")]
    UnableToPreparePlatform,
    #[error("Unable to sync into repo")]
    UnableToSyncIntoRepo,
}
