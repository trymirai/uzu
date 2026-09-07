use crate::{DownloadTaskRequest, FileCheck};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadTaskKind {
    File {
        source_url: String,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    },
    Group(Vec<DownloadTaskRequest>),
}
