use crate::{Checksum, DownloadTaskRequest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadTaskKind {
    File {
        source_url: String,
        expected_checksum: Option<Checksum>,
        expected_bytes: Option<u64>,
    },
    Group(Vec<DownloadTaskRequest>),
}
