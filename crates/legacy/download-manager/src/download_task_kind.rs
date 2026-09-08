use crate::{Crc32c, DownloadTaskRequest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadTaskKind {
    File {
        source_url: String,
        expected_crc32c: Option<Crc32c>,
        expected_bytes: Option<u64>,
    },
    Group(Vec<DownloadTaskRequest>),
}
