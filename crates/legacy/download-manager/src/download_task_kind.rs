use crate::{BearerToken, Checksum, DownloadTaskRequest};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadTaskKind {
    File {
        source_url: String,
        bearer_token: Option<BearerToken>,
        expected_checksum: Option<Checksum>,
        expected_bytes: Option<u64>,
    },
    Group(Vec<DownloadTaskRequest>),
}
