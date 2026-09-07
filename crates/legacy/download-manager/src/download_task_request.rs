use std::path::PathBuf;

use bon::bon;

use crate::{DownloadId, DownloadTaskKind, FileCheck, compute_download_id};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadTaskRequest {
    pub destination: PathBuf,
    pub kind: DownloadTaskKind,
}

#[bon]
impl DownloadTaskRequest {
    #[builder(finish_fn = build)]
    pub fn file(
        #[builder(into)] destination: PathBuf,
        #[builder(into)] source_url: String,
        #[builder(default)] file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Self {
        Self {
            destination,
            kind: DownloadTaskKind::File {
                source_url,
                file_check,
                expected_bytes,
            },
        }
    }

    #[builder(finish_fn = build)]
    pub fn group(
        #[builder(into)] destination: PathBuf,
        subrequests: Vec<Self>,
    ) -> Self {
        Self {
            destination,
            kind: DownloadTaskKind::Group(subrequests),
        }
    }
}

impl DownloadTaskRequest {
    pub fn download_id(&self) -> DownloadId {
        compute_download_id(&self.destination)
    }
}
