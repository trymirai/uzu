use std::path::{Path, PathBuf};

use bon::bon;
use uuid::Uuid;

use crate::{Crc32c, DownloadId, DownloadTaskKind};

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
        #[builder(into)] expected_crc32c: Option<Crc32c>,
        expected_bytes: Option<u64>,
    ) -> Self {
        Self {
            destination,
            kind: DownloadTaskKind::File {
                source_url,
                expected_crc32c,
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
            kind: DownloadTaskKind::Group(
                subrequests.into_iter().map(|subrequest| subrequest.resolved(&destination)).collect(),
            ),
            destination,
        }
    }

    pub fn download_id(&self) -> DownloadId {
        Uuid::new_v5(&Uuid::NAMESPACE_URL, self.destination.to_string_lossy().as_bytes())
    }

    fn resolved(
        self,
        parent: &Path,
    ) -> Self {
        Self {
            destination: parent.join(&self.destination),
            kind: match self.kind {
                DownloadTaskKind::Group(subrequests) => DownloadTaskKind::Group(
                    subrequests.into_iter().map(|subrequest| subrequest.resolved(parent)).collect(),
                ),
                kind => kind,
            },
        }
    }
}
