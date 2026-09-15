use serde::Deserialize;

use super::hugging_face_lfs::HuggingFaceLfs;

#[derive(Deserialize)]
pub struct HuggingFaceFile {
    pub rfilename: String,
    pub size: Option<u64>,
    #[serde(rename = "blobId")]
    pub blob_id: Option<String>,
    pub lfs: Option<HuggingFaceLfs>,
}
