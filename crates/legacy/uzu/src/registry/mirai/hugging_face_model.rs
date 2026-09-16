use serde::Deserialize;

use super::hugging_face_file::HuggingFaceFile;

#[derive(Deserialize)]
pub struct HuggingFaceModel {
    pub sha: String,
    pub siblings: Vec<HuggingFaceFile>,
}
