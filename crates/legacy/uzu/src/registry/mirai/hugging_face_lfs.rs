use serde::Deserialize;

#[derive(Deserialize)]
pub struct HuggingFaceLfs {
    pub sha256: Option<String>,
    pub size: Option<u64>,
}
