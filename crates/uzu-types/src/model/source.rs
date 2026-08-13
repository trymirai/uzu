use std::path::PathBuf;

#[derive(Default)]
pub struct ModelSource {
    pub identifier: Option<String>,
    pub local_path: Option<PathBuf>,
}

impl ModelSource {
    pub fn from_local_path(path: PathBuf) -> ModelSource {
        ModelSource {
            local_path: Some(path),
            ..Self::default()
        }
    }
}
