use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snippet {
    pub name: String,
    pub content: String,
    pub source_path: PathBuf,
}

impl Snippet {
    pub fn new(
        name: String,
        content: String,
        source_path: PathBuf,
    ) -> Self {
        Self {
            name,
            content: content.trim().to_string(),
            source_path,
        }
    }
}
