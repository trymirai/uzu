use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Example {
    pub name: String,
    pub relative_path: PathBuf,
    pub full_path: PathBuf,
    pub content: String,
    pub exportable: bool,
}

impl Example {
    pub fn new(
        name: String,
        relative_path: PathBuf,
        full_path: PathBuf,
        content: String,
        exportable: bool,
    ) -> Self {
        Self {
            name,
            relative_path,
            full_path,
            content: content.trim().to_string(),
            exportable,
        }
    }
}
