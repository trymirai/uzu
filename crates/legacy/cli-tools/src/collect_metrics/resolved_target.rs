use std::path::PathBuf;

use crate::types::SourceMode;

/// A target whose `config.json`, `tokenizer.json`, and safetensors header are on disk.
pub struct ResolvedTarget {
    pub source: SourceMode,
    pub id: String,
    pub model_dir: PathBuf,
    pub header_path: PathBuf,
}
