use std::path::PathBuf;

use crate::types::SourceMode;

pub struct Options {
    pub source: SourceMode,
    pub storage: Option<PathBuf>,
    pub output: PathBuf,
    pub model_ids: Vec<String>,
    pub prefill: Vec<usize>,
    pub generate: Vec<usize>,
    pub iterations: usize,
}

impl Options {
    /// Whether `id` is selected, i.e. it was named with `--model-id` or none were.
    pub fn selects(
        &self,
        id: &str,
    ) -> bool {
        self.model_ids.is_empty() || self.model_ids.iter().any(|candidate| candidate == id)
    }
}
