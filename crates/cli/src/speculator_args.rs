use std::{path::Path, sync::Arc};

use clap::{Args, ValueEnum};
use uzu::{
    backends::metal::Metal,
    prelude::{NGramSpeculator, NeuralSpeculator, SpeculatorConfig},
};

#[derive(ValueEnum, Debug, Clone)]
pub enum SpeculatorType {
    Pard,
    Ngram,
}

#[derive(Args, Debug, Clone)]
pub struct SpeculatorArgs {
    #[arg(long, value_enum)]
    /// Type of the speculator to use. If not specified, speculation is disabled.
    pub speculator_type: Option<SpeculatorType>,
    #[arg(long)]
    /// Path to the speculator model.
    pub speculator_path: Option<String>,
    #[arg(long = "speculator-tokens", default_value_t = 1)]
    /// Number of tokens to speculate.
    pub speculated_tokens: usize,
}

impl SpeculatorArgs {
    pub fn build_speculator_config(
        &self,
        model_path: &Path,
    ) -> SpeculatorConfig {
        let Some(spec_type) = &self.speculator_type else {
            return SpeculatorConfig::default();
        };

        let n = self.speculated_tokens;

        match spec_type {
            SpeculatorType::Pard => {
                let path_str = self
                    .speculator_path
                    .as_deref()
                    .expect("--speculator-path is required when --speculator-type is pard");
                let speculator =
                    NeuralSpeculator::<Metal>::new(Path::new(path_str), n, 8).expect("Failed to load PARD draft model");
                SpeculatorConfig::new(n + 1, Arc::new(speculator))
            },
            SpeculatorType::Ngram => {
                let ngram_path = match self.speculator_path.as_deref() {
                    Some(path) => path,
                    None => &self.resolve_ngram_path(model_path).to_string_lossy().into_owned(),
                };
                let speculator = NGramSpeculator::load(ngram_path).expect("Failed to load NGram speculator");
                SpeculatorConfig::new(n, Arc::new(speculator))
            },
        }
    }

    fn resolve_ngram_path(
        &self,
        model_path: &Path,
    ) -> std::path::PathBuf {
        if let Some(explicit) = self.speculator_path.as_deref() {
            return std::path::PathBuf::from(explicit);
        }

        let speculators_dir = model_path.join("speculators");
        let mut found: Vec<std::path::PathBuf> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&speculators_dir) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("model.bin");
                if candidate.exists() {
                    found.push(candidate);
                }
            }
        }

        if found.is_empty() {
            eprintln!(
                "error: no ngram speculator found in {}\n\
                 Looked for: {}/*/model.bin\n\
                 Specify a path explicitly with --speculator-path <path>",
                speculators_dir.display(),
                speculators_dir.display(),
            );
            std::process::exit(1);
        }

        if let Some(chat) =
            found.iter().find(|p| p.parent().and_then(|d| d.file_name()).map(|n| n == "chat").unwrap_or(false))
        {
            return chat.clone();
        }

        found.remove(0)
    }
}
