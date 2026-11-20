use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;

use crate::bench::{BenchSampling, ReasoningEfforts};

#[derive(Debug, Parser)]
pub struct Args {
    pub model: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub model: String,
    pub specdec: Option<bool>,
    pub sampling: Option<BenchSampling>,
    #[expect(dead_code, reason = "callers apply the mapping before they send a request")]
    pub reasoning_efforts: Option<ReasoningEfforts>,
    #[expect(dead_code, reason = "it describes the model to those who compare results")]
    pub model_size: Option<u64>,
}
