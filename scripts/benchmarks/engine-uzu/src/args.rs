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
    pub specdec: bool,
    pub sampling: BenchSampling,
    #[expect(dead_code, reason = "callers apply the mapping before they send a request")]
    pub reasoning_efforts: ReasoningEfforts,
}
