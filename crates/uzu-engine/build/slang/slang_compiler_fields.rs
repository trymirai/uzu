use std::{env, marker::PhantomData, path::PathBuf};

use anyhow::Context;
use shader_slang::OptimizationLevel;

use crate::slang::SlangTarget;

#[derive(Clone)]
pub struct SlangCompilerFields<T: SlangTarget> {
    pub source_directory: PathBuf,
    pub output_directory: PathBuf,
    pub optimization_level: OptimizationLevel,
    _phantom: PhantomData<T>,
}

impl<T: SlangTarget> SlangCompilerFields<T> {
    pub fn new() -> anyhow::Result<Self> {
        let source_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/slang/kernel");

        println!("cargo::rerun-if-changed={}", source_directory.display());

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join(T::TARGET_NAME);

        let optimization_level = match env::var("OPT_LEVEL").context("missing OPT_LEVEL")?.as_str() {
            "0" => OptimizationLevel::Default,
            _ => OptimizationLevel::High,
        };

        Ok(Self {
            source_directory,
            output_directory,
            optimization_level,
            _phantom: PhantomData,
        })
    }
}
