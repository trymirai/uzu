//! The directory walk [`igata::gpu_types`] deliberately does not do.

use std::{env, fs, path::PathBuf};

use anyhow::Context;
use igata::gpu_types::{GpuTypeFile, GpuTypes};
use walkdir::WalkDir;

pub fn scan() -> anyhow::Result<GpuTypes> {
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("Missing CARGO_MANIFEST_DIR")?)
        .join("src/backends/common/gpu_types");

    let mut sources: Vec<PathBuf> = WalkDir::new(&src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_type().is_file().then(|| e.into_path()))
        .filter(|e| e.extension().and_then(|s| s.to_str()) == Some("rs") && e.file_stem() != Some("mod".as_ref()))
        .collect();

    sources.sort();

    sources
        .into_iter()
        .map(|source| {
            let name = source.file_stem().context("No file stem")?.to_string_lossy();
            let text = fs::read_to_string(&source).context("Cannot read source")?;
            GpuTypeFile::parse(&name, &text)
                .with_context(|| format!("Failed to scan {}", source.as_os_str().to_str().unwrap()))
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .map(GpuTypes::new)
}
