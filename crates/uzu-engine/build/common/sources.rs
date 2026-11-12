use std::path::{Path, PathBuf};

use anyhow::Context;
use walkdir::WalkDir;

pub fn source_files(
    directory: &Path,
    extension: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    anyhow::ensure!(directory.is_dir(), "source directory {} does not exist", directory.display());
    let mut files = Vec::new();
    for entry in WalkDir::new(directory) {
        let entry = entry.with_context(|| format!("cannot walk {}", directory.display()))?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|found| found == extension) {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}
