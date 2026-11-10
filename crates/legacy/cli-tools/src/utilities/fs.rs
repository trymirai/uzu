use std::{fs, path::Path};

use anyhow::Result;

pub fn copy_directory(
    source: &Path,
    destination: &Path,
) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let target = destination.join(entry.file_name());
        if path.is_dir() {
            copy_directory(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Removes `path` if it exists and creates it empty.
pub fn recreate_directory(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir_all(path)?;
    Ok(())
}

/// Removes every file with `extension` under `directory`, recursively.
pub fn remove_files_with_extension(
    directory: &Path,
    extension: &str,
) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            remove_files_with_extension(&path, extension)?;
        } else if path.extension().is_some_and(|candidate| candidate == extension) {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}
