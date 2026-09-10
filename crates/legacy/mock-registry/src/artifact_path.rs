use std::path::{Path, PathBuf};

pub fn artifact_path(
    destination: &Path,
    extension: &str,
) -> PathBuf {
    PathBuf::from(format!("{}.{extension}", destination.display()))
}
