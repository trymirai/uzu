use std::{
    io,
    path::{Path, PathBuf},
};

/// Whether the path is an operating-system directory alias rather than a
/// user-controlled symlink.
///
/// Apple platforms expose `/var`, `/tmp` and `/etc` as symlinks into
/// `/private`, so application container paths legitimately traverse them.
pub(crate) fn is_platform_path_alias(path: &Path) -> bool {
    #[cfg(target_vendor = "apple")]
    {
        matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = path;
        false
    }
}

/// The first ancestor of `path` that is a symlink we must not write through.
///
/// Walking stops at the first component that does not exist yet, so this is
/// safe to call for paths that are about to be created.
pub(crate) async fn first_symlink_ancestor(path: &Path) -> io::Result<Option<PathBuf>> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Ok(Some(current));
            },
            Ok(_) => {},
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

/// Blocking form of [`first_symlink_ancestor`], for callers already running on
/// a blocking thread such as Apple delegate callbacks.
pub(crate) fn first_symlink_ancestor_blocking(path: &Path) -> io::Result<Option<PathBuf>> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Ok(Some(current));
            },
            Ok(_) => {},
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}
