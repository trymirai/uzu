use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use heck::ToKebabCase;
use ignore::WalkBuilder;

use crate::types::Error;

pub fn clone_dir_with_ignore_respect(
    src: &Path,
    dst: &Path,
) -> Result<(), Error> {
    fs::create_dir_all(dst).map_err(|_| Error::UnableToCloneDirectory)?;

    let walker = WalkBuilder::new(src).standard_filters(true).parents(false).hidden(false).build();

    for result in walker {
        let entry = result.map_err(|_| Error::UnableToCloneDirectory)?;
        let path = entry.path();

        if path == src {
            continue;
        }

        let relative = path.strip_prefix(src).map_err(|_| Error::UnableToCloneDirectory)?;
        let dst_path = dst.join(relative);

        if path.is_dir() {
            fs::create_dir_all(&dst_path).map_err(|_| Error::UnableToCloneDirectory)?;
        } else if path.is_file() {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent).map_err(|_| Error::UnableToCloneDirectory)?;
            }
            fs::copy(path, &dst_path).map_err(|_| Error::UnableToCloneDirectory)?;
        }
    }

    let ignore_path: PathBuf = dst.join(".ignore");
    if ignore_path.exists() {
        fs::remove_file(&ignore_path).map_err(|_| Error::UnableToCloneDirectory)?;
    }

    Ok(())
}

pub fn remove_ignored_entities_from_directory(directory_path: PathBuf) -> Result<(), Error> {
    // Build a walker that respects ignore rules – these are the entries we KEEP
    let respected_walker =
        WalkBuilder::new(&directory_path).standard_filters(true).hidden(false).follow_links(false).build();

    let mut kept_paths: HashSet<PathBuf> = HashSet::new();
    for result in respected_walker {
        let directory_entry = result.map_err(|_| Error::UnableToRemoveIgnoredEntities)?;
        let entry_path = directory_entry.path();

        // Keep the root directory, but don't store it in the set to avoid accidental deletion logic
        if entry_path == directory_path {
            continue;
        }
        kept_paths.insert(entry_path.to_path_buf());
    }

    // Build a walker that does NOT respect ignore rules – this sees EVERYTHING
    let all_walker =
        WalkBuilder::new(&directory_path).standard_filters(false).hidden(false).follow_links(false).build();

    // Partition items to remove into files and directories.
    // We remove files first, then directories deepest-first to avoid ENOTEMPTY.
    let mut files_to_remove: Vec<PathBuf> = Vec::new();
    let mut directories_to_remove: Vec<PathBuf> = Vec::new();

    for result in all_walker {
        let directory_entry = result.map_err(|_| Error::UnableToRemoveIgnoredEntities)?;
        let entry_path = directory_entry.path();

        if entry_path == directory_path {
            continue; // never remove the root
        }

        // If not in kept set, it’s ignored -> schedule for removal
        if !kept_paths.contains(entry_path) {
            if entry_path.is_file() {
                files_to_remove.push(entry_path.to_path_buf());
            } else if entry_path.is_dir() {
                directories_to_remove.push(entry_path.to_path_buf());
            }
        }
    }

    // Remove files first
    for file_path in files_to_remove {
        fs::remove_file(&file_path).map_err(|_| Error::UnableToRemoveIgnoredEntities)?;
    }

    // Then remove directories, deepest-first
    directories_to_remove.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for directory_to_remove in directories_to_remove {
        fs::remove_dir_all(&directory_to_remove).map_err(|_| Error::UnableToRemoveIgnoredEntities)?;
    }

    Ok(())
}

pub fn relative_path(
    target: PathBuf,
    base: PathBuf,
) -> Option<PathBuf> {
    let target_iter = target.components();
    let base_iter = base.components();

    let target_components: Vec<_> = target_iter.collect();
    let base_components: Vec<_> = base_iter.collect();

    let common_len = target_components.iter().zip(&base_components).take_while(|(a, b)| a == b).count();

    let mut result = PathBuf::new();
    for _ in base_components.iter().skip(common_len) {
        result.push("..");
    }

    for comp in target_components.iter().skip(common_len) {
        result.push(comp.as_os_str());
    }

    Some(result)
}

pub fn kebab_filename(path: &PathBuf) -> Option<String> {
    path.file_stem().and_then(|stem| stem.to_str()).map(|stem| stem.to_kebab_case())
}
