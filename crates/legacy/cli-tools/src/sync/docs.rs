use std::{collections::BTreeSet, ffi::OsStr, fs, path::Path};

use anyhow::{Context, Result, anyhow};
use itertools::Itertools;
use walkdir::WalkDir;

use crate::configs::PlatformsConfig;

pub fn sync_docs(
    platforms: &PlatformsConfig,
    root_path: &Path,
    check: bool,
) -> Result<()> {
    let examples_root = root_path.join("docs").join("snippets").join("generated").join("examples");

    let mut existing: BTreeSet<_> = if examples_root.exists() {
        WalkDir::new(&examples_root)
            .into_iter()
            .map(|entry| entry.map(|entry| entry.into_path()))
            .filter_ok(|path| path.is_file() && path.file_name() != Some(OsStr::new(".DS_Store")))
            .try_collect()?
    } else {
        BTreeSet::new()
    };

    let mut check_errors = Vec::new();

    for language in platforms.languages.keys() {
        let source_root = platforms.examples_path_for_language(*language)?;
        for example_name in platforms.examples.keys() {
            let converted_name = language.convert_file_name(example_name);
            let source_path = source_root.join(format!("{converted_name}.{}", language.file_extension()));
            let body = fs::read_to_string(&source_path)
                .with_context(|| format!("Failed to read example: {}", source_path.display()))?;
            let mdx = format!("```{}\n{}\n```\n", language.code_fence(), body.trim_end());
            let path = examples_root.join(language.name()).join(format!("{example_name}.mdx"));

            if check {
                if !existing.remove(&path) {
                    check_errors.push(format!("Missing generated docs snippet: {}", path.display()));
                } else if fs::read_to_string(&path)? != mdx {
                    check_errors.push(format!("The file is out of sync: {}", path.display()));
                }
            } else {
                existing.remove(&path);
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&path, mdx)?;
            }
        }
    }

    if check {
        check_errors.extend(existing.iter().map(|path| format!("Stale generated docs snippet: {}", path.display())));
        if !check_errors.is_empty() {
            return Err(anyhow!("Docs snippets are out of sync:\n{}", check_errors.join("\n")));
        }
    } else {
        for path in existing {
            fs::remove_file(path)?;
        }
    }

    Ok(())
}
