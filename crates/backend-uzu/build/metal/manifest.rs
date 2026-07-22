//! Dump of every instantiated `(kernel, axis-bindings)` tuple.
//!
//! The VARIANTS/CONSTRAINT surface decides which kernels get compiled, so any refactor
//! of it has to be proven not to change that set. Diffing this manifest before and after
//! is that proof.

use std::{fmt::Write as _, fs, path::Path};

use anyhow::Context;
use itertools::Itertools;

use super::{ast::MetalKernelInfo, wrapper::accepted_variants};
use crate::common::enum_paths::EnumPaths;

/// Renders the sorted manifest for every kernel, keyed by source path so identically
/// named kernels in different files stay distinguishable.
pub fn render<'a>(
    kernels: impl IntoIterator<Item = (&'a Path, &'a MetalKernelInfo)>,
    enum_paths: &EnumPaths,
) -> anyhow::Result<String> {
    let mut lines: Vec<String> = Vec::new();

    for (source, kernel) in kernels {
        for type_variant in accepted_variants(kernel, enum_paths)? {
            let bindings = type_variant.iter().flatten().map(|(name, value)| format!("{name}={value}")).join(" ");
            let mut line = format!("{}\t{}", source.display(), kernel.name);
            if !bindings.is_empty() {
                let _ = write!(line, "\t{bindings}");
            }
            lines.push(line);
        }
    }

    lines.sort();
    Ok(lines.join("\n") + "\n")
}

pub fn write(
    manifest: &str,
    path: &Path,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    fs::write(path, manifest).with_context(|| format!("cannot write manifest {}", path.display()))
}
