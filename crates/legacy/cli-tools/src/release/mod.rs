mod binaries;
mod platform;
mod version;

use std::{fs, path::Path};

use anyhow::{Result, anyhow};
pub use version::bump_workspace_version;

use crate::{
    configs::{Paths, PlatformsConfig, WorkspaceManifest},
    languages::{
        LanguageBackend, PythonLanguageBackend, RustLanguageBackend, SwiftLanguageBackend, TypeScriptLanguageBackend,
    },
    sync::run_sync,
    types::Language,
};

pub fn run_release(
    version: &str,
    release_path: &Path,
) -> Result<()> {
    let paths = Paths::new()?;

    bump_workspace_version(version)?;
    run_sync(false)?;

    let platforms = PlatformsConfig::load()?;
    let workspace = WorkspaceManifest::load()?;
    if workspace.workspace.package.version != version {
        return Err(anyhow!("Workspace version mismatch after sync"));
    }

    fs::create_dir_all(release_path)?;

    platform::stage_platform(&paths, &platforms, release_path)?;
    binaries::stage_binaries(&paths, &platforms, release_path)?;

    for language in platforms.languages.keys() {
        let backend = backend_for_language(*language, platforms.clone());
        backend.release(version, release_path)?;
    }

    Ok(())
}

fn backend_for_language(
    language: Language,
    config: PlatformsConfig,
) -> Box<dyn LanguageBackend> {
    match language {
        Language::Rust => Box::new(RustLanguageBackend::new(config)),
        Language::Python => Box::new(PythonLanguageBackend::new(config)),
        Language::Swift => Box::new(SwiftLanguageBackend::new(config)),
        Language::TypeScript => Box::new(TypeScriptLanguageBackend::new(config)),
    }
}
