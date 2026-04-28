mod docs;
mod platform;
mod version;

use std::fs;

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

pub fn run_release(version: &str) -> Result<()> {
    let paths = Paths::new()?;

    bump_workspace_version(version)?;
    run_sync(false)?;

    let platforms = PlatformsConfig::load()?;
    let workspace = WorkspaceManifest::load()?;
    if workspace.workspace.package.version != version {
        return Err(anyhow!("Workspace version mismatch after sync"));
    }

    let release_path = paths.release_workspace_path();
    if release_path.exists() {
        fs::remove_dir_all(&release_path)?;
    }
    fs::create_dir_all(&release_path)?;

    platform::stage_platform(&paths, &platforms)?;
    docs::stage_docs(&paths, &platforms)?;

    for language in platforms.languages.keys() {
        let backend = backend_for_language(*language, platforms.clone());
        backend.release(version)?;
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
