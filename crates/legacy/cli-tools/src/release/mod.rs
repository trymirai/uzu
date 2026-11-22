mod binaries;
mod platform;
mod version;

use std::{cmp::Ordering, fs, path::Path};

use anyhow::{Result, anyhow};
use semver::Version;
pub use version::{bump_published_version, bump_workspace_version};

use crate::{
    configs::{Paths, PlatformsConfig, WorkspaceManifest},
    languages::{
        KotlinLanguageBackend, LanguageBackend, PythonLanguageBackend, RustLanguageBackend, SwiftLanguageBackend,
        TypeScriptLanguageBackend, WasmLanguageBackend,
    },
    sync::run_sync,
    types::{Command, Language},
};

pub fn run_version(version: &str) -> Result<()> {
    let paths = Paths::new()?;

    let workspace_version = WorkspaceManifest::load()?.workspace.package.version;
    match Version::parse(version)?.cmp(&Version::parse(&workspace_version)?) {
        Ordering::Less => return Err(anyhow!("The workspace is already at version {workspace_version}")),
        Ordering::Equal => return Ok(()),
        Ordering::Greater => {},
    }

    bump_workspace_version(version)?;
    run_sync(false)?;
    Command::new("cargo")
        .with_arguments(["update", "--workspace", "--manifest-path"].map(String::from).to_vec())
        .with_argument(&paths.workspace_manifest_path().to_string_lossy())
        .run()
}

pub fn run_release(
    version: &str,
    release_path: &Path,
) -> Result<()> {
    let paths = Paths::new()?;

    let workspace_version = WorkspaceManifest::load()?.workspace.package.version;
    if workspace_version != version {
        return Err(anyhow!(
            "The workspace is at version {workspace_version}: run `cargo tools version {version}` first"
        ));
    }

    // Before the build, which packs the READMEs into the packages.
    bump_published_version(version)?;
    run_sync(false)?;

    let platforms = PlatformsConfig::load()?;

    fs::create_dir_all(release_path)?;

    platform::stage_platform(&paths, &platforms, release_path)?;
    binaries::stage_binaries(&paths, &platforms, release_path)?;

    for language in platforms.languages.keys() {
        // Not released until the WebGPU kernels land: the package would install but couldn't run a model.
        if *language == Language::Wasm {
            continue;
        }
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
        Language::Wasm => Box::new(WasmLanguageBackend::new(config)),
        Language::Kotlin => Box::new(KotlinLanguageBackend::new(config)),
    }
}
