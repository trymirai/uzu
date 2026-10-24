mod binaries;
mod platform;
mod version;

use std::{fs, path::Path};

use anyhow::{Result, anyhow};
pub use version::bump_workspace_version;

use crate::{
    configs::{Paths, PlatformsConfig, WorkspaceManifest},
    sync::run_sync,
    types::Command,
};

pub fn run_release(
    version: &str,
    release_path: &Path,
) -> Result<()> {
    let paths = Paths::new()?;

    bump_workspace_version(version)?;
    run_sync(false)?;
    Command::new("cargo")
        .with_arguments(["update", "--workspace", "--manifest-path"].map(String::from).to_vec())
        .with_argument(&paths.workspace_manifest_path().to_string_lossy())
        .run()?;

    let platforms = PlatformsConfig::load()?;
    let workspace = WorkspaceManifest::load()?;
    if workspace.workspace.package.version != version {
        return Err(anyhow!("Workspace version mismatch after sync"));
    }

    fs::create_dir_all(release_path)?;

    platform::stage_platform(&paths, &platforms, release_path)?;
    binaries::stage_binaries(&paths, &platforms, release_path)?;

    for language in platforms.languages.keys() {
        language.backend(platforms.clone()).release(version, release_path)?;
    }

    Ok(())
}
