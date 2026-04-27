use std::path::Path;

mod jsr;
mod package_json;
mod playground;
mod pyproject;
mod swift_package;
mod toolchains;

use anyhow::{Ok, Result, anyhow};
pub use jsr::JsrSyncTask;
pub use package_json::PackageJsonSyncTask;
pub use playground::PlaygroundSyncTask;
pub use pyproject::PyprojectSyncTask;
pub use swift_package::SwiftPackageSyncTask;
pub use toolchains::ToolchainsSyncTask;

use crate::configs::{Paths, PlatformsConfig, WorkspaceManifest};

pub trait SyncTask {
    fn process(
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String>;

    fn run(
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input_path: &Path,
        check: bool,
    ) -> Result<()> {
        let input = std::fs::read_to_string(input_path)?;
        let output = Self::process(platforms, workspace, &input)?;
        if check {
            if input != output {
                return Err(anyhow!("The file is out of sync: {}", input_path.display()));
            }
        } else {
            std::fs::write(input_path, output)?;
        }
        Ok(())
    }
}

pub fn run_sync(check: bool) -> Result<()> {
    let paths = Paths::new()?;
    let platforms = PlatformsConfig::load()?;
    let workspace = WorkspaceManifest::load()?;
    let root_path = &paths.root_path;

    ToolchainsSyncTask::run(&platforms, &workspace, &root_path.join("rust-toolchain.toml"), check)?;

    PyprojectSyncTask::run(&platforms, &workspace, &root_path.join("bindings/python/pyproject.toml"), check)?;

    PackageJsonSyncTask::run(&platforms, &workspace, &root_path.join("bindings/typescript/package.json"), check)?;
    JsrSyncTask::run(&platforms, &workspace, &root_path.join("bindings/typescript/jsr.json"), check)?;
    JsrSyncTask::run(&platforms, &workspace, &root_path.join("bindings/typescript/jsr.json.orig"), check)?;

    SwiftPackageSyncTask::run(&platforms, &workspace, &root_path.join("bindings/swift/Package.swift"), check)?;

    PlaygroundSyncTask::run(&platforms, &workspace, &root_path.join("apps/playground/Project.swift"), check)?;

    Ok(())
}
