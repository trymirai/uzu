use std::path::Path;

mod jsr;
mod license;
mod package_json;
mod playground;
mod pyproject;
mod readme;
mod swift_package;
mod toolchains;

use anyhow::{Ok, Result, anyhow};
pub use jsr::JsrSyncTask;
pub use license::LicenseSyncTask;
pub use package_json::PackageJsonSyncTask;
pub use playground::PlaygroundSyncTask;
pub use pyproject::PyprojectSyncTask;
pub use readme::ReadmeSyncTask;
pub use swift_package::SwiftPackageSyncTask;
pub use toolchains::ToolchainsSyncTask;

use crate::configs::{Paths, PlatformsConfig, WorkspaceManifest};

pub trait SyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String>;

    fn run(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input_path: &Path,
        check: bool,
    ) -> Result<()> {
        let input = std::fs::read_to_string(input_path).unwrap_or_default();
        let output = self.process(platforms, workspace, &input)?;
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
    use crate::types::Language;

    let paths = Paths::new()?;
    let platforms = PlatformsConfig::load()?;
    let workspace = WorkspaceManifest::load()?;
    let root_path = &paths.root_path;

    ToolchainsSyncTask.run(&platforms, &workspace, &root_path.join("rust-toolchain.toml"), check)?;
    ReadmeSyncTask::new(vec![Language::Rust, Language::Python, Language::Swift, Language::TypeScript]).run(
        &platforms,
        &workspace,
        &root_path.join("README.md"),
        check,
    )?;

    PyprojectSyncTask.run(&platforms, &workspace, &root_path.join("bindings/python/pyproject.toml"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &root_path.join("bindings/python/LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::Python]).run(
        &platforms,
        &workspace,
        &root_path.join("bindings/python/README.md"),
        check,
    )?;

    SwiftPackageSyncTask.run(&platforms, &workspace, &root_path.join("bindings/swift/Package.swift"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &root_path.join("bindings/swift/LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::Swift]).run(
        &platforms,
        &workspace,
        &root_path.join("bindings/swift/README.md"),
        check,
    )?;

    PackageJsonSyncTask.run(&platforms, &workspace, &root_path.join("bindings/typescript/package.json"), check)?;
    JsrSyncTask.run(&platforms, &workspace, &root_path.join("bindings/typescript/jsr.json"), check)?;
    JsrSyncTask.run(&platforms, &workspace, &root_path.join("bindings/typescript/jsr.json.orig"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &root_path.join("bindings/typescript/LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::TypeScript]).run(
        &platforms,
        &workspace,
        &root_path.join("bindings/typescript/README.md"),
        check,
    )?;

    PlaygroundSyncTask.run(&platforms, &workspace, &root_path.join("apps/playground/Project.swift"), check)?;

    Ok(())
}
