use std::path::Path;

mod docs;
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

    let python_bindings_path = paths.bindings_for_language_path(Language::Python);
    PyprojectSyncTask.run(&platforms, &workspace, &python_bindings_path.join("pyproject.toml"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &python_bindings_path.join("LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::Python]).run(
        &platforms,
        &workspace,
        &python_bindings_path.join("README.md"),
        check,
    )?;

    let swift_bindings_path = paths.bindings_for_language_path(Language::Swift);
    SwiftPackageSyncTask.run(&platforms, &workspace, &swift_bindings_path.join("Package.swift"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &swift_bindings_path.join("LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::Swift]).run(
        &platforms,
        &workspace,
        &swift_bindings_path.join("README.md"),
        check,
    )?;

    let typescript_bindings_path = paths.bindings_for_language_path(Language::TypeScript);
    PackageJsonSyncTask.run(&platforms, &workspace, &typescript_bindings_path.join("package.json"), check)?;
    JsrSyncTask.run(&platforms, &workspace, &typescript_bindings_path.join("jsr.json"), check)?;
    JsrSyncTask.run(&platforms, &workspace, &typescript_bindings_path.join("jsr.json.orig"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &typescript_bindings_path.join("LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::TypeScript]).run(
        &platforms,
        &workspace,
        &typescript_bindings_path.join("README.md"),
        check,
    )?;

    PlaygroundSyncTask.run(&platforms, &workspace, &root_path.join("apps/playground/Project.swift"), check)?;

    docs::sync_docs(&platforms, root_path, check)?;

    Ok(())
}
