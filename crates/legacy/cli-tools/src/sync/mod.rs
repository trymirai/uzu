use std::path::Path;

mod docs;
mod license;
mod mirai_gui;
mod package_json;
mod playground;
mod pyproject;
mod readme;
mod replace;
mod swift_package;
mod toolchains;
mod typescript_version;
mod wasm_tool;

use anyhow::{Context, Ok, Result, anyhow};
pub use license::LicenseSyncTask;
pub use mirai_gui::MiraiGuiSyncTask;
pub use package_json::PackageJsonSyncTask;
pub use playground::PlaygroundSyncTask;
pub use pyproject::PyprojectSyncTask;
pub use readme::ReadmeSyncTask;
pub use swift_package::SwiftPackageSyncTask;
pub use toolchains::ToolchainsSyncTask;
pub use typescript_version::TypeScriptVersionSyncTask;
pub use wasm_tool::WasmToolSyncTask;

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
        // A missing file is generated whole; any other read error stops the sync.
        let input = match std::fs::read_to_string(input_path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            input => input.with_context(|| format!("Failed to read {}", input_path.display()))?,
        };
        let output = self
            .process(platforms, workspace, &input)
            .with_context(|| format!("Failed to sync {}", input_path.display()))?;
        if check {
            if input != output {
                return Err(anyhow!("The file is out of sync: {}", input_path.display()));
            }
        } else {
            std::fs::write(input_path, output).with_context(|| format!("Failed to write {}", input_path.display()))?;
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
    PackageJsonSyncTask::Bindings(Language::TypeScript).run(
        &platforms,
        &workspace,
        &typescript_bindings_path.join("package.json"),
        check,
    )?;
    TypeScriptVersionSyncTask.run(&platforms, &workspace, &paths.napi_output_path().join("index.js"), check)?;
    LicenseSyncTask.run(&platforms, &workspace, &typescript_bindings_path.join("LICENSE"), check)?;
    ReadmeSyncTask::new(vec![Language::TypeScript]).run(
        &platforms,
        &workspace,
        &typescript_bindings_path.join("README.md"),
        check,
    )?;

    let wasm_bindings_path = paths.bindings_for_language_path(Language::Wasm);
    PackageJsonSyncTask::Bindings(Language::Wasm).run(
        &platforms,
        &workspace,
        &wasm_bindings_path.join("package.json"),
        check,
    )?;
    LicenseSyncTask.run(&platforms, &workspace, &wasm_bindings_path.join("LICENSE"), check)?;
    WasmToolSyncTask::new(typescript_bindings_path.join("src/tool.ts")).run(
        &platforms,
        &workspace,
        &wasm_bindings_path.join("src/tool.ts"),
        check,
    )?;

    PlaygroundSyncTask.run(&platforms, &workspace, &root_path.join("apps/playground/Project.swift"), check)?;

    PackageJsonSyncTask::App.run(
        &platforms,
        &workspace,
        &root_path.join("apps/uzu-pi-extension/package.json"),
        check,
    )?;

    let mirai_gui_path = root_path.join("apps/mirai-gui");
    PackageJsonSyncTask::App.run(&platforms, &workspace, &mirai_gui_path.join("package.json"), check)?;
    MiraiGuiSyncTask.run(&platforms, &workspace, &mirai_gui_path.join("desktop/tauri.macos.conf.json"), check)?;

    docs::sync_docs(&platforms, root_path, check)?;

    Ok(())
}
