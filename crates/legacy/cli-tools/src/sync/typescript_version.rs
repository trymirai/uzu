use anyhow::Result;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::{SyncTask, replace::replace_required},
};

/// The napi loader the TypeScript build stamps with the version, kept in step without building.
pub struct TypeScriptVersionSyncTask;

impl SyncTask for TypeScriptVersionSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let replacement = format!("${{1}}{}${{2}}", workspace.workspace.package.version);
        let output = replace_required(input, r"(bindingPackageVersion !== ')[^']*(')", &replacement)?;
        replace_required(&output, r"(version mismatch, expected )\S+( but got)", &replacement)
    }
}
