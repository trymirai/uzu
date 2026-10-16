use anyhow::Result;
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct PiExtensionSyncTask;

impl SyncTask for PiExtensionSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let regex = Regex::new(r#"("version"\s*:\s*)"[^"]*""#)?;
        let version = serde_json::to_string(&workspace.workspace.package.version)?;
        Ok(regex.replace(input, format!("$1{version}").as_str()).into_owned())
    }
}
