use anyhow::Result;
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct JsrSyncTask;

impl SyncTask for JsrSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let package = &workspace.workspace.package;
        let regex = Regex::new(r#"("version"\s*:\s*)"[^"]*""#)?;
        let literal = serde_json::to_string(&package.version)?;
        let replacement = format!("$1{literal}");
        Ok(regex.replace(input, replacement.as_str()).into_owned())
    }
}
