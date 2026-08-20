use std::fs;

use anyhow::Result;

use crate::{
    configs::{Paths, PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct LicenseSyncTask;

impl SyncTask for LicenseSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        _workspace: &WorkspaceManifest,
        _input: &str,
    ) -> Result<String> {
        let path = Paths::new()?.root_path.join("LICENSE");
        let body = fs::read_to_string(&path)?;
        Ok(body)
    }
}
