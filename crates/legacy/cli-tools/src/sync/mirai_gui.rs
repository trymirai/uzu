use anyhow::Result;
use regex::Regex;
use toml_edit::{DocumentMut, value};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub enum MiraiGuiSyncTask {
    PackageJson,
    CargoToml,
}

impl SyncTask for MiraiGuiSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let version = &workspace.workspace.package.version;
        match self {
            Self::PackageJson => replace_string_field(input, "version", version),
            Self::CargoToml => {
                let mut document: DocumentMut = input.parse()?;
                document["package"]["version"] = value(version);
                Ok(document.to_string())
            },
        }
    }
}

fn replace_string_field(
    input: &str,
    key: &str,
    value: &str,
) -> Result<String> {
    let regex = Regex::new(&format!(r#"("{key}"\s*:\s*)"[^"]*""#))?;
    let literal = serde_json::to_string(value)?;
    Ok(regex.replace(input, format!("$1{literal}").as_str()).into_owned())
}
