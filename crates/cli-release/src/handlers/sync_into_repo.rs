use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use clap::ValueEnum;

use crate::types::{Environment, Error};

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum SyncSource {
    Swift,
    TS,
    Docs,
}

impl SyncSource {
    pub fn relative_path(&self) -> String {
        match self {
            SyncSource::Swift => "./".to_string(),
            SyncSource::TS => "./".to_string(),
            SyncSource::Docs => "./snippets/generated".to_string(),
        }
    }

    pub fn source_path(
        &self,
        environment: &Environment,
    ) -> PathBuf {
        match self {
            SyncSource::Swift => environment.workspace().swift_path(),
            SyncSource::TS => environment.workspace().ts_path(),
            SyncSource::Docs => environment.workspace().docs_path(),
        }
    }
}

pub fn sync_into_repo(
    environment: &Environment,
    source: SyncSource,
    repo_path: PathBuf,
) -> Result<(), Error> {
    let scripts_path = environment.scripts().sync_into_repo();
    if !scripts_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_path,
        });
    }

    let source_path = source.source_path(environment);
    if !source_path.exists() {
        return Err(Error::PathNotFound {
            path: source_path,
        });
    }

    if !repo_path.exists() {
        return Err(Error::PathNotFound {
            path: repo_path,
        });
    }

    let output = Command::new("bash")
        .arg(scripts_path.clone())
        .arg(repo_path.clone())
        .arg(source.relative_path())
        .arg(source_path.clone())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToSyncIntoRepo)?;
    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToSyncIntoRepo);
    }

    Ok(())
}
