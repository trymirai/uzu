use std::{
    fs,
    process::{Command, Stdio},
};

use crate::{
    handlers::{prepare_docs, prepare_platform},
    types::{Environment, Error, Language},
    utilities::{
        clone_dir_with_ignore_respect, extract_examples, extract_snippets, remove_ignored_entities_from_directory,
        update_json_field, update_readme,
    },
};

pub fn prepare_workspace_ts(
    environment: &Environment,
    version: String,
) -> Result<(), Error> {
    let scripts_build_path = environment.scripts().ts_build();
    if !scripts_build_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_build_path,
        });
    }

    let workspace = environment.workspace();
    let workspace_ts_path = workspace.ts_path();
    if workspace_ts_path.exists() {
        fs::remove_dir_all(workspace_ts_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTS)?;
    }
    fs::create_dir_all(workspace_ts_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTS)?;

    let bindings_ts_path = environment.bindings().ts_path();
    if !bindings_ts_path.exists() {
        return Err(Error::PathNotFound {
            path: bindings_ts_path,
        });
    }

    clone_dir_with_ignore_respect(&bindings_ts_path, &workspace_ts_path)?;

    update_json_field(workspace_ts_path.join("package.json"), "version".to_string(), version.clone())?;

    update_json_field(workspace_ts_path.join("jsr.json"), "version".to_string(), version.clone())?;

    let examples_path = workspace_ts_path.join("examples");
    let snippets = extract_snippets(examples_path.clone())?;
    let examples = extract_examples(examples_path.clone(), workspace_ts_path.clone())?;

    let readme_path = workspace_ts_path.join("README.md");
    update_readme(readme_path.clone(), snippets.clone(), examples.clone(), version.clone())?;

    let output = Command::new("bash")
        .arg(scripts_build_path.clone())
        .arg(workspace_ts_path.clone())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToPrepareWorkspaceTS)?;
    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToPrepareWorkspaceTS);
    }

    let workspace_ts_npm_path = workspace.ts_npm_path();
    if workspace_ts_npm_path.exists() {
        fs::remove_dir_all(workspace_ts_npm_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTSNPM)?;
    }
    fs::create_dir_all(workspace_ts_npm_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTSNPM)?;

    let dist_path = workspace_ts_path.join("dist");
    if !dist_path.exists() {
        return Err(Error::PathNotFound {
            path: dist_path,
        });
    }
    clone_dir_with_ignore_respect(&dist_path, &workspace_ts_npm_path)?;

    remove_ignored_entities_from_directory(workspace_ts_path.clone())?;

    prepare_docs(environment, version.clone(), Language::TS, snippets.clone(), examples.clone())?;

    prepare_platform(environment, Language::TS, examples.clone())?;

    Ok(())
}
