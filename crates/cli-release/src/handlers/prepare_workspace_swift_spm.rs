use std::{
    fs,
    process::{Command, Stdio},
};

use crate::types::{Environment, Error};

pub fn prepare_workspace_swift_spm(
    environment: &Environment,
    version: String,
) -> Result<String, Error> {
    let scripts_build_path = environment.scripts().zip_xcframework();
    if !scripts_build_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_build_path,
        });
    }

    let framework_path = environment.bindings().swift_framework_path();
    if !framework_path.exists() {
        return Err(Error::PathNotFound {
            path: framework_path,
        });
    }

    let workspace = environment.workspace();
    let workspace_swift_spm_path = workspace.swift_spm_path();
    if workspace_swift_spm_path.exists() {
        fs::remove_dir_all(workspace_swift_spm_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceSwiftSPM)?;
    }
    fs::create_dir_all(workspace_swift_spm_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceSwiftSPM)?;

    let output = Command::new("bash")
        .arg(scripts_build_path.clone())
        .arg(framework_path.clone())
        .arg(workspace_swift_spm_path.clone())
        .arg(version.clone())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToPrepareWorkspaceSwiftSPM)?;

    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToPrepareWorkspaceSwiftSPM);
    }

    let result = String::from_utf8_lossy(&output.stdout);
    let checksum =
        result.lines().filter(|line| !line.trim().is_empty()).last().ok_or(Error::UnableToPrepareWorkspaceSwiftSPM)?;

    Ok(checksum.to_string())
}
