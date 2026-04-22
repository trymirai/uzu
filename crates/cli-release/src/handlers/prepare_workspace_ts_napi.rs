use std::{
    fs,
    process::{Command, Stdio},
};

use crate::types::{Environment, Error};

pub fn prepare_workspace_ts_napi(environment: &Environment) -> Result<(), Error> {
    let scripts_build_path = environment.scripts().ts_napi_build();
    if !scripts_build_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_build_path,
        });
    }

    let bindings_ts_napi_path = environment.bindings().ts_napi_path();
    if !bindings_ts_napi_path.exists() {
        return Err(Error::PathNotFound {
            path: bindings_ts_napi_path,
        });
    }

    let crates_uzu_manifest_path = environment.package_uzu().manifest_path();
    if !crates_uzu_manifest_path.exists() {
        return Err(Error::PathNotFound {
            path: crates_uzu_manifest_path,
        });
    }

    let workspace = environment.workspace();
    let workspace_ts_napi_path = workspace.ts_napi_path();
    if workspace_ts_napi_path.exists() {
        fs::remove_dir_all(workspace_ts_napi_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTSNAPI)?;
    }
    fs::create_dir_all(workspace_ts_napi_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceTSNAPI)?;

    let output = Command::new("bash")
        .arg(scripts_build_path.clone())
        .arg(bindings_ts_napi_path.clone())
        .arg(crates_uzu_manifest_path.clone())
        .arg(workspace_ts_napi_path.clone())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToPrepareWorkspaceTSNAPI)?;

    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToPrepareWorkspaceTSNAPI);
    }

    let expected_paths = vec![workspace.ts_napi_index_d_ts(), workspace.ts_napi_uzu_node()];
    for path in expected_paths {
        if !path.exists() {
            return Err(Error::PathNotFound {
                path: path,
            });
        }
    }

    fs::rename(workspace.ts_napi_index_d_ts(), workspace.ts_napi_uzu_d_ts())
        .map_err(|_| Error::UnableToPrepareWorkspaceTSNAPI)?;

    Ok(())
}
