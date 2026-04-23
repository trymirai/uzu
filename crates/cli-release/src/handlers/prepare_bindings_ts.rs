use std::process::{Command, Stdio};

use crate::{
    types::{Environment, Error},
    utilities::clone_dir_with_ignore_respect,
};

pub fn prepare_bindings_ts(environment: &Environment) -> Result<(), Error> {
    let scripts_build_path = environment.scripts().ts_build();
    if !scripts_build_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_build_path,
        });
    }

    let workspace_ts_napi_path = environment.workspace().ts_napi_path();
    if !workspace_ts_napi_path.exists() {
        return Err(Error::PathNotFound {
            path: workspace_ts_napi_path,
        });
    }

    let bindings_ts_path = environment.bindings().ts_path();
    let bindings_ts_src_path = bindings_ts_path.join("src");
    if !bindings_ts_src_path.exists() {
        return Err(Error::PathNotFound {
            path: bindings_ts_src_path,
        });
    }

    let bindings_ts_src_napi_path = bindings_ts_src_path.join("napi");
    clone_dir_with_ignore_respect(&workspace_ts_napi_path, &bindings_ts_src_napi_path)
        .map_err(|_| Error::UnableToPrepareBindingsTS)?;

    let output = Command::new("bash")
        .arg(scripts_build_path.clone())
        .arg(bindings_ts_path.clone())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToPrepareBindingsTS)?;

    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToPrepareBindingsTS);
    }

    Ok(())
}
