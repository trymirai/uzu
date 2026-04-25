use std::process::{Command, Stdio};

use crate::{
    handlers::generate_swift_extensions,
    types::{Environment, Error},
};

pub fn prepare_bindings_swift(environment: &Environment) -> Result<(), Error> {
    let scripts_build_path = environment.bindings().swift_build_script_path();
    if !scripts_build_path.exists() {
        return Err(Error::PathNotFound {
            path: scripts_build_path,
        });
    }

    let output = Command::new("bash")
        .arg(scripts_build_path.clone())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|_| Error::UnableToPrepareBindingsSwift)?;

    let status = output.status;
    if !status.success() {
        return Err(Error::UnableToPrepareBindingsSwift);
    }

    generate_swift_extensions(environment)?;

    Ok(())
}
