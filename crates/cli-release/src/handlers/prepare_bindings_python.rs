use std::process::{Command, Stdio};

use crate::types::{Environment, Error};

pub fn prepare_bindings_python(environment: &Environment) -> Result<(), Error> {
    let bindings_python_path = environment.bindings().python_path();
    if !bindings_python_path.exists() {
        return Err(Error::PathNotFound {
            path: bindings_python_path,
        });
    }

    let manifest_path = environment.package_uzu().root_path.join("Cargo.toml");
    if !manifest_path.exists() {
        return Err(Error::PathNotFound {
            path: manifest_path,
        });
    }

    let build_status = Command::new("maturin")
        .args(["build", "--release", "--features", "bindings-pyo3", "--strip", "--manifest-path"])
        .arg(&manifest_path)
        .current_dir(&bindings_python_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|_| Error::UnableToPrepareBindingsPython)?;
    if !build_status.success() {
        return Err(Error::UnableToPrepareBindingsPython);
    }

    let sync_status = Command::new("uv")
        .args(["sync", "--reinstall-package", "uzu"])
        .current_dir(&bindings_python_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|_| Error::UnableToPrepareBindingsPython)?;
    if !sync_status.success() {
        return Err(Error::UnableToPrepareBindingsPython);
    }

    let annotate_status = Command::new("uv")
        .arg("run")
        .arg("--directory")
        .arg(&bindings_python_path)
        .args(["python", "-c", "import uzu; uzu.generate_annotations()"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|_| Error::UnableToPrepareBindingsPython)?;
    if !annotate_status.success() {
        return Err(Error::UnableToPrepareBindingsPython);
    }

    Ok(())
}
