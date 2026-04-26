use std::{
    path::PathBuf,
    process::{Command as StdCommand, Stdio},
};

use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct Command {
    current_path: Option<PathBuf>,
    name: String,
    arguments: Vec<String>,
}

impl Command {
    pub fn new(name: &str) -> Self {
        Self {
            current_path: None,
            name: name.to_string(),
            arguments: vec![],
        }
    }

    pub fn with_current_path(
        &self,
        path: &PathBuf,
    ) -> Self {
        Self {
            current_path: Some(path.clone()),
            ..self.clone()
        }
    }

    pub fn with_argument(
        &self,
        argument: &str,
    ) -> Self {
        let mut current_arguments = self.arguments.clone();
        current_arguments.push(argument.to_string());

        Self {
            arguments: current_arguments,
            ..self.clone()
        }
    }

    pub fn with_arguments(
        &self,
        arguments: Vec<String>,
    ) -> Self {
        let mut current_arguments = self.arguments.clone();
        current_arguments.extend(arguments);

        Self {
            arguments: current_arguments,
            ..self.clone()
        }
    }

    pub fn run(self) -> Result<()> {
        let mut command = StdCommand::new(&self.name);
        if let Some(current_path) = self.current_path {
            command.current_dir(current_path);
        }
        command.args(&self.arguments);
        command.stdout(Stdio::inherit());
        command.stderr(Stdio::inherit());
        let status = command.status()?;
        if !status.success() {
            return Err(anyhow!("Command failed"));
        }
        Ok(())
    }
}

impl Command {
    pub fn rustup_setup() -> Self {
        Self::new("sh")
            .with_argument("-c")
            .with_argument("rustup --version >/dev/null 2>&1 || curl https://sh.rustup.rs -sSf | sh")
    }

    pub fn rustup_update() -> Self {
        Self::new("rustup").with_argument("update")
    }

    pub fn rustup_show() -> Self {
        Self::new("rustup").with_argument("show")
    }

    pub fn cargo_install(
        name: &str,
        version: &str,
    ) -> Self {
        Self::new("cargo")
            .with_argument("install")
            .with_argument("--locked")
            .with_argument(&format!("cargo-{name}@{}", version))
    }
}

impl Command {
    pub fn uv_setup() -> Self {
        Self::new("sh")
            .with_argument("-c")
            .with_argument("uv --version >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh")
    }

    pub fn uv_install(
        name: &str,
        version: &str,
    ) -> Self {
        Self::new("uv")
            .with_argument("tool")
            .with_argument("install")
            .with_argument("--force")
            .with_argument(&format!("{name}=={}", version))
    }

    pub fn uv_sync() -> Self {
        Self::new("uv").with_argument("sync").with_argument("--reinstall")
    }

    pub fn uv_python(code: &str) -> Self {
        Self::new("uv").with_argument("run").with_argument("python").with_argument("-c").with_argument(code)
    }

    // pub fn maturin_build(
    //     configuration: Configuration,
    //     target: &Target,
    //     manifest_path: &PathBuf,
    // ) -> Self {
    //     let mut command = StdCommand::new("maturin").arg("build");
    //     command = match configuration {
    //         Configuration::Debug => command,
    //         Configuration::Release => command.arg("--release"),
    //     };
    //     command = command
    //         .args(["--target", &target.name])
    //         .args(["--manifest-path", manifest_path.to_string_lossy().as_ref()]);
    //     Self::new(command)
    // }
}

impl Command {}
