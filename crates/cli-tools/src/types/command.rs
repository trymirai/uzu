use std::{
    path::PathBuf,
    process::{Command as StdCommand, Stdio},
};

use anyhow::{Result, anyhow};

use crate::types::Configuration;

#[derive(Debug, Clone)]
pub struct Command {
    current_path: Option<PathBuf>,
    name: String,
    arguments: Vec<String>,
    envs: Vec<(String, String)>,
}

impl Command {
    pub fn new(name: &str) -> Self {
        Self {
            current_path: None,
            name: name.to_string(),
            arguments: vec![],
            envs: vec![],
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

    pub fn with_env(
        &self,
        key: &str,
        value: &str,
    ) -> Self {
        let mut current_envs = self.envs.clone();
        current_envs.push((key.to_string(), value.to_string()));

        Self {
            envs: current_envs,
            ..self.clone()
        }
    }

    pub fn run(self) -> Result<()> {
        let mut command = self.std_command();
        command.stdout(Stdio::inherit());
        command.stderr(Stdio::inherit());
        let status = command.status()?;
        if !status.success() {
            return Err(anyhow!("Command failed"));
        }
        Ok(())
    }

    pub fn output(self) -> Result<String> {
        let mut command = self.std_command();
        command.stderr(Stdio::inherit());
        let output = command.output()?;
        if !output.status.success() {
            return Err(anyhow!("Command failed"));
        }
        Ok(String::from_utf8(output.stdout)?.trim().to_string())
    }

    fn std_command(&self) -> StdCommand {
        let mut command = StdCommand::new(&self.name);
        if let Some(current_path) = &self.current_path {
            command.current_dir(current_path);
        }
        command.args(&self.arguments);
        for (key, value) in &self.envs {
            command.env(key, value);
        }
        command
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

    pub fn cargo_build(
        package: String,
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("cargo")
            .with_argument("build")
            .with_argument("-p")
            .with_argument(&package)
            .with_arguments(vec!["--target".to_string(), target])
            .with_argument("--no-default-features")
            .with_arguments(vec!["--features".to_string(), features.join(",")]);
        command = match configuration {
            Configuration::Debug => command,
            Configuration::Release => command.with_argument("--release"),
        };
        command
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

    pub fn maturin_build(
        manifest_path: PathBuf,
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("maturin")
            .with_argument("build")
            .with_arguments(vec!["--manifest-path".to_string(), manifest_path.to_string_lossy().to_string()])
            .with_arguments(vec!["--target".to_string(), target])
            .with_argument("--no-default-features")
            .with_arguments(vec!["--features".to_string(), features.join(",")])
            .with_argument("--strip");
        command = match configuration {
            Configuration::Debug => command,
            Configuration::Release => command.with_argument("--release"),
        };
        command
    }
}

impl Command {
    pub fn which(name: String) -> Self {
        Self::new("which").with_argument(&name)
    }
}
