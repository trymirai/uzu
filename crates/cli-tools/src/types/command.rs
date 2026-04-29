use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Command as StdCommand, Stdio},
    thread,
};

use anyhow::{Result, anyhow};
use indexmap::IndexMap;

use crate::types::Configuration;

#[derive(Debug, Clone)]
pub struct Command {
    current_path: Option<PathBuf>,
    name: String,
    arguments: Vec<String>,
    envs: IndexMap<String, String>,
}

impl Command {
    pub fn new(name: &str) -> Self {
        Self {
            current_path: None,
            name: name.to_string(),
            arguments: vec![],
            envs: IndexMap::new(),
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
        current_envs.insert(key.to_string(), value.to_string());

        Self {
            envs: current_envs,
            ..self.clone()
        }
    }

    pub fn with_envs(
        &self,
        envs: IndexMap<String, String>,
    ) -> Self {
        let mut current_envs = self.envs.clone();
        current_envs.extend(envs);

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

    pub fn output(self) -> Result<(String, String)> {
        let mut command = self.std_command();
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());

        let mut child = command.spawn()?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Missing stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("Missing stderr"))?;

        let stdout_thread = thread::spawn(move || -> Result<String> {
            let mut buffer = String::new();
            for line in BufReader::new(stdout).lines() {
                let line = line?;
                println!("{line}");
                buffer.push_str(&line);
                buffer.push('\n');
            }
            Ok(buffer)
        });
        let stderr_thread = thread::spawn(move || -> Result<String> {
            let mut buffer = String::new();
            for line in BufReader::new(stderr).lines() {
                let line = line?;
                eprintln!("{line}");
                buffer.push_str(&line);
                buffer.push('\n');
            }
            Ok(buffer)
        });

        let stdout = stdout_thread.join().map_err(|_| anyhow!("stdout thread panicked"))??;
        let stderr = stderr_thread.join().map_err(|_| anyhow!("stderr thread panicked"))??;
        let status = child.wait()?;
        if !status.success() {
            return Err(anyhow!("Command failed"));
        }
        Ok((stdout.trim().to_string(), stderr.trim().to_string()))
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

    pub fn cargo_test(
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("cargo")
            .with_argument("test")
            .with_arguments(vec!["--target".to_string(), target])
            .with_argument("--no-default-features")
            .with_arguments(vec!["--features".to_string(), features.join(",")]);
        command = match configuration {
            Configuration::Debug => command,
            Configuration::Release => command.with_argument("--release"),
        };
        command
    }

    pub fn cargo_run_example(
        package: String,
        name: String,
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("cargo")
            .with_argument("run")
            .with_argument("-p")
            .with_argument(&package)
            .with_arguments(vec!["--example".to_string(), name])
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

    pub fn uv_venv() -> Self {
        Self::new("uv").with_argument("venv")
    }

    pub fn uv_pip_install_wheel(path: PathBuf) -> Self {
        Self::new("uv")
            .with_argument("pip")
            .with_argument("install")
            .with_argument("--force-reinstall")
            .with_argument("--no-deps")
            .with_argument(&path.to_string_lossy())
    }

    pub fn uv_python(code: &str) -> Self {
        Self::new("uv")
            .with_argument("run")
            .with_argument("--no-sync")
            .with_argument("python")
            .with_argument("-c")
            .with_argument(code)
    }

    pub fn uv_pytest() -> Self {
        Self::new("uv")
            .with_argument("run")
            .with_arguments(vec!["--extra".to_string(), "dev".to_string()])
            .with_argument("pytest")
    }

    pub fn uv_python_file(path: PathBuf) -> Self {
        Self::new("uv")
            .with_argument("run")
            .with_arguments(vec!["--extra".to_string(), "examples".to_string()])
            .with_argument("python")
            .with_argument(&path.to_string_lossy())
    }

    pub fn maturin_build(
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("maturin")
            .with_argument("build")
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
    pub fn pnpm_setup() -> Self {
        Self::new("sh")
            .with_argument("-c")
            .with_argument("pnpm --version >/dev/null 2>&1 || curl -fsSL https://get.pnpm.io/install.sh | sh")
    }

    pub fn pnpm_install() -> Self {
        Self::new("pnpm").with_argument("install")
    }

    pub fn pnpm_run(script: &str) -> Self {
        Self::new("pnpm").with_argument("run").with_argument(script)
    }

    pub fn pnpm_exec() -> Self {
        Self::new("pnpm").with_argument("exec")
    }

    pub fn pnpm_jest() -> Self {
        Self::pnpm_exec().with_argument("jest")
    }

    pub fn pnpm_tsn(path: PathBuf) -> Self {
        Self::pnpm_run("tsn").with_argument(&path.to_string_lossy())
    }

    pub fn napi_build(
        manifest_path: PathBuf,
        target: String,
        features: Vec<String>,
        configuration: Configuration,
        output_path: PathBuf,
    ) -> Self {
        let cross_flag = if target.contains("linux-gnu") {
            "--use-napi-cross"
        } else {
            "-x"
        };
        let mut command = Self::pnpm_exec()
            .with_argument("napi")
            .with_argument("build")
            .with_arguments(vec!["--manifest-path".to_string(), manifest_path.to_string_lossy().to_string()])
            .with_arguments(vec!["--target".to_string(), target])
            .with_argument("--no-default-features")
            .with_arguments(vec!["--features".to_string(), features.join(",")])
            .with_argument("--platform")
            .with_argument("--output-dir")
            .with_argument(&output_path.to_string_lossy())
            .with_argument(cross_flag);
        command = match configuration {
            Configuration::Debug => command,
            Configuration::Release => command.with_argument("--release"),
        };
        command
    }
}

impl Command {
    pub fn cargo_swift_package(
        name: String,
        target: String,
        features: Vec<String>,
        configuration: Configuration,
    ) -> Self {
        let mut command = Self::new("cargo")
            .with_argument("swift")
            .with_argument("package")
            .with_arguments(vec!["--name".to_string(), name.clone()])
            .with_arguments(vec!["--xcframework-name".to_string(), name.clone()])
            .with_arguments(vec!["--target".to_string(), target])
            .with_argument("--no-default-features")
            .with_arguments(vec!["--features".to_string(), features.join(",")])
            .with_argument("-y");
        command = match configuration {
            Configuration::Debug => command,
            Configuration::Release => command.with_argument("--release"),
        };
        command
    }

    pub fn xcodebuild() -> Self {
        Self::new("xcodebuild")
    }

    pub fn xcodebuild_first_launch() -> Self {
        Self::xcodebuild().with_argument("-runFirstLaunch")
    }

    pub fn xcodebuild_download_metal_toolchain() -> Self {
        Self::xcodebuild().with_arguments(vec!["-downloadComponent".to_string(), "MetalToolchain".to_string()])
    }

    pub fn xcodebuild_create_xcframework(
        slice_libraries_with_headers_paths: Vec<(PathBuf, PathBuf)>,
        output_path: PathBuf,
    ) -> Self {
        let mut command = Self::xcodebuild().with_argument("-create-xcframework");
        for (library_path, headers_path) in slice_libraries_with_headers_paths {
            command = command.with_arguments(vec!["-library".to_string(), library_path.to_string_lossy().to_string()]);
            command = command.with_arguments(vec!["-headers".to_string(), headers_path.to_string_lossy().to_string()]);
        }
        command.with_arguments(vec!["-output".to_string(), output_path.to_string_lossy().to_string()])
    }

    pub fn swift_test() -> Self {
        Self::new("swift").with_argument("test")
    }

    pub fn swift_compute_checksum(path: PathBuf) -> Self {
        Self::new("swift")
            .with_argument("package")
            .with_argument("compute-checksum")
            .with_argument(&path.to_string_lossy())
    }

    pub fn swift_run_example(name: String) -> Self {
        Self::new("swift").with_argument("run").with_argument("examples").with_argument(&name)
    }

    pub fn codesign_adhoc(path: PathBuf) -> Self {
        Self::new("codesign")
            .with_argument("--force")
            .with_argument("--sign")
            .with_argument("-")
            .with_argument(&path.to_string_lossy())
    }
}

impl Command {
    pub fn which(name: String) -> Self {
        Self::new("which").with_argument(&name)
    }

    pub fn git_status_porcelain() -> Self {
        Self::new("git").with_argument("status").with_argument("--porcelain")
    }

    pub fn zip_directory(
        source: PathBuf,
        output: PathBuf,
    ) -> Self {
        Self::new("zip")
            .with_argument("-r")
            .with_argument(&output.to_string_lossy())
            .with_argument(&source.to_string_lossy())
    }
}
