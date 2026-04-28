use std::{fs, path::PathBuf};

use anyhow::{Context, Result};

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Capability, Command, Configuration, Language},
};

pub struct PythonLanguageBackend {
    config: PlatformsConfig,
}

impl PythonLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

impl LanguageBackend for PythonLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Python
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let (zig_path, _) = Command::which("python-zig".to_string()).output()?;

        let host_target = self.config.host_target()?;
        for target in targets {
            let (_, stderr) = Command::maturin_build(target.name.clone(), target.features.clone(), configuration)
                .with_current_path(&bindings_path)
                .with_env("CARGO_ZIGBUILD_ZIG_PATH", &zig_path)
                .with_envs(self.config.required_envs_for_target(target.name.clone())?)
                .output()?;
            let wheel_path = parse_wheel_path(&stderr)?;
            if target.name == host_target {
                let envs = self.config.required_envs_for_target(target.name.clone())?;
                Command::uv_pip_install_wheel(wheel_path)
                    .with_current_path(&bindings_path)
                    .with_envs(envs.clone())
                    .run()?;
                Command::uv_python("import uzu; uzu.generate_annotations()")
                    .with_current_path(&bindings_path)
                    .with_envs(envs)
                    .run()?;
            }
        }
        Ok(())
    }

    fn test_target(
        &self,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        Command::uv_pytest().with_current_path(&bindings_path).run()
    }

    fn example_target(
        &self,
        name: &str,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let examples_path = self.config.examples_path_for_language(self.language())?;
        let name = self.language().convert_file_name(name);
        let file_path = examples_path.join(format!("{name}.py"));
        Command::uv_python_file(file_path).with_current_path(&bindings_path).run()
    }

    fn release(
        &self,
        _version: &str,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let wheels_root = paths.target_wheels_path();
        if wheels_root.exists() {
            fs::remove_dir_all(&wheels_root)?;
        }

        self.build(Configuration::Release, vec![ALL_TARGET.to_string()], Vec::<Capability>::new())?;

        let destination = paths.release_python_pypi_path();
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::create_dir_all(&destination)?;

        if !wheels_root.exists() {
            anyhow::bail!("No wheels at {}", wheels_root.display());
        }

        let mut copied = 0;
        for entry in fs::read_dir(&wheels_root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("whl") {
                fs::copy(&path, destination.join(entry.file_name()))?;
                copied += 1;
            }
        }
        if copied == 0 {
            anyhow::bail!("No `.whl` files found in {}", wheels_root.display());
        }
        Ok(())
    }
}

fn parse_wheel_path(stdout: &str) -> Result<PathBuf> {
    let line =
        stdout.lines().find(|line| line.contains("Built wheel")).context("maturin did not report a built wheel")?;
    let (_, path) = line.rsplit_once(" to ").context("Could not parse wheel path from maturin output")?;
    Ok(PathBuf::from(path.trim()))
}
