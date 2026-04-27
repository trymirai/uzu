use anyhow::Result;

use crate::{
    configs::{Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Command, Configuration, Language},
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
        let manifest_path = paths.crate_manifest_path(&paths.main_crate);
        let bindings_path = paths.bindings_for_language_path(self.language());
        let zig_path = Command::which("python-zig".to_string()).output()?;

        let host_target = self.config.host_target()?;
        for target in targets {
            Command::maturin_build(manifest_path.clone(), target.name.clone(), target.features.clone(), configuration)
                .with_current_path(&bindings_path)
                .with_env("CARGO_ZIGBUILD_ZIG_PATH", &zig_path)
                .with_envs(self.config.required_envs_for_target(target.name.clone())?)
                .run()?;
            if target.name == host_target {
                Command::uv_sync().with_current_path(&bindings_path).run()?;
                Command::uv_python("import uzu; uzu.generate_annotations()").with_current_path(&bindings_path).run()?;
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
}
