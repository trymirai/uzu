use anyhow::Result;

use crate::{
    configs::{Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Command, Configuration, Language},
};

pub struct RustLanguageBackend {
    config: PlatformsConfig,
}

impl RustLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

impl LanguageBackend for RustLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Rust
    }

    fn expects_prebuild_for_run(&self) -> bool {
        return false;
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        for target in targets {
            Command::cargo_build(paths.main_crate.clone(), target.name.clone(), target.features.clone(), configuration)
                .with_envs(self.config.required_envs_for_target(target.name.clone())?)
                .run()?;
        }
        Ok(())
    }

    fn test_target(
        &self,
        configuration: Configuration,
        target: LanguageBackendTarget,
    ) -> Result<()> {
        Command::cargo_test(target.name.clone(), target.features.clone(), configuration).run()
    }

    fn example_target(
        &self,
        name: &str,
        configuration: Configuration,
        target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let name = self.language().convert_command_name(name);
        Command::cargo_run_example(
            paths.main_crate.clone(),
            name,
            target.name.clone(),
            target.features.clone(),
            configuration,
        )
        .run()
    }
}
