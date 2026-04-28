use std::fs;

use anyhow::Result;

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Capability, Command, Configuration, Language},
    utilities::fs::copy_directory,
};

pub struct TypeScriptLanguageBackend {
    config: PlatformsConfig,
}

impl TypeScriptLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

impl LanguageBackend for TypeScriptLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::TypeScript
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let manifest_path = paths.crate_manifest_path(&paths.main_crate);
        let bindings_path = paths.bindings_for_language_path(self.language());
        let output_path = paths.napi_output_path();
        let zig_path = Command::which("python-zig".to_string()).output()?;
        let host_target = self.config.host_target()?;

        Command::pnpm_install().with_current_path(&bindings_path).run()?;

        for target in targets.iter() {
            Command::napi_build(
                manifest_path.clone(),
                target.name.clone(),
                target.features.clone(),
                configuration,
                output_path.clone(),
            )
            .with_current_path(&bindings_path)
            .with_env("CARGO_ZIGBUILD_ZIG_PATH", &zig_path)
            .with_envs(self.config.required_envs_for_target(target.name.clone())?)
            .run()?;
            if target.name == host_target {
                Command::pnpm_run("fix").with_current_path(&bindings_path).run()?;
                Command::pnpm_run("build").with_current_path(&bindings_path).run()?;
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
        Command::pnpm_jest().with_current_path(&bindings_path).run()
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
        let file_path = examples_path.join(format!("{name}.ts"));
        Command::pnpm_tsn(file_path).with_current_path(&bindings_path).run()
    }

    fn release(
        &self,
        _version: &str,
    ) -> Result<()> {
        self.build(Configuration::Release, vec![ALL_TARGET.to_string()], Vec::<Capability>::new())?;

        let paths = Paths::new()?;
        let source = paths.bindings_for_language_path(self.language()).join("dist");
        if !source.exists() {
            anyhow::bail!("Missing dist folder at {}", source.display());
        }

        let destination = paths.release_typescript_npm_path();
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        copy_directory(&source, &destination)?;
        Ok(())
    }
}
