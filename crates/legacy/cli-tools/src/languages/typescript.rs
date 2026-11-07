use std::{fs, path::Path};

use anyhow::Result;

use crate::{
    configs::{ALL_TARGET, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Command, Configuration, Language},
    utilities::fs::{copy_directory, recreate_directory, remove_files_with_extension},
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
            .with_envs(self.config.required_envs_for_target(target.name.clone())?)
            .run()?;
            Command::pnpm_run("fix").with_current_path(&bindings_path).run()?;
            Command::pnpm_run("build").with_current_path(&bindings_path).run()?;
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
        release_path: &Path,
    ) -> Result<()> {
        self.build(Configuration::Release, vec![ALL_TARGET.to_string()])?;

        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let source = bindings_path.join("dist");
        if !source.exists() {
            anyhow::bail!("Missing dist folder at {}", source.display());
        }

        // The umbrella package: the loader requires the binary for the platform it runs on from that
        // platform's package, which package managers install from `optionalDependencies` when its
        // `os` and `cpu` match.
        let destination = release_path.join("typescript-npm");
        recreate_directory(&destination)?;
        copy_directory(&source, &destination)?;
        remove_files_with_extension(&destination, "node")?;

        // A package per `napi.targets` entry holding its binary, published before the umbrella package.
        let platforms_path = release_path.join("typescript-npm-platforms");
        recreate_directory(&platforms_path)?;
        Command::napi_create_npm_dirs(&platforms_path).with_current_path(&bindings_path).run()?;
        copy_platform_binaries(&paths.napi_output_path(), &platforms_path)
    }
}

/// Copies each platform package's binary (its `main`) from the napi build output. Fails when one
/// is missing, which would publish a package that installs but can't load. (`napi artifacts`
/// would do this, but it resolves absolute paths against its working directory.)
fn copy_platform_binaries(
    output_path: &Path,
    platforms_path: &Path,
) -> Result<()> {
    for entry in fs::read_dir(platforms_path)? {
        let package_path = entry?.path();
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(package_path.join("package.json"))?)?;
        let binary =
            manifest["main"].as_str().ok_or_else(|| anyhow::anyhow!("Missing `main` in {}", package_path.display()))?;
        let source = output_path.join(binary);
        if !source.exists() {
            anyhow::bail!("Missing {} for {}", source.display(), package_path.display());
        }
        fs::copy(&source, package_path.join(binary))?;
    }
    Ok(())
}
