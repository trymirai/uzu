use anyhow::{Result, anyhow};

use crate::{
    configs::{ALL_TARGET, CargoLock, Paths, PlatformsConfig},
    languages::{LanguageBackend, LanguageBackendTarget},
    types::{Command, Configuration, Language},
};

pub struct WasmLanguageBackend {
    config: PlatformsConfig,
}

impl WasmLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }

    fn check_wasm_bindgen(&self) -> Result<()> {
        let required_version = CargoLock::load()?.package_version("wasm-bindgen")?;
        let install_hint =
            format!("Install it with: cargo install wasm-bindgen-cli --version {required_version} --locked");
        let (output, _) = Command::wasm_bindgen_version()
            .output()
            .map_err(|_| anyhow!("wasm-bindgen-cli is not installed. {install_hint}"))?;
        let installed_version = output.split_whitespace().nth(1);
        if installed_version != Some(required_version.as_str()) {
            return Err(anyhow!(
                "wasm-bindgen-cli {required_version} is required (matches the `wasm-bindgen` crate in Cargo.lock), found `{output}`. {install_hint}"
            ));
        }
        Ok(())
    }
}

impl LanguageBackend for WasmLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Wasm
    }

    fn default_targets(&self) -> Vec<String> {
        vec![ALL_TARGET.to_string()]
    }

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        self.check_wasm_bindgen()?;

        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let output_path = paths.wasm_output_path();

        Command::pnpm_install().with_current_path(&bindings_path).run()?;

        for target in targets {
            Command::cargo_build(paths.main_crate.clone(), target.name.clone(), target.features.clone(), configuration)
                .with_envs(self.config.required_envs_for_target(target.name.clone())?)
                .run()?;
            let wasm_path =
                paths.target_profile_path(&target.name, configuration).join(format!("{}.wasm", paths.main_crate));
            Command::wasm_bindgen(wasm_path, output_path.clone(), paths.main_crate.clone()).run()?;
            Command::pnpm_run("build").with_current_path(&bindings_path).run()?;
        }
        Ok(())
    }

    fn example_target(
        &self,
        name: &str,
        _configuration: Configuration,
        _target: LanguageBackendTarget,
    ) -> Result<()> {
        let paths = Paths::new()?;
        let bindings_path = paths.bindings_for_language_path(self.language());
        let name = self.language().convert_file_name(name);
        let file_path = paths.wasm_examples_path().join(format!("{name}.{}", self.language().file_extension()));
        if !file_path.exists() {
            return Err(anyhow!("Example not found: {}", file_path.display()));
        }
        Command::pnpm_examples(&name).with_current_path(&bindings_path).run()
    }
}
