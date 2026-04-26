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
                .run()?;
            if target.name == host_target {
                Command::uv_sync().with_current_path(&bindings_path).run()?;
                Command::uv_python("import uzu; uzu.generate_annotations()").with_current_path(&bindings_path).run()?;
            }
        }
        Ok(())
    }
}
