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

    fn build_targets(
        &self,
        configuration: Configuration,
        targets: Vec<LanguageBackendTarget>,
    ) -> Result<()> {
        let paths = Paths::new()?;
        for target in targets {
            Command::cargo_build(paths.main_crate.clone(), target.name.clone(), target.features.clone(), configuration)
                .run()?;
        }
        Ok(())
    }
}
