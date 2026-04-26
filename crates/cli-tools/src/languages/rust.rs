use crate::{configs::PlatformsConfig, languages::LanguageBackend, types::Language};

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
}
