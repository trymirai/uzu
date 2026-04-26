use crate::{configs::PlatformsConfig, languages::LanguageBackend, types::Language};

pub struct SwiftLanguageBackend {
    config: PlatformsConfig,
}

impl SwiftLanguageBackend {
    pub fn new(config: PlatformsConfig) -> Self {
        Self {
            config,
        }
    }
}

impl LanguageBackend for SwiftLanguageBackend {
    fn config(&self) -> PlatformsConfig {
        self.config.clone()
    }

    fn language(&self) -> Language {
        Language::Swift
    }
}
