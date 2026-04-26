use crate::{configs::PlatformsConfig, languages::LanguageBackend, types::Language};

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
}
