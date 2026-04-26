use crate::{configs::PlatformsConfig, languages::LanguageBackend, types::Language};

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
}
