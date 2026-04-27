use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::types::Language;

pub struct Paths {
    pub root_path: PathBuf,
    pub main_crate: String,
}

impl Paths {
    pub fn new() -> Result<Self> {
        let error_message = "ROOT_PATH not found";
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .context(error_message)?
            .parent()
            .context(error_message)?
            .to_path_buf();
        Ok(Self {
            root_path,
            main_crate: "uzu".to_string(),
        })
    }

    pub fn platforms_toml(&self) -> PathBuf {
        self.root_path.join("platforms.toml")
    }

    pub fn bindings_path(&self) -> PathBuf {
        self.root_path.join("bindings")
    }

    pub fn bindings_for_language_path(
        &self,
        language: Language,
    ) -> PathBuf {
        self.bindings_path().join(language.name())
    }

    pub fn crates_path(&self) -> PathBuf {
        self.root_path.join("crates")
    }

    pub fn crate_path(
        &self,
        name: &str,
    ) -> PathBuf {
        self.crates_path().join(name)
    }

    pub fn crate_manifest_path(
        &self,
        name: &str,
    ) -> PathBuf {
        self.crate_path(name).join("Cargo.toml")
    }

    pub fn napi_output_path(&self) -> PathBuf {
        self.bindings_for_language_path(Language::TypeScript).join("src").join("napi")
    }
}
