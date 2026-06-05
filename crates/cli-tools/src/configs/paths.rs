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

    pub fn target_path(&self) -> PathBuf {
        self.root_path.join("target")
    }

    pub fn target_wheels_path(&self) -> PathBuf {
        self.target_path().join("wheels")
    }

    pub fn artifacts_path(&self) -> PathBuf {
        self.target_path().join("cli-tools")
    }

    pub fn platforms_toml(&self) -> PathBuf {
        self.root_path.join("platforms.toml")
    }

    pub fn readme_template_path(&self) -> PathBuf {
        self.root_path.join("workspace").join("readme").join("template.md")
    }

    pub fn readme_fragments_path(&self) -> PathBuf {
        self.root_path.join("workspace").join("readme").join("fragments")
    }

    pub fn workspace_manifest_path(&self) -> PathBuf {
        self.root_path.join("Cargo.toml")
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

    pub fn swift_xcframework_path(&self) -> PathBuf {
        self.bindings_for_language_path(Language::Swift).join(format!("{}.xcframework", self.main_crate))
    }

    pub fn swift_generated_sources_path(&self) -> PathBuf {
        self.bindings_for_language_path(Language::Swift).join("Sources").join("Uzu").join("Generated")
    }

    pub fn swift_slices_path(&self) -> PathBuf {
        self.artifacts_path().join("swift").join("slices")
    }

    pub fn release_workspace_path(&self) -> PathBuf {
        self.root_path.join("workspace").join("release")
    }

    pub fn release_platform_path(&self) -> PathBuf {
        self.release_workspace_path().join("platform")
    }

    pub fn release_docs_path(&self) -> PathBuf {
        self.release_workspace_path().join("docs")
    }

    pub fn release_binaries_path(&self) -> PathBuf {
        self.release_workspace_path().join("binaries")
    }

    pub fn release_binary_path(
        &self,
        binary_name: &str,
    ) -> PathBuf {
        self.release_binaries_path().join(binary_name)
    }

    pub fn release_swift_spm_path(&self) -> PathBuf {
        self.release_workspace_path().join("swift-spm")
    }

    pub fn release_typescript_npm_path(&self) -> PathBuf {
        self.release_workspace_path().join("typescript-npm")
    }

    pub fn release_python_pypi_path(&self) -> PathBuf {
        self.release_workspace_path().join("python-pypi")
    }

    pub fn root_package_swift_path(&self) -> PathBuf {
        self.root_path.join("Package.swift")
    }
}
