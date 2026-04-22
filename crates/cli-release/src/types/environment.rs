use std::path::{Path, PathBuf};

use crate::types::{Bindings, Error, Package, Scripts, Workspace};

#[derive(Debug)]
pub struct Environment {
    pub root_path: PathBuf,
}

impl Environment {
    pub fn new() -> Result<Self, Error> {
        let crate_manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();

        let crates_path = crate_manifest_path.parent().ok_or(Error::RootPathNotFound)?.to_path_buf();

        let root_path = crates_path.parent().ok_or(Error::RootPathNotFound)?.to_path_buf();

        let root_manifest_path = root_path.join("Cargo.toml");
        if !root_manifest_path.exists() {
            return Err(Error::RootPathNotFound);
        }

        Ok(Self {
            root_path: root_path,
        })
    }
}

impl Environment {
    pub fn crates_path(&self) -> PathBuf {
        self.root_path.join("crates")
    }

    pub fn bindings_path(&self) -> PathBuf {
        self.root_path.join("bindings")
    }

    pub fn workspace_path(&self) -> PathBuf {
        self.root_path.join("workspace")
    }
}

impl Environment {
    pub fn package_uzu(&self) -> Package {
        Package::new(self.crates_path().join("uzu"))
    }

    pub fn package_release_cli(&self) -> Package {
        Package::new(self.crates_path().join("cli-release"))
    }

    pub fn scripts(&self) -> Scripts {
        Scripts::new(self.package_release_cli().root_path.join("scripts"))
    }

    pub fn bindings(&self) -> Bindings {
        Bindings::new(self.bindings_path())
    }

    pub fn workspace(&self) -> Workspace {
        Workspace::new(self.workspace_path())
    }
}
