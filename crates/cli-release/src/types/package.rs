use std::path::PathBuf;

#[derive(Debug)]
pub struct Package {
    pub root_path: PathBuf,
}

impl Package {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
        }
    }
}

impl Package {
    pub fn manifest_path(&self) -> PathBuf {
        self.root_path.join("Cargo.toml")
    }
}
