use std::path::PathBuf;

#[derive(Debug)]
pub struct Scripts {
    pub root_path: PathBuf,
}

impl Scripts {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
        }
    }
}

impl Scripts {
    pub fn ts_napi_build(&self) -> PathBuf {
        self.root_path.join("ts_napi_build.sh")
    }

    pub fn ts_build(&self) -> PathBuf {
        self.root_path.join("ts_build.sh")
    }

    pub fn sync_into_repo(&self) -> PathBuf {
        self.root_path.join("sync_into_repo.sh")
    }

    pub fn zip_xcframework(&self) -> PathBuf {
        self.root_path.join("zip_xcframework.sh")
    }
}
