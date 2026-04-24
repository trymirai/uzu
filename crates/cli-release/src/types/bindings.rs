use std::path::PathBuf;

#[derive(Debug)]
pub struct Bindings {
    pub root_path: PathBuf,
}

impl Bindings {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
        }
    }
}

impl Bindings {
    pub fn ts_napi_path(&self) -> PathBuf {
        self.root_path.join("ts-napi")
    }

    pub fn ts_path(&self) -> PathBuf {
        self.root_path.join("ts")
    }
}

impl Bindings {
    pub fn swift_path(&self) -> PathBuf {
        self.root_path.join("swift")
    }

    pub fn swift_build_script_path(&self) -> PathBuf {
        self.swift_path().join("build_release_xcframework.sh")
    }

    pub fn swift_framework_path(&self) -> PathBuf {
        self.swift_path().join("uzu.xcframework")
    }
}
