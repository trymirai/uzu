use std::path::PathBuf;

#[derive(Debug)]
pub struct Workspace {
    pub root_path: PathBuf,
}

impl Workspace {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
        }
    }
}

impl Workspace {
    pub fn ts_path(&self) -> PathBuf {
        self.root_path.join("ts")
    }

    pub fn ts_npm_path(&self) -> PathBuf {
        self.root_path.join("ts-npm")
    }

    pub fn swift_path(&self) -> PathBuf {
        self.root_path.join("swift")
    }

    pub fn swift_spm_path(&self) -> PathBuf {
        self.root_path.join("swift-spm")
    }

    pub fn platform_path(&self) -> PathBuf {
        self.root_path.join("platform")
    }
}

impl Workspace {
    pub fn docs_path(&self) -> PathBuf {
        self.root_path.join("docs")
    }

    pub fn docs_snippets_path(&self) -> PathBuf {
        self.docs_path().join("snippets")
    }

    pub fn docs_examples_path(&self) -> PathBuf {
        self.docs_path().join("examples")
    }
}

impl Workspace {
    pub fn ts_napi_path(&self) -> PathBuf {
        self.root_path.join("ts-napi")
    }

    pub fn ts_napi_index_d_ts(&self) -> PathBuf {
        self.ts_napi_path().join("index.d.ts")
    }

    pub fn ts_napi_uzu_d_ts(&self) -> PathBuf {
        self.ts_napi_path().join("uzu.d.ts")
    }

    pub fn ts_napi_uzu_node(&self) -> PathBuf {
        self.ts_napi_path().join("uzu.node")
    }
}
