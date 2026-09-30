use anyhow::Result;

use crate::{configs::Paths, types::Command};

pub fn update_lock(paths: &Paths) -> Result<()> {
    Command::new("cargo")
        .with_arguments(["update", "--workspace", "--manifest-path"].map(String::from).to_vec())
        .with_argument(&paths.root_path.join("apps/mirai-gui/src-tauri/Cargo.toml").to_string_lossy())
        .run()
}
