use std::path::{Path, PathBuf};

pub const CACHE_NAME: &str = "mirai";
pub const CONFIG_FILE: &str = "config.json";
pub const HEADER_FILE: &str = "model.header.safetensors";
pub const WEIGHTS_FILE: &str = "model.safetensors";

pub fn cache_models_path(storage_base: &Path) -> PathBuf {
    storage_base.join(".cache").join(CACHE_NAME).join("models")
}

pub fn resolve_weights_path(model_dir: &Path) -> Option<PathBuf> {
    let header_path = model_dir.join(HEADER_FILE);
    if header_path.is_file() {
        return Some(header_path);
    }
    let weights_path = model_dir.join(WEIGHTS_FILE);
    if weights_path.is_file() {
        return Some(weights_path);
    }
    None
}
