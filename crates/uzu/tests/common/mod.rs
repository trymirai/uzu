#![allow(dead_code)]
use std::path::PathBuf;

pub const MODEL_DIR_NAME: &str = "Qwen3-4B-AWQ";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_test_model_path() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("mirai_models")
        .join(MODEL_DIR_NAME);
    if !path.exists() {
        panic!(
            "Test model not found at {:?}. Run `bash scripts/download_test_model.sh` to fetch it.",
            path
        );
    }
    path
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

pub fn get_traces_path() -> PathBuf {
    get_test_model_path().join(TRACES_FILE_NAME)
}
