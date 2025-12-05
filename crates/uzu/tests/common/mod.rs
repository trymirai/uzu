#![allow(dead_code)]
use std::path::PathBuf;

use uzu::VERSION;

const DEFAULT_MODEL_DIR_NAME: &str = "Qwen3-0.6B-MLX-4bit";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_model_dir_name() -> String {
    std::env::var("UZU_TEST_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR_NAME.to_string())
}

pub fn get_test_model_path() -> PathBuf {
    let model_dir_name = get_model_dir_name();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join(VERSION)
        .join(&model_dir_name);
    if !path.exists() {
        panic!(
            "Test model not found at {:?}. Run `./scripts/download_test_model.sh` to fetch it.",
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
