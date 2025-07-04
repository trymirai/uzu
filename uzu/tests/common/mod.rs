#![allow(dead_code)]
use std::path::PathBuf;

use uzu::storage::storage_path;

pub const MODEL_DIR_NAME: &str = "Meta-Llama-3.2-1B-Instruct-float16";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_test_model_path() -> PathBuf {
    let path = storage_path().join(MODEL_DIR_NAME);
    if !path.exists() {
        panic!(
            "Test model not found at {:?}. Please make sure the model is downloaded to the SDK storage area.",
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
