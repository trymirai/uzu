use std::path::PathBuf;

use super::*;
use crate::VERSION;
pub const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";

pub fn get_test_model_path() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("workspace")
        .join("models")
        .join(VERSION)
        .join(MODEL_DIR_NAME);
    if !path.exists() {
        panic!("Test model not found at {:?}. Run `./scripts/download_test_model.sh` to fetch it.", path);
    }
    path
}

#[test]
fn test_model_size_from_path() {
    let model_path = get_test_model_path();

    if model_path.exists() {
        let size = ModelSize::from_path(&model_path);
        println!("Model size: {:?}", size);
        assert_eq!(size, ModelSize::Small);
    }
}
