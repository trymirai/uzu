use std::path::PathBuf;

pub const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
#[cfg(feature = "tracing")]
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn get_test_model_path() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join(get_version())
        .join(MODEL_DIR_NAME);
    if !path.exists() {
        panic!("Test model not found at {:?}. Run `./scripts/download_test_model.sh` to fetch it.", path);
    }
    path
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

#[cfg(feature = "tracing")]
pub fn get_traces_path() -> PathBuf {
    get_test_model_path().join(TRACES_FILE_NAME)
}
