use std::path::PathBuf;

pub const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
#[cfg(all(metal_backend, feature = "tracing"))]
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn get_test_model_path() -> PathBuf {
    let model_dir = std::env::var("TEST_MODEL").unwrap_or_else(|_| MODEL_DIR_NAME.to_string());
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("workspace")
        .join("models")
        .join(get_version())
        .join(model_dir);
    if !path.exists() {
        panic!("Test model not found at {:?}. Set TEST_MODEL env var or run `./scripts/download_test_model.sh`.", path);
    }
    path
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

#[cfg(all(metal_backend, feature = "tracing"))]
pub fn get_traces_path() -> PathBuf {
    get_test_model_path().join(TRACES_FILE_NAME)
}

#[cfg(target_os = "ios")]
pub fn ios_set_current_dir() {
    use objc2_foundation::{NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains};
    let paths = NSSearchPathForDirectoriesInDomains(
        NSSearchPathDirectory(9),  // NSDocumentDirectory
        NSSearchPathDomainMask(1), // NSUserDomainMask
        true,
    );
    if let Some(docs) = paths.firstObject() {
        let _ = std::env::set_current_dir(docs.to_string());
    }
}
