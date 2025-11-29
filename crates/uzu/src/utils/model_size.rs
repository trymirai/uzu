use std::path::Path;

const FOUR_GB: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    Large,
    Small,
}

impl ModelSize {
    pub fn from_path(model_path: &Path) -> Self {
        let weights_path = model_path.join("model.safetensors");
        let size_bytes =
            std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0);

        Self::from_bytes(size_bytes)
    }

    pub fn from_bytes(size_bytes: u64) -> Self {
        if size_bytes > FOUR_GB {
            ModelSize::Large
        } else {
            ModelSize::Small
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::VERSION;
    pub const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";

    pub fn get_test_model_path() -> PathBuf {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join(VERSION)
            .join(MODEL_DIR_NAME);
        if !path.exists() {
            panic!(
                "Test model not found at {:?}. Run `./scripts/download_test_model.sh` to fetch it.",
                path
            );
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
}
