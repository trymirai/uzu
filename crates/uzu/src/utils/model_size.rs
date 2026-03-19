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
        let size_bytes = std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0);

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
#[path = "../../tests_unit/utils/model_size_test.rs"]
mod tests;
