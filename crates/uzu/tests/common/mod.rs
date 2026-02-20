#![allow(dead_code)]

use std::{collections::HashMap, path::PathBuf};

use uzu::{VERSION, speculators::speculator::Speculator};

pub const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
pub const TRACES_FILE_NAME: &str = "traces.safetensors";

pub fn get_test_model_path() -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join(VERSION)
        .join(MODEL_DIR_NAME);
    if !path.exists() {
        panic!("Test model not found at {:?}. Run `./scripts/download_test_model.sh` to fetch it.", path);
    }
    path
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

pub fn get_traces_path() -> PathBuf {
    get_test_model_path().join(TRACES_FILE_NAME)
}

pub struct StaticSpeculator {
    responses: HashMap<Vec<u64>, HashMap<u64, f32>>,
    default_response: Option<HashMap<u64, f32>>,
}

impl StaticSpeculator {
    pub fn new(responses: HashMap<Vec<u64>, HashMap<u64, f32>>) -> Self {
        Self {
            responses,
            default_response: None,
        }
    }

    pub fn with_default_response(default_response: HashMap<u64, f32>) -> Self {
        Self {
            responses: HashMap::new(),
            default_response: Some(default_response),
        }
    }
}

impl Speculator for StaticSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        self.responses.get(prefix).cloned().or_else(|| self.default_response.clone()).unwrap_or_default()
    }
}

pub struct RepeatSpeculator;

impl Speculator for RepeatSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        let mut hm = HashMap::new();

        for (pos, token) in prefix.iter().copied().enumerate() {
            *hm.entry(token).or_insert(0.0) += f32::sqrt((1 + pos) as f32);
        }

        let sum = hm.values().sum::<f32>();

        hm.into_iter().map(|(pos, weight)| (pos, weight / sum)).collect()
    }
}
