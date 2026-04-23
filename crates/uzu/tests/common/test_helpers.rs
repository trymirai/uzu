#![allow(dead_code)]

use std::path::PathBuf;

use shoji::types::{
    basic::{File, Hash, HashMethod, Metadata},
    model::{Model, ModelAccessibility, ModelBackend, ModelReference, ModelRegistry},
};
use tokio::runtime::Handle;
use uzu::{
    device::Device,
    registry::FixedRegistry,
    storage::{Config, Storage},
};

pub struct TestStorage {
    pub models: Vec<Model>,
    pub config: Config,
    pub registry: FixedRegistry,
    pub storage: Storage,
    pub base_path: PathBuf,
}

impl TestStorage {
    pub async fn new_with_base_path(base_path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let model1 = test_model(0);
        let model2 = test_model(1);

        let registry = FixedRegistry::new("test_registry".to_string(), vec![model1.clone(), model2.clone()]);

        let device = Device::new().unwrap();
        let config = Config::new(device, Some(base_path.clone()), "test_storage".to_string());

        let handle = Handle::current();
        let storage = Storage::new(handle, config.clone()).await.unwrap();
        storage.refresh(vec![model1.clone(), model2.clone()]).await.unwrap();
        Ok(Self {
            models: vec![model1, model2],
            config,
            registry,
            storage,
            base_path,
        })
    }

    pub fn model(
        &self,
        index: usize,
    ) -> Model {
        self.models[index].clone()
    }
}

fn test_model(index: usize) -> Model {
    let registry = ModelRegistry {
        identifier: "test-registry".to_string(),
        metadata: Metadata::external("test-registry-metadata".to_string()),
    };

    let backend = ModelBackend {
        identifier: "test-backend".to_string(),
        version: "1.0".to_string(),
        metadata: Metadata::external("test-backend-metadata".to_string()),
    };

    let files_1 = vec![
        File {
            url: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json".to_string(),
            name: "config.json".to_string(),
            size: 2907,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "nx43ZQ==".to_string(),
            }],
        },
        File {
            url: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json".to_string(),
            name: "tokenizer.json".to_string(),
            size: 12807982,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "C/tYqQ==".to_string(),
            }],
        },
        File {
            url: "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer_config.json".to_string(),
            name: "tokenizer_config.json".to_string(),
            size: 16709,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "MtGCnw==".to_string(),
            }],
        },
    ];

    let files_2 = vec![
        File {
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json".to_string(),
            name: "config.json".to_string(),
            size: 726,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "BLHgXg==".to_string(),
            }],
        },
        File {
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json".to_string(),
            name: "tokenizer.json".to_string(),
            size: 11422654,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "t3YzMQ==".to_string(),
            }],
        },
        File {
            url: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json".to_string(),
            name: "tokenizer_config.json".to_string(),
            size: 9732,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "PUp7Mg==".to_string(),
            }],
        },
    ];

    Model {
        identifier: format!("test-model-{}", index),
        registry: registry,
        backends: vec![backend],
        family: None,
        properties: None,
        quantization: None,
        specializations: vec![],
        accessibility: ModelAccessibility::Local {
            reference: ModelReference::Mirai {
                toolchain_version: "1.0".to_string(),
                repository: None,
                source_repository: None,
                files: if index % 2 == 0 {
                    files_1
                } else {
                    files_2
                },
            },
        },
    }
}
