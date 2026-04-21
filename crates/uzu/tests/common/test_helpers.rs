#![allow(dead_code)]

use std::path::PathBuf;

use shoji::types::model::{Accessibility, Entity, EntityType, File, Hash, HashMethod, Model, Reference};
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
    let registry_entity = Entity {
        r#type: EntityType::Registry,
        identifier: "test-registry".to_string(),
        parent_identifier: None,
        version: None,
        name: "Test Registry".to_string(),
        description: None,
        icons: vec![],
    };

    let backend_entity = Entity {
        r#type: EntityType::Backend,
        identifier: "test-backend".to_string(),
        parent_identifier: None,
        version: None,
        name: "Test Backend".to_string(),
        description: None,
        icons: vec![],
    };

    let vendor_entity = Entity {
        r#type: EntityType::Vendor,
        identifier: "test-vendor".to_string(),
        parent_identifier: None,
        version: None,
        name: "Test Vendor".to_string(),
        description: None,
        icons: vec![],
    };

    let family_entity = Entity {
        r#type: EntityType::Family,
        identifier: "test-family".to_string(),
        parent_identifier: None,
        version: None,
        name: "Test Family".to_string(),
        description: None,
        icons: vec![],
    };

    let model_entity = Entity {
        r#type: EntityType::Variant,
        identifier: format!("test-model-{}", index),
        parent_identifier: None,
        version: None,
        name: format!("Test Model {}", index),
        description: None,
        icons: vec![],
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
        entities: vec![registry_entity, backend_entity, vendor_entity, family_entity, model_entity],
        specializations: vec![],
        number_of_parameters: None,
        quantization: None,
        accessibility: Accessibility::Local {
            reference: Reference::Mirai {
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
