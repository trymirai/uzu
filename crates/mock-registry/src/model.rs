use shoji::types::{
    basic::Metadata,
    model::{Model, ModelAccessibility, ModelBackend, ModelReference, ModelRegistry},
};

use crate::ServedFile;

pub fn mock_model(files: &[ServedFile]) -> Model {
    let registry = ModelRegistry {
        identifier: "mock-registry".to_string(),
        metadata: Metadata {
            identifier: "mock-registry-metadata".to_string(),
            name: "Mock Registry".to_string(),
            description: None,
            icons: vec![],
        },
    };
    let backend = ModelBackend {
        identifier: "mock-backend".to_string(),
        version: "1.0".to_string(),
        metadata: Metadata {
            identifier: "mock-backend-metadata".to_string(),
            name: "Mock Backend".to_string(),
            description: None,
            icons: vec![],
        },
    };
    Model {
        identifier: "mock/MockModel-1B".to_string(),
        registry,
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
                files: files.iter().map(|served_file| served_file.file.clone()).collect(),
            },
        },
    }
}
