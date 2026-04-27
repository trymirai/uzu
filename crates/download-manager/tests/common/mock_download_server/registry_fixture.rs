use std::{collections::HashMap, sync::Arc};

use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use bytesize::ByteSize;
use shoji::types::{
    basic::{File, Hash, HashMethod, Metadata},
    model::{Model, ModelAccessibility, ModelBackend, ModelFamily, ModelReference, ModelRegistry, ModelVendor},
};

use crate::common::mock_download_server::FilePayload;

#[derive(Clone, Debug)]
pub struct RegistryFixture {
    pub model: Model,
    pub files: Box<[File]>,
    payloads_by_name: HashMap<String, Arc<[u8]>>,
    last_modified: String,
}

impl RegistryFixture {
    pub fn llama_3_2_1b_instruct(
        base_url: &str,
        path_prefix: &str,
    ) -> Self {
        let model_path = Self::llama_model_path(path_prefix);
        let file_payloads = [
            Self::file_payload(base_url, &model_path, "config.json", ByteSize::b(55_559)),
            Self::file_payload(base_url, &model_path, "model.safetensors", ByteSize::mib(32)),
            Self::file_payload(base_url, &model_path, "tokenizer.json", ByteSize::b(17_209_920)),
            Self::file_payload(base_url, &model_path, "traces.safetensors", ByteSize::b(31_050_840)),
        ];
        let files = file_payloads.iter().map(|payload| payload.file.clone()).collect::<Box<_>>();
        let payloads_by_name =
            file_payloads.into_iter().map(|payload| (payload.file.name, payload.bytes)).collect::<HashMap<_, _>>();
        let model = Self::model(files.iter().cloned().collect());
        Self {
            model,
            files,
            payloads_by_name,
            last_modified: "Sun, 26 Apr 2026 12:00:00 GMT".to_string(),
        }
    }

    pub fn payload(
        &self,
        file_name: &str,
    ) -> FilePayload {
        let file = self
            .files
            .iter()
            .find(|candidate| candidate.name == file_name)
            .cloned()
            .expect("Llama fixture file should exist");
        let bytes = self.payloads_by_name.get(file_name).cloned().expect("Llama fixture payload should exist");
        FilePayload {
            file,
            bytes,
            last_modified: self.last_modified.clone(),
        }
    }

    pub fn payloads(&self) -> Box<[FilePayload]> {
        self.files.iter().map(|file| self.payload(&file.name)).collect()
    }

    fn file_payload(
        base_url: &str,
        path_prefix: &str,
        name: &str,
        size: ByteSize,
    ) -> FilePayload {
        let bytes = Self::bytes(name, size);
        let crc32c = crc32c::crc32c(&bytes);
        let crc32c = BASE64_STANDARD.encode(crc32c.to_be_bytes());
        let path = format!("/{}/{}", path_prefix.trim_matches('/'), name);
        let file = File {
            url: format!("{}{}", base_url.trim_end_matches('/'), path),
            name: name.to_string(),
            size: i64::try_from(size.as_u64()).expect("mock file size should fit in i64"),
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: crc32c,
            }],
        };
        FilePayload {
            file,
            bytes: Arc::from(bytes),
            last_modified: "Sun, 26 Apr 2026 12:00:00 GMT".to_string(),
        }
    }

    fn model(files: Vec<File>) -> Model {
        let registry = ModelRegistry {
            identifier: "meta-llama".to_string(),
            metadata: Metadata::external("meta-llama".to_string()),
        };
        let backend = ModelBackend {
            identifier: "mirai".to_string(),
            version: "0.1.9".to_string(),
            metadata: Metadata::external("mirai".to_string()),
        };
        let vendor = ModelVendor {
            identifier: "Meta".to_string(),
            metadata: Metadata::external("Meta".to_string()),
        };
        Model {
            identifier: "meta-llama/Llama-3.2-1B-Instruct".to_string(),
            registry,
            backends: vec![backend],
            family: Some(ModelFamily {
                identifier: "Llama-3.2".to_string(),
                vendor,
                metadata: Metadata::external("Llama-3.2".to_string()),
            }),
            properties: None,
            quantization: None,
            specializations: vec![],
            accessibility: ModelAccessibility::Local {
                reference: ModelReference::Mirai {
                    toolchain_version: "1.0".to_string(),
                    repository: None,
                    source_repository: None,
                    files,
                },
            },
        }
    }

    fn llama_model_path(path_prefix: &str) -> String {
        let trimmed_path_prefix = path_prefix.trim_matches('/');
        if trimmed_path_prefix.ends_with("Llama-3.2-1B-Instruct") {
            trimmed_path_prefix.to_string()
        } else {
            format!("{trimmed_path_prefix}/Llama-3.2-1B-Instruct")
        }
    }

    fn bytes(
        name: &str,
        size: ByteSize,
    ) -> Vec<u8> {
        let name_seed = name.bytes().fold(0u8, |seed, byte| seed.wrapping_add(byte));
        (0..usize::try_from(size.as_u64()).expect("mock file size should fit in usize"))
            .map(|byte_index| {
                let low_bits = byte_index as u8;
                low_bits.wrapping_mul(31).wrapping_add(name_seed).wrapping_add((byte_index / 7) as u8)
            })
            .collect()
    }
}
