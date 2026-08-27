use shoji::types::{
    basic::{File, Metadata},
    model::{ModelFamily, ModelProperties, ModelQuantization, ModelRegistry, ModelVendor},
};

use super::*;

const GB: i64 = 1_000_000_000;

#[test]
fn family_only_prefers_largest_mirai_model_that_fits() {
    let models = vec![
        checkpoint("0.8b", 800_000_000, "mirai", "mirai-m", 4, GB),
        checkpoint("4b", 4 * GB, "mirai", "mirai-m", 4, 4 * GB),
        checkpoint("27b", 27 * GB, "mlx-community", "mlx", 4, 16 * GB),
    ];

    let resolved = resolve_model_shorthand(&models, "qwen3.5", 32 * GB as u64).unwrap().unwrap();

    assert_eq!(resolved.properties.as_ref().unwrap().identifier, "4b");
    assert_eq!(resolved.quantization.as_ref().unwrap().method, "mirai-m");
}

#[test]
fn family_only_uses_largest_checkpoint_when_no_mirai_model_exists() {
    let models = vec![
        checkpoint("4b", 4 * GB, "mlx-community", "mlx", 8, 4 * GB),
        checkpoint("27b", 27 * GB, "mlx-community", "mlx", 4, 16 * GB),
    ];

    let resolved = resolve_model_shorthand(&models, "qwen3.5", 32 * GB as u64).unwrap().unwrap();

    assert_eq!(resolved.properties.as_ref().unwrap().identifier, "27b");
}

#[test]
fn shorthand_prefers_mirai_m_then_mirai_s() {
    let mlx = checkpoint("4b", 4 * GB, "mlx-community", "mlx", 8, 4 * GB);
    let mirai_s = checkpoint("4b", 4 * GB, "mirai", "mirai-s", 4, 4 * GB);
    let mirai_m = checkpoint("4b", 4 * GB, "mirai", "mirai-m", 4, 4 * GB);

    let models = vec![mlx.clone(), mirai_s.clone(), mirai_m.clone()];
    assert_eq!(resolve_model_shorthand(&models, "qwen3.5:4b", 8 * GB as u64), Ok(Some(&mirai_m)));

    let models = vec![mlx, mirai_s.clone()];
    assert_eq!(resolve_model_shorthand(&models, "qwen3.5:4b", 8 * GB as u64), Ok(Some(&mirai_s)));
}

#[test]
fn shorthand_infers_family_and_quantization_vendors() {
    let model = checkpoint("4b", 4 * GB, "mlx-community", "mlx", 8, 4 * GB);
    let models = [model.clone()];

    assert_eq!(resolve_model_shorthand(&models, "qwen3.5:4b:mlx:8", 8 * GB as u64), Ok(Some(&model)));
    assert_eq!(
        resolve_model_shorthand(&models, "alibaba:qwen3.5:4b:mlx-community:mlx:8", 8 * GB as u64),
        Ok(Some(&model))
    );
}

#[test]
fn valid_shorthand_returns_insufficient_memory_when_no_checkpoint_fits() {
    let models = [checkpoint("4b", 4 * GB, "mirai", "mirai-m", 4, 4 * GB)];

    let result = resolve_model_shorthand(&models, "qwen3.5:4b", GB as u64);

    assert_eq!(
        result,
        Err(ModelResolutionError::InsufficientMemory {
            model: "qwen3.5:4b".to_string(),
            memory_total: GB as u64,
        })
    );
}

#[test]
fn unknown_shorthand_remains_unresolved() {
    let models = [checkpoint("4b", 4 * GB, "mirai", "mirai-m", 4, 4 * GB)];

    assert_eq!(resolve_model_shorthand(&models, "unknown", GB as u64), Ok(None));
}

fn checkpoint(
    size_id: &str,
    parameters: i64,
    quantization_vendor_id: &str,
    quantization_method: &str,
    bits_per_weight: u32,
    checkpoint_size: i64,
) -> Model {
    let vendor = ModelVendor {
        identifier: "alibaba".to_string(),
        metadata: metadata("Alibaba"),
    };
    let quantization_vendor = ModelVendor {
        identifier: quantization_vendor_id.to_string(),
        metadata: metadata(quantization_vendor_id),
    };
    Model {
        identifier: format!(
            "alibaba:qwen3.5:{size_id}:{quantization_vendor_id}:{quantization_method}:{bits_per_weight}"
        ),
        registry: ModelRegistry {
            identifier: "mirai".to_string(),
            metadata: metadata("Mirai"),
        },
        backends: vec![],
        family: Some(ModelFamily {
            identifier: "alibaba:qwen3.5".to_string(),
            vendor,
            metadata: metadata("Qwen3.5"),
        }),
        properties: Some(ModelProperties {
            identifier: size_id.to_string(),
            size: parameters,
            version: None,
            metadata: metadata(size_id),
        }),
        quantization: Some(ModelQuantization {
            identifier: format!("{quantization_vendor_id}:{quantization_method}:{bits_per_weight}"),
            method: quantization_method.to_string(),
            bits_per_weight,
            vendor: quantization_vendor,
            metadata: metadata(quantization_method),
        }),
        specializations: vec![],
        accessibility: ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                toolchain_version: "test".to_string(),
                repository: None,
                source_repository: None,
                files: vec![File {
                    url: "https://example.com/model.safetensors".to_string(),
                    name: "model.safetensors".to_string(),
                    size: checkpoint_size,
                    hashes: vec![],
                }],
            },
        },
        encoding: None,
    }
}

fn metadata(name: &str) -> Metadata {
    Metadata {
        identifier: name.to_lowercase(),
        name: name.to_string(),
        description: None,
        icons: vec![],
    }
}
