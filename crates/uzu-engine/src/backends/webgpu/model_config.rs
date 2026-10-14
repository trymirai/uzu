//! Lightweight architecture checks before downloading weights. Tensor formats
//! and adapter allocation limits are still checked by the engine during loading.
use crate::config::{mlp::AnyMLPConfig, model::language_model::LanguageModelConfig, token_mixer::AnyTokenMixerConfig};

pub fn validate_model_config(bytes: &[u8]) -> Result<(), super::WebGpuError> {
    inspect(bytes).map_err(super::WebGpuError::Unsupported)
}

fn inspect(bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > 16 * 1024 * 1024 {
        return Err("Model config is empty or exceeds the 16 MB browser size limit".into());
    }
    let config: LanguageModelConfig =
        serde_json::from_slice(bytes).map_err(|e| format!("Invalid model config: {e}"))?;
    let decoder = &config.decoder_config;
    if decoder.ple_model_config.is_some() {
        return Err("WebGPU does not yet support per-layer embeddings".into());
    }
    let transformer = &decoder.transformer_config;
    if decoder.vocab_size == 0 || transformer.model_dim == 0 || transformer.layer_configs.is_empty() {
        return Err("Model dimensions and layer count must be positive".into());
    }
    for (index, layer) in transformer.layer_configs.iter().enumerate() {
        let reject = |reason: &str| Err(format!("WebGPU layer {index}: {reason}"));
        if layer.ple_config.is_some() {
            return reject("per-layer embeddings are not yet supported");
        }
        if layer.mixer_conv_config.is_some() || layer.mlp_conv_config.is_some() {
            return reject("general transformer convolutions are not yet supported");
        }
        if !matches!(layer.mlp_config, AnyMLPConfig::DenseMLPConfig(_)) {
            return reject("mixture-of-experts models are not yet supported");
        }
        match &layer.mixer_config {
            AnyTokenMixerConfig::AttentionConfig(c) => {
                if !(1..=256).contains(&c.head_dim) {
                    return reject("attention head dimensions must be between 1 and 256");
                }
                if c.num_groups == 0 || c.num_heads == 0 || !c.num_heads.is_multiple_of(c.num_groups) {
                    return reject("attention heads must be a positive multiple of the KV groups");
                }
            },
            AnyTokenMixerConfig::DeltaNetConfig(c) => {
                if !(1..=256).contains(&c.head_dim) || !(1..=256).contains(&c.value_head_dim) {
                    return reject("DeltaNet head dimensions must be between 1 and 256");
                }
                if c.kernel_size == 0
                    || c.num_groups == 0
                    || c.num_heads == 0
                    || !c.num_heads.is_multiple_of(c.num_groups)
                {
                    return reject("DeltaNet requires a positive kernel size and heads divisible by groups");
                }
            },
            AnyTokenMixerConfig::ShortConvConfig(c) => {
                if c.kernel_size == 0 {
                    return reject("short convolution kernel size must be positive");
                }
            },
            AnyTokenMixerConfig::Mamba2Config(_) => return reject("Mamba2 models are not yet supported"),
        }
    }
    Ok(())
}
