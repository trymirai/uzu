use serde::{Deserialize, Serialize};

use crate::{
    DecoderConfig, DecoderLayerConfig, EmbeddingConfig, GenerationConfig,
    MessageProcessorConfig, MixerConfig, TransformerConfig,
};

/// Inner model config matching the new lalamo export format.
/// Contains embedding_config at the top level, with transformer_config nested.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InnerModelConfig {
    pub embedding_config: EmbeddingConfig,
    pub transformer_config: TransformerConfig,
    pub vocab_size: usize,
}

impl InnerModelConfig {
    /// Convert to DecoderConfig for backward compatibility with the rest of the codebase.
    pub fn to_decoder_config(&self) -> DecoderConfig {
        let tf = &self.transformer_config;

        // Get the first layer config as the template for layer_config
        let first_layer = tf
            .layer_configs
            .first()
            .expect("transformer_config must have at least one layer");

        let layer_config = DecoderLayerConfig {
            pre_attention_norm_config: first_layer
                .pre_attention_norm_config
                .clone()
                .unwrap_or_else(|| tf.output_norm_config.clone()),
            mixer_config: MixerConfig::Attention(first_layer.attention_config.clone()),
            post_attention_norm_config: first_layer.post_attention_norm_config.clone(),
            pre_mlp_norm_config: first_layer.pre_mlp_norm_config.clone(),
            mlp_config: first_layer.mlp_config.clone(),
            post_mlp_norm_config: first_layer.post_mlp_norm_config.clone(),
        };

        // Derive head parameters from the first attention config
        let attn = &first_layer.attention_config;
        let num_heads = tf.num_heads.or(attn.num_heads).unwrap_or(32);
        let num_groups = tf.num_groups.or(attn.num_groups).unwrap_or(num_heads);
        let head_dim = tf.head_dim.or(attn.head_dim).unwrap_or(64);
        let attention_scale = tf.attention_scale.or(attn.scale);
        let num_layers = tf.num_layers.unwrap_or(tf.layer_configs.len());

        // Extract sliding window sizes from each layer's attention config
        let sliding_window_sizes: Box<[Option<usize>]> = tf
            .layer_configs
            .iter()
            .map(|layer| layer.attention_config.sliding_window_size)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // Convert layer_configs to DecoderLayerConfig format
        let layer_configs: Box<[DecoderLayerConfig]> = tf
            .layer_configs
            .iter()
            .map(|layer| DecoderLayerConfig {
                pre_attention_norm_config: layer
                    .pre_attention_norm_config
                    .clone()
                    .unwrap_or_else(|| tf.output_norm_config.clone()),
                mixer_config: MixerConfig::Attention(layer.attention_config.clone()),
                post_attention_norm_config: layer.post_attention_norm_config.clone(),
                pre_mlp_norm_config: layer.pre_mlp_norm_config.clone(),
                mlp_config: layer.mlp_config.clone(),
                post_mlp_norm_config: layer.post_mlp_norm_config.clone(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        DecoderConfig {
            embedding_config: self.embedding_config.clone(),
            global_rope_config: Some(tf.global_rope_config.clone()),
            local_rope_config: tf.local_rope_config.clone(),
            layer_config,
            layer_configs: Some(layer_configs),
            output_norm_config: tf.output_norm_config.clone(),
            vocab_size: self.vocab_size,
            model_dim: tf.model_dim,
            hidden_dim: tf.hidden_dim,
            num_heads,
            num_groups,
            head_dim,
            attention_scale,
            num_layers,
            sliding_window_sizes: Some(sliding_window_sizes),
            layer_types: None,
            context_length: tf.context_length,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LanguageModelConfig {
    pub model_config: InnerModelConfig,
    pub message_processor_config: MessageProcessorConfig,
    pub generation_config: GenerationConfig,
}

impl LanguageModelConfig {
    /// Get the decoder config for backward compatibility.
    /// This converts the new format to the old DecoderConfig format.
    pub fn decoder_config(&self) -> DecoderConfig {
        self.model_config.to_decoder_config()
    }
}
