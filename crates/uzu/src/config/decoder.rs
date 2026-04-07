use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
};

use super::{
    decoder_layer::{DecoderLayerConfig, MixerConfig},
    embedding::EmbeddingConfig,
    linear::LinearConfig,
    normalization::NormalizationConfig,
    rope::RoPEConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecoderLayerType {
    Transformer,
    #[serde(rename = "ssm")]
    StateSpace {
        conv_dim: usize,
        kernel_size: usize,
        state_dim: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
    },
    #[serde(rename = "short_conv")]
    ShortConv {
        kernel_size: usize,
    },
    #[serde(rename = "delta_net")]
    DeltaNet {
        conv_dim: usize,
        kernel_size: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        value_head_dim: usize,
    },
}

#[derive(Debug, Serialize, PartialEq, Clone)]
pub struct DecoderConfig {
    pub embedding_config: EmbeddingConfig,
    pub global_rope_config: Option<RoPEConfig>,
    pub local_rope_config: Option<RoPEConfig>,
    pub layer_config: DecoderLayerConfig,
    pub layer_configs: Option<Box<[DecoderLayerConfig]>>,
    pub output_norm_config: NormalizationConfig,

    pub vocab_size: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub attention_scale: Option<f32>,
    pub num_layers: usize,
    pub sliding_window_sizes: Option<Box<[Option<usize>]>>,
    pub hidden_dims: Option<Box<[usize]>>,
    pub layer_types: Option<Box<[DecoderLayerType]>>,
    pub context_length: usize,

    /// For each layer, optionally the index of the layer whose KV cache to reuse.
    pub kv_shared_layer_sources: Option<Box<[Option<usize>]>>,

    pub ple_dim: Option<usize>,
    pub ple_embed_scale: Option<f32>,
    pub ple_projection_scale: Option<f32>,
    pub ple_combination_scale: Option<f32>,
    pub ple_linear_config: Option<LinearConfig>,
    pub ple_norm_config: Option<NormalizationConfig>,
}

impl<'de> Deserialize<'de> for DecoderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawDecoderConfig::deserialize(deserializer)?;
        let RawDecoderConfig {
            embedding_config,
            global_rope_config,
            local_rope_config,
            layer_config,
            layer_configs,
            output_norm_config,
            vocab_size,
            model_dim,
            hidden_dim,
            num_heads,
            num_groups,
            head_dim,
            attention_scale,
            num_layers,
            sliding_window_sizes,
            hidden_dims,
            layer_types,
            context_length,
            kv_shared_layer_sources,
            ple_dim,
            ple_embed_scale,
            ple_projection_scale,
            ple_combination_scale,
            ple_linear_config,
            ple_norm_config,
        } = raw;

        let layer_configs_boxed = layer_configs.map(|layers| layers.into_boxed_slice());

        let layer_config_value = if let Some(config) = layer_config {
            config
        } else if let Some(configs) = layer_configs_boxed.as_ref() {
            configs.first().cloned().ok_or_else(|| de::Error::custom("layer_configs must not be empty"))?
        } else {
            return Err(de::Error::custom("decoder config must include layer_config or layer_configs"));
        };

        let num_layers_value = if let Some(value) = num_layers {
            value
        } else if let Some(configs) = layer_configs_boxed.as_ref() {
            configs.len()
        } else {
            return Err(de::Error::custom("num_layers missing and layer_configs not provided"));
        };

        let (num_heads_value, num_groups_value, head_dim_value) = match (num_heads, num_groups, head_dim) {
            (Some(h), Some(g), Some(d)) => (h, g, d),
            _ => derive_dims_from_layer(&layer_config_value).ok_or_else(|| {
                de::Error::custom(
                    "num_heads/num_groups/head_dim missing and \
                             cannot be derived from layer config",
                )
            })?,
        };

        let attention_scale_value = if attention_scale.is_some() {
            attention_scale
        } else {
            layer_config_value.mixer_config.attention_scale()
        };

        let sliding_window_sizes_boxed = if let Some(sizes) = sliding_window_sizes {
            Some(sizes.into_boxed_slice())
        } else if let Some(configs) = layer_configs_boxed.as_ref() {
            Some(
                configs
                    .iter()
                    .map(|layer| layer.mixer_config.sliding_window_size())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else {
            None
        };

        let hidden_dims = hidden_dims.map(|v| v.into_boxed_slice());
        let kv_shared_layer_sources = kv_shared_layer_sources.map(|v| v.into_boxed_slice());

        let explicit_layer_types = layer_types.map(|types| types.into_boxed_slice());
        let derived_layer_types = if let Some(configs) = layer_configs_boxed.as_ref() {
            Some(configs.iter().map(layer_type_from_config).collect::<Vec<_>>().into_boxed_slice())
        } else {
            None
        };
        let layer_types_value = explicit_layer_types.or(derived_layer_types);

        Ok(Self {
            embedding_config,
            global_rope_config,
            local_rope_config,
            layer_config: layer_config_value,
            layer_configs: layer_configs_boxed,
            output_norm_config,
            vocab_size,
            model_dim,
            hidden_dim,
            num_heads: num_heads_value,
            num_groups: num_groups_value,
            head_dim: head_dim_value,
            attention_scale: attention_scale_value,
            num_layers: num_layers_value,
            sliding_window_sizes: sliding_window_sizes_boxed,
            hidden_dims,
            layer_types: layer_types_value,
            context_length,
            kv_shared_layer_sources,
            ple_dim,
            ple_embed_scale,
            ple_projection_scale,
            ple_combination_scale,
            ple_linear_config,
            ple_norm_config,
        })
    }
}

#[derive(Deserialize)]
struct RawDecoderConfig {
    embedding_config: EmbeddingConfig,
    #[serde(default)]
    global_rope_config: Option<RoPEConfig>,
    #[serde(default)]
    local_rope_config: Option<RoPEConfig>,
    #[serde(default)]
    layer_config: Option<DecoderLayerConfig>,
    #[serde(default)]
    layer_configs: Option<Vec<DecoderLayerConfig>>,
    output_norm_config: NormalizationConfig,
    vocab_size: usize,
    model_dim: usize,
    hidden_dim: usize,
    #[serde(default)]
    num_heads: Option<usize>,
    #[serde(default)]
    num_groups: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    attention_scale: Option<f32>,
    #[serde(default)]
    num_layers: Option<usize>,
    #[serde(default)]
    sliding_window_sizes: Option<Vec<Option<usize>>>,
    #[serde(default)]
    hidden_dims: Option<Vec<usize>>,
    #[serde(default)]
    layer_types: Option<Vec<DecoderLayerType>>,
    context_length: usize,
    #[serde(default)]
    kv_shared_layer_sources: Option<Vec<Option<usize>>>,
    #[serde(default)]
    ple_dim: Option<usize>,
    #[serde(default)]
    ple_embed_scale: Option<f32>,
    #[serde(default)]
    ple_projection_scale: Option<f32>,
    #[serde(default)]
    ple_combination_scale: Option<f32>,
    #[serde(default)]
    ple_linear_config: Option<LinearConfig>,
    #[serde(default)]
    ple_norm_config: Option<NormalizationConfig>,
}

fn derive_dims_from_layer(layer: &DecoderLayerConfig) -> Option<(usize, usize, usize)> {
    Some((layer.mixer_config.num_heads()?, layer.mixer_config.num_groups()?, layer.mixer_config.head_dim()?))
}

fn layer_type_from_config(layer: &DecoderLayerConfig) -> DecoderLayerType {
    match &layer.mixer_config {
        MixerConfig::Attention(_) => DecoderLayerType::Transformer,
        MixerConfig::Mamba(config) => DecoderLayerType::StateSpace {
            conv_dim: config.conv_dim(),
            kernel_size: config.kernel_size,
            state_dim: config.state_dim,
            num_heads: config.num_heads,
            num_groups: config.num_groups,
            head_dim: config.head_dim,
        },
        MixerConfig::ShortConv(config) => DecoderLayerType::ShortConv {
            kernel_size: config.kernel_size,
        },
        MixerConfig::DeltaNet(config) => DecoderLayerType::DeltaNet {
            conv_dim: config.conv_dim(),
            kernel_size: config.kernel_size,
            num_heads: config.num_heads,
            num_groups: config.num_groups,
            head_dim: config.head_dim,
            value_head_dim: config.value_head_dim,
        },
    }
}
impl DecoderConfig {
    pub fn group_size(&self) -> usize {
        self.num_heads * self.num_groups
    }

    pub fn has_attention_layers(&self) -> bool {
        self.layer_configs
            .as_ref()
            .map(|configs| {
                configs.iter().any(|config| matches!(config.mixer_config, crate::config::MixerConfig::Attention(_)))
            })
            .unwrap_or(true)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/decoder_test.rs"]
mod tests;
