use crate::{
    DataType,
    config::{DecoderConfig, MLPConfig, MixerConfig},
};

#[derive(Debug, Clone)]
pub struct ModelShape {
    activation_type: DataType,
    kv_cache_type: DataType,

    vocabulary_size: usize,
    model_dim: usize,
    context_length: usize,

    num_groups: usize,
    head_dim: usize,
    pub num_layers: usize,
    pub sliding_window_length_per_layer: Box<[Option<usize>]>,
    layer_mixers: Box<[MixerConfig]>,
}

impl ModelShape {
    pub fn from_decoder_config(decoder_config: &DecoderConfig) -> Self {
        let tf = &decoder_config.transformer_config;
        let layer_configs = &tf.layer_configs;
        let num_layers = layer_configs.len();

        let (num_heads, num_groups, head_dim) =
            decoder_config.first_attention().map(|a| (a.num_heads, a.num_groups, a.head_dim)).unwrap_or_default();
        for attn in layer_configs.iter().filter_map(|l| l.mixer_config.as_attention()) {
            assert_eq!(attn.num_heads, num_heads, "attention layers must share num_heads");
            assert_eq!(attn.num_groups, num_groups, "attention layers must share num_groups");
            assert_eq!(attn.head_dim, head_dim, "attention layers must share head_dim");
        }

        let activation_type: DataType = match &layer_configs[0].mlp_config {
            MLPConfig::Dense(d) => d.linear_config.activation_precision().into(),
            MLPConfig::MixtureOfExperts(m) => m.expert_config.linear_config.activation_precision().into(),
        };

        let sliding_window_length_per_layer: Box<[Option<usize>]> =
            layer_configs.iter().map(|l| l.mixer_config.sliding_window_size()).collect();
        let layer_mixers: Box<[MixerConfig]> = layer_configs.iter().map(|l| l.mixer_config.clone()).collect();

        Self {
            activation_type,
            kv_cache_type: activation_type,
            vocabulary_size: decoder_config.vocab_size,
            model_dim: tf.model_dim,
            context_length: tf.context_length,
            num_groups,
            head_dim,
            num_layers,
            sliding_window_length_per_layer,
            layer_mixers,
        }
    }

    pub fn activation_data_type(&self) -> DataType {
        self.activation_type
    }

    pub fn kv_cache_data_type(&self) -> DataType {
        self.kv_cache_type
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn model_dim(&self) -> usize {
        self.model_dim
    }

    pub fn layer_mixers(&self) -> &[MixerConfig] {
        &self.layer_mixers
    }

    pub fn main_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.model_dim]
    }

    pub fn subtrie_ranges_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, 3]
    }

    pub fn bitmask_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        let bitmask_size = (self.vocabulary_size + 31) / 32;
        [suffix_length, bitmask_size]
    }

    pub fn logits_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.vocabulary_size]
    }

    pub fn kv_cache_layer_shapes(
        &self,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> impl Iterator<Item = [usize; 3]> {
        self.sliding_window_length_per_layer.iter().map(move |length| {
            let length = length.unwrap_or(max_prefix_length);
            [length + max_suffix_length, self.num_groups, self.head_dim]
        })
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }
}
