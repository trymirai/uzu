use crate::{
    config::{decoder::DecoderConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
};

#[derive(Debug, Clone)]
pub struct ModelShape {
    pub data_type: DataType,
    pub rope_data_type: DataType,

    vocabulary_size: usize,
    model_dim: usize,

    num_groups: usize,
    pub num_layers: usize,
    layer_mixers: Box<[AnyTokenMixerConfig]>,
    kv_source_layers: Box<[Option<usize>]>,
}

impl ModelShape {
    pub fn from_decoder_config(
        decoder_config: &DecoderConfig,
        data_type: DataType,
    ) -> Self {
        let tf = &decoder_config.transformer_config;
        let layer_configs = &tf.layer_configs;
        let num_layers = layer_configs.len();

        let (num_heads, num_groups) =
            decoder_config.first_attention().map(|a| (a.num_heads, a.num_groups)).unwrap_or_default();
        for attn in layer_configs.iter().filter_map(|l| l.mixer_config.as_attention()) {
            assert_eq!(attn.num_heads, num_heads, "attention layers must share num_heads");
            assert_eq!(attn.num_groups, num_groups, "attention layers must share num_groups");
        }

        let layer_mixers: Box<[AnyTokenMixerConfig]> = layer_configs.iter().map(|l| l.mixer_config.clone()).collect();
        let kv_source_layers: Box<[Option<usize>]> = layer_configs.iter().map(|l| l.kv_source_layer_index).collect();

        Self {
            data_type,
            rope_data_type: DataType::F32,
            vocabulary_size: decoder_config.vocab_size,
            model_dim: tf.model_dim,
            num_groups,
            num_layers,
            layer_mixers,
            kv_source_layers,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn model_dim(&self) -> usize {
        self.model_dim
    }

    pub fn kv_source_layers(&self) -> &[Option<usize>] {
        &self.kv_source_layers
    }

    pub fn layer_mixers(&self) -> &[AnyTokenMixerConfig] {
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
}
