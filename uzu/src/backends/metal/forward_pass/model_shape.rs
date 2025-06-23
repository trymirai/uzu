use crate::{DataType, config::DecoderConfig};

#[derive(Debug)]
pub struct ModelShape {
    activation_type: DataType,
    kv_cache_type: DataType,

    vocabulary_size: usize,
    model_dim: usize,
    context_length: usize,

    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    pub num_layers: usize,
    pub sliding_window_length_per_layer: Box<[Option<usize>]>,
}

impl ModelShape {
    pub fn from_decoder_config(decoder_config: &DecoderConfig) -> Self {
        let activation_type: DataType = decoder_config
            .layer_config
            .mlp_config
            .linear_config
            .activation_precision()
            .into();
        let num_layers = decoder_config.num_layers;
        Self {
            activation_type,
            kv_cache_type: activation_type,
            vocabulary_size: decoder_config.vocab_size,
            model_dim: decoder_config.model_dim,
            context_length: decoder_config.context_length,
            num_heads: decoder_config.num_heads,
            num_groups: decoder_config.num_groups,
            head_dim: decoder_config.head_dim,
            num_layers: num_layers,
            sliding_window_length_per_layer: decoder_config
                .sliding_window_sizes
                .clone()
                .unwrap_or(vec![None; num_layers].into_boxed_slice()),
        }
    }

    pub fn activation_data_type(&self) -> DataType {
        self.activation_type
    }

    pub fn kv_cache_data_type(&self) -> DataType {
        self.kv_cache_type
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn main_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.model_dim]
    }

    pub fn qkv_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, (2 * self.num_groups + self.num_heads) * self.head_dim]
    }

    pub fn logits_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.vocabulary_size]
    }

    pub fn embeddings_input_shape(&self) -> [usize; 2] {
        [self.vocabulary_size, self.model_dim]
    }

    pub fn embeddings_output_shape(&self) -> [usize; 2] {
        [self.model_dim, self.vocabulary_size]
    }

    pub fn attention_output_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.num_heads * self.head_dim]
    }

    pub fn rotated_queries_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 3] {
        [self.num_heads, suffix_length, self.head_dim]
    }

    pub fn rotated_keys_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 3] {
        [self.num_groups, suffix_length, self.head_dim]
    }

    pub fn kv_cache_layer_shapes(
        &self,
        max_prefix_length: usize,
        max_suffix_length: usize,
    ) -> impl Iterator<Item = [usize; 3]> {
        self.sliding_window_length_per_layer.iter().map(move |length| {
            let length = length.unwrap_or(max_prefix_length);
            [self.num_groups, length + max_suffix_length, self.head_dim]
        })
    }

    pub fn context_length(&self) -> usize {
        self.context_length
    }

    pub fn attention_partials_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 1] {
        const TOTAL_BLOCKS_COUNT: usize = 32;
        [self.num_heads * suffix_length * TOTAL_BLOCKS_COUNT * self.head_dim]
    }

    pub fn attention_sums_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 1] {
        const TOTAL_BLOCKS_COUNT: usize = 32;
        [self.num_heads * suffix_length * TOTAL_BLOCKS_COUNT]
    }

    pub fn attention_maxs_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 1] {
        const TOTAL_BLOCKS_COUNT: usize = 32;
        [self.num_heads * suffix_length * TOTAL_BLOCKS_COUNT]
    }
}
