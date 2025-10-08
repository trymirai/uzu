use crate::{DataType, config::DecoderConfig};

pub const MOE_TWO_PASS_K_TILE: usize = 128;

#[derive(Debug)]
pub struct ModelShape {
    activation_type: DataType,
    kv_cache_type: DataType,

    vocabulary_size: usize,
    model_dim: usize,
    hidden_dim: usize,
    context_length: usize,

    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    pub num_layers: usize,
    pub sliding_window_length_per_layer: Box<[Option<usize>]>,
}

impl ModelShape {
    pub fn from_decoder_config(decoder_config: &DecoderConfig) -> Self {
        let activation_type: DataType =
            match &decoder_config.layer_config.mlp_config {
                crate::config::MLPConfig::Dense(d) => {
                    d.linear_config.activation_precision().into()
                },
                crate::config::MLPConfig::MixtureOfExperts(m) => {
                    m.expert_config.linear_config.activation_precision().into()
                },
            };
        let num_layers = decoder_config.num_layers;
        Self {
            activation_type,
            kv_cache_type: activation_type,
            vocabulary_size: decoder_config.vocab_size,
            model_dim: decoder_config.model_dim,
            hidden_dim: decoder_config.hidden_dim,
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

    pub fn mlp_hidden_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.hidden_dim]
    }

    pub fn mlp_fused_up_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, 2 * self.hidden_dim]
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
        [self.vocabulary_size, self.model_dim]
    }

    pub fn quantized_embeddings_weights_shape(&self) -> [usize; 2] {
        [self.vocabulary_size, self.model_dim]
    }

    pub fn quantized_embeddings_scales_shape(&self) -> [usize; 1] {
        [self.vocabulary_size]
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

    pub fn moe_router_logits_shape(
        &self,
        suffix_length: usize,
        num_experts: usize,
    ) -> [usize; 2] {
        [suffix_length, num_experts]
    }

    pub fn moe_topk_ids_shape(
        &self,
        suffix_length: usize,
        k: usize,
    ) -> [usize; 2] {
        [suffix_length, k]
    }

    pub fn moe_topk_probs_shape(
        &self,
        suffix_length: usize,
        k: usize,
    ) -> [usize; 2] {
        [suffix_length, k]
    }

    pub fn moe_counts_shape(
        &self,
        num_experts: usize,
    ) -> [usize; 1] {
        [num_experts]
    }

    pub fn moe_offsets_shape(
        &self,
        num_experts: usize,
    ) -> [usize; 1] {
        [num_experts + 1]
    }

    pub fn moe_sumk_shape(&self) -> [usize; 1] {
        [1]
    }

    pub fn moe_bucketed_token_ids_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 1] {
        [max_routed_tokens]
    }

    pub fn moe_bucketed_probs_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 1] {
        [max_routed_tokens]
    }

    pub fn moe_tok2row_shape(
        &self,
        suffix_length: usize,
        k: usize,
    ) -> [usize; 1] {
        [suffix_length * k]
    }

    pub fn moe_y_partial_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 2] {
        [max_routed_tokens, self.model_dim]
    }

    pub fn moe_hidden_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 2] {
        [max_routed_tokens, self.hidden_dim]
    }

    pub fn moe_two_pass_partial_shape(
        &self,
        max_routed_tokens: usize,
        k_tile: usize,
    ) -> [usize; 2] {
        let num_tiles = (self.hidden_dim + k_tile - 1) / k_tile;
        [max_routed_tokens * num_tiles, self.model_dim]
    }

    pub fn moe_x_perm_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 2] {
        [max_routed_tokens, self.model_dim]
    }

    pub fn moe_tile_map_shape(
        &self,
        max_routed_tokens: usize,
    ) -> [usize; 1] {
        // For tiled GEMM: max_routed_tokens * 3
        // For two-pass decode indirect dispatch: max_routed_tokens * h_blocks * 3
        // where h_blocks = (d_ff + 3) / 4
        // We size for the larger (two-pass decode)
        let d_ff = self.hidden_dim;
        if d_ff > 0 {
            let h_blocks = (d_ff + 3) / 4;
            [max_routed_tokens * h_blocks * 3]
        } else {
            [max_routed_tokens * 3]
        }
    }

    pub fn moe_dispatch_args_shape(&self) -> [usize; 1] {
        [3]
    }

    pub fn moe_total_tiles_shape(&self) -> [usize; 1] {
        // layout: [total_tiles, total_jobs,
        //          nan_fc1_count, nan_fc2_count,
        //          nan_fc1_bits, nan_fc2_bits,
        //          nan_gate_bits, nan_up_bits]
        [8]
    }
}
