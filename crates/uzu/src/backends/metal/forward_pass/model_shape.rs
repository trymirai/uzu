use crate::{
    DataType,
    config::{DecoderConfig, DecoderLayerType},
};

pub const SSM_PREFILL_CHUNK: usize = 64;

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
    pub layer_types: Box<[DecoderLayerType]>,
    max_mamba_heads: usize,
    max_mamba_groups: usize,
    max_mamba_head_dim: usize,
    max_mamba_conv_dim: usize,
    max_mamba_state_dim: usize,
    max_mamba_kernel_size: usize,
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
        let layer_types: Box<[DecoderLayerType]> = if let Some(layer_types) =
            &decoder_config.layer_types
        {
            assert_eq!(
                layer_types.len(),
                num_layers,
                "layer_types entry count ({}) must equal num_layers ({})",
                layer_types.len(),
                num_layers
            );
            layer_types.clone()
        } else {
            vec![DecoderLayerType::Transformer; num_layers].into_boxed_slice()
        };
        let mut max_mamba_heads = 0;
        let mut max_mamba_groups = 0;
        let mut max_mamba_head_dim = 0;
        let mut max_mamba_conv_dim = 0;
        let mut max_mamba_state_dim = 0;
        let mut max_mamba_kernel_size = 0;
        for layer in layer_types.iter() {
            if let DecoderLayerType::StateSpace {
                conv_dim,
                kernel_size,
                state_dim,
                num_heads,
                num_groups,
                head_dim,
                ..
            } = layer
            {
                max_mamba_heads = max_mamba_heads.max(*num_heads);
                max_mamba_groups = max_mamba_groups.max(*num_groups);
                max_mamba_head_dim = max_mamba_head_dim.max(*head_dim);
                max_mamba_conv_dim = max_mamba_conv_dim.max(*conv_dim);
                max_mamba_state_dim = max_mamba_state_dim.max(*state_dim);
                max_mamba_kernel_size =
                    max_mamba_kernel_size.max(*kernel_size as usize);
            }
        }
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
            layer_types,
            max_mamba_heads,
            max_mamba_groups,
            max_mamba_head_dim,
            max_mamba_conv_dim,
            max_mamba_state_dim,
            max_mamba_kernel_size,
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

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn layer_type(
        &self,
        layer_index: usize,
    ) -> &DecoderLayerType {
        &self.layer_types[layer_index]
    }

    pub fn layer_types(&self) -> &[DecoderLayerType] {
        &self.layer_types
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

    pub fn ssm_inproj_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        self.max_mamba_inproj_dim_internal().map(|dim| [suffix_length, dim])
    }

    pub fn has_state_space_layers(&self) -> bool {
        self.max_mamba_heads > 0
    }

    pub fn max_mamba_conv_dim(&self) -> Option<usize> {
        if self.max_mamba_conv_dim == 0 {
            None
        } else {
            Some(self.max_mamba_conv_dim)
        }
    }

    pub fn max_mamba_state_dim(&self) -> Option<usize> {
        if self.max_mamba_state_dim == 0 {
            None
        } else {
            Some(self.max_mamba_state_dim)
        }
    }

    pub fn max_mamba_kernel_size(&self) -> Option<usize> {
        if self.max_mamba_kernel_size == 0 {
            None
        } else {
            Some(self.max_mamba_kernel_size)
        }
    }

    pub fn ssm_packed_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        self.max_mamba_conv_dim().map(|dim| [suffix_length, dim])
    }

    pub fn ssm_conv_padded_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        match (self.max_mamba_conv_dim(), self.max_mamba_kernel_size()) {
            (Some(conv_dim), Some(kernel_size)) if kernel_size > 0 => {
                Some([suffix_length + kernel_size - 1, conv_dim])
            },
            _ => None,
        }
    }

    pub fn ssm_x_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            Some([suffix_length, self.max_mamba_heads, self.max_mamba_head_dim])
        } else {
            None
        }
    }

    pub fn ssm_z_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        self.ssm_x_shape(suffix_length)
    }

    pub fn ssm_dt_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        if self.has_state_space_layers() {
            Some([suffix_length, self.max_mamba_heads])
        } else {
            None
        }
    }

    pub fn ssm_bc_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            self.max_mamba_state_dim().map(|state_dim| {
                [suffix_length, self.max_mamba_groups, state_dim]
            })
        } else {
            None
        }
    }

    pub fn ssm_chunk_count(
        &self,
        suffix_length: usize,
    ) -> Option<usize> {
        if self.has_state_space_layers() {
            Some((suffix_length + SSM_PREFILL_CHUNK - 1) / SSM_PREFILL_CHUNK)
        } else {
            None
        }
    }

    pub fn ssm_matrix_dt_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        self.ssm_dt_shape(suffix_length)
    }

    pub fn ssm_matrix_chunk_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        self.ssm_chunk_count(suffix_length)
            .map(|chunks| [chunks, self.max_mamba_heads])
    }

    pub fn ssm_matrix_group_pack_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            self.max_mamba_state_dim().map(|state_dim| {
                [self.max_mamba_groups, suffix_length, state_dim]
            })
        } else {
            None
        }
    }

    pub fn ssm_matrix_group_pack_transposed_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            self.max_mamba_state_dim().map(|state_dim| {
                [self.max_mamba_groups, state_dim, suffix_length]
            })
        } else {
            None
        }
    }

    pub fn ssm_matrix_group_square_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            Some([self.max_mamba_groups, suffix_length, suffix_length])
        } else {
            None
        }
    }

    pub fn ssm_matrix_head_square_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            Some([self.max_mamba_heads, suffix_length, suffix_length])
        } else {
            None
        }
    }

    pub fn ssm_matrix_decay_last_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        if self.has_state_space_layers() {
            Some([self.max_mamba_heads, suffix_length])
        } else {
            None
        }
    }

    pub fn ssm_matrix_b_head_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            self.max_mamba_state_dim().map(|state_dim| {
                [self.max_mamba_heads, suffix_length, state_dim]
            })
        } else {
            None
        }
    }

    pub fn ssm_matrix_dtx_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            Some([self.max_mamba_heads, suffix_length, self.max_mamba_head_dim])
        } else {
            None
        }
    }

    pub fn ssm_matrix_dtxdecay_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            Some([self.max_mamba_heads, self.max_mamba_head_dim, suffix_length])
        } else {
            None
        }
    }

    pub fn ssm_matrix_c_transposed_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 3]> {
        if self.has_state_space_layers() {
            self.max_mamba_state_dim().map(|state_dim| {
                [self.max_mamba_heads, state_dim, suffix_length]
            })
        } else {
            None
        }
    }

    fn max_mamba_inproj_dim_internal(&self) -> Option<usize> {
        self.layer_types
            .iter()
            .filter_map(|layer_type| match layer_type {
                DecoderLayerType::StateSpace {
                    conv_dim,
                    num_heads,
                    head_dim,
                    ..
                } => {
                    let inner_dim = num_heads * head_dim;
                    Some(conv_dim + inner_dim + num_heads)
                },
                _ => None,
            })
            .max()
    }
}
