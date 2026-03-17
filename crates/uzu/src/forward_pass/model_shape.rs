use crate::{
    DataType,
    config::{DecoderConfig, DecoderLayerType, MixerConfig},
};

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
    max_delta_net_kernel_size: usize,
    max_delta_net_padded_stride: usize,
    max_attention_qkv_dim: usize,
    max_attention_output_dim: usize,
    max_attention_num_heads: usize,
    max_attention_num_groups: usize,
    max_attention_head_dim: usize,
    max_attention_rope_dim: usize,
}

impl ModelShape {
    pub fn from_decoder_config(decoder_config: &DecoderConfig) -> Self {
        let activation_type: DataType = match &decoder_config.layer_config.mlp_config {
            crate::config::MLPConfig::Dense(d) => d.linear_config.activation_precision().into(),
            crate::config::MLPConfig::MixtureOfExperts(m) => {
                m.expert_config.linear_config.activation_precision().into()
            },
        };
        let num_layers = decoder_config.num_layers;
        let layer_types: Box<[DecoderLayerType]> = if let Some(layer_types) = &decoder_config.layer_types {
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
        let mut max_delta_net_kernel_size = 0;
        let mut max_delta_net_padded_stride = 0;
        for layer in layer_types.iter() {
            match layer {
                DecoderLayerType::StateSpace {
                    conv_dim,
                    kernel_size,
                    state_dim,
                    num_heads,
                    num_groups,
                    head_dim,
                    ..
                } => {
                    max_mamba_heads = max_mamba_heads.max(*num_heads);
                    max_mamba_groups = max_mamba_groups.max(*num_groups);
                    max_mamba_head_dim = max_mamba_head_dim.max(*head_dim);
                    max_mamba_conv_dim = max_mamba_conv_dim.max(*conv_dim);
                    max_mamba_state_dim = max_mamba_state_dim.max(*state_dim);
                    max_mamba_kernel_size = max_mamba_kernel_size.max(*kernel_size as usize);
                },
                DecoderLayerType::DeltaNet {
                    conv_dim,
                    kernel_size,
                    ..
                } => {
                    max_delta_net_kernel_size = max_delta_net_kernel_size.max(*kernel_size);
                    max_delta_net_padded_stride = max_delta_net_padded_stride.max(*conv_dim);
                },
                _ => {},
            }
        }
        let mut max_attention_qkv_dim = 0usize;
        let mut max_attention_output_dim = 0usize;
        let mut max_attention_num_heads = 0usize;
        let mut max_attention_num_groups = 0usize;
        let mut max_attention_head_dim = 0usize;
        let mut max_attention_rope_dim = 0usize;

        let all_layer_configs = decoder_config
            .layer_configs
            .as_ref()
            .map(|configs| configs.iter().collect::<Vec<_>>())
            .unwrap_or_else(|| vec![&decoder_config.layer_config; num_layers]);

        for layer_config in &all_layer_configs {
            if let MixerConfig::Attention(attn) = &layer_config.mixer_config {
                let nh = attn.num_heads.unwrap_or(decoder_config.num_heads);
                let ng = attn.num_groups.unwrap_or(decoder_config.num_groups);
                let hd = attn.head_dim.unwrap_or(decoder_config.head_dim);
                let rope_dim = attn.partial_rope_dim.unwrap_or(hd);
                let gate_dim = if attn.has_gate {
                    nh * hd
                } else {
                    0
                };
                let qkv_dim = (nh + ng + ng) * hd + gate_dim;
                max_attention_qkv_dim = max_attention_qkv_dim.max(qkv_dim);
                max_attention_output_dim = max_attention_output_dim.max(nh * hd);
                max_attention_num_heads = max_attention_num_heads.max(nh);
                max_attention_num_groups = max_attention_num_groups.max(ng);
                max_attention_head_dim = max_attention_head_dim.max(hd);
                max_attention_rope_dim = max_attention_rope_dim.max(rope_dim);
            }
        }

        if max_attention_qkv_dim == 0 {
            let nh = decoder_config.num_heads;
            let ng = decoder_config.num_groups;
            let hd = decoder_config.head_dim;
            max_attention_qkv_dim = (nh + ng + ng) * hd;
            max_attention_output_dim = nh * hd;
            max_attention_num_heads = nh;
            max_attention_num_groups = ng;
            max_attention_head_dim = hd;
            max_attention_rope_dim = hd;
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
            max_delta_net_kernel_size,
            max_delta_net_padded_stride,
            max_attention_qkv_dim,
            max_attention_output_dim,
            max_attention_num_heads,
            max_attention_num_groups,
            max_attention_head_dim,
            max_attention_rope_dim,
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

    pub fn model_dim(&self) -> usize {
        self.model_dim
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

    pub fn bitmask_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        let bitmask_size = (self.vocabulary_size + 31) / 32;
        [suffix_length, bitmask_size]
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
        [suffix_length, self.max_attention_qkv_dim]
    }

    pub fn logits_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        [suffix_length, self.vocabulary_size]
    }

    pub fn attention_output_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 2] {
        let max_delta_net_conv_dim = self
            .layer_types
            .iter()
            .filter_map(|lt| match lt {
                DecoderLayerType::DeltaNet {
                    conv_dim,
                    ..
                } => Some(*conv_dim),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let dim = self.max_attention_output_dim.max(max_delta_net_conv_dim);
        [suffix_length, dim]
    }

    pub fn rotated_queries_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 3] {
        [self.max_attention_num_heads, suffix_length, self.max_attention_head_dim]
    }

    pub fn rotated_keys_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 3] {
        [self.max_attention_num_groups, suffix_length, self.max_attention_head_dim]
    }

    pub fn extracted_values_shape(
        &self,
        suffix_length: usize,
    ) -> [usize; 3] {
        // Values share the same grouping as keys (grouped by num_groups)
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

    pub fn has_short_conv_layers(&self) -> bool {
        self.layer_types.iter().any(|layer_type| matches!(layer_type, DecoderLayerType::ShortConv { .. }))
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
        let mamba_kernel = self.max_mamba_kernel_size;
        let delta_net_kernel = self.max_delta_net_kernel_size;
        let max_kernel_size = mamba_kernel.max(delta_net_kernel);
        if max_kernel_size == 0 {
            return None;
        }
        // Row stride: Mamba uses conv_dim, DeltaNet uses total_proj_dim
        let max_stride = self.max_mamba_conv_dim.max(self.max_delta_net_padded_stride);
        Some([suffix_length + max_kernel_size - 1, max_stride])
    }

    pub fn short_conv_padded_shape(
        &self,
        suffix_length: usize,
    ) -> Option<[usize; 2]> {
        if !self.has_short_conv_layers() {
            return None;
        }

        let max_kernel_size = self
            .layer_types
            .iter()
            .filter_map(|layer_type| match layer_type {
                DecoderLayerType::ShortConv {
                    kernel_size,
                } => Some(*kernel_size),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        let max_state_stride = max_kernel_size.saturating_sub(1);
        Some([suffix_length + max_state_stride, self.model_dim])
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
            self.max_mamba_state_dim().map(|state_dim| [suffix_length, self.max_mamba_groups, state_dim])
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
                DecoderLayerType::ShortConv {
                    ..
                } => Some(self.model_dim * 3),
                DecoderLayerType::DeltaNet {
                    num_heads,
                    num_groups,
                    head_dim,
                    value_head_dim,
                    ..
                } => {
                    let key_dim = num_groups * head_dim;
                    let value_dim = num_heads * value_head_dim;
                    Some(key_dim + key_dim + value_dim + value_dim + num_heads + num_heads)
                },
                _ => None,
            })
            .max()
    }

    pub fn rope_dim(&self) -> usize {
        self.max_attention_rope_dim
    }

    pub fn has_delta_net_layers(&self) -> bool {
        self.layer_types.iter().any(|layer_type| matches!(layer_type, DecoderLayerType::DeltaNet { .. }))
    }
}
