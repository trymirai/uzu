use std::collections::HashMap;

use super::model_shape::ModelShape;
use crate::{
    DataType, DeviceContext,
    config::{DecoderConfig, MLPConfig},
};

#[derive(Debug)]
pub struct ScratchBuffers<C: DeviceContext> {
    // 1-D
    pub token_ids: C::DeviceArray,
    pub token_positions: C::DeviceArray,
    pub token_bitmask: C::DeviceArray,
    pub token_seeds: C::DeviceArray,
    pub sampling_output: C::DeviceArray,

    // 2-D
    pub attention_window_size_to_bias: HashMap<Option<usize>, C::DeviceArray>,
    pub logits: C::DeviceArray,
    pub main: C::DeviceArray,
    pub shortcut: C::DeviceArray,
    pub qkv: C::DeviceArray,
    pub attention_output: C::DeviceArray,
    pub mlp_fused_up: C::DeviceArray,
    pub mlp_hidden: C::DeviceArray,
    pub ssm_inproj: Option<C::DeviceArray>,
    pub ssm_packed: Option<C::DeviceArray>,
    pub ssm_conv_padded: Option<C::DeviceArray>,
    pub ssm_x: Option<C::DeviceArray>,
    pub ssm_b: Option<C::DeviceArray>,
    pub ssm_c: Option<C::DeviceArray>,
    pub ssm_dt: Option<C::DeviceArray>,
    pub ssm_z: Option<C::DeviceArray>,

    // 3-D
    pub rotated_queries: C::DeviceArray,
    pub rotated_keys: C::DeviceArray,
    pub extracted_values: C::DeviceArray,

    // 2-pass attention intermediate buffers
    pub attention_partials: C::DeviceArray, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: C::DeviceArray, // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: C::DeviceArray, // [num_heads * max_suffix_len * total_blocks_count]

    pub moe_topk_ids: Option<C::DeviceArray>,
    pub moe_topk_probs: Option<C::DeviceArray>,
    pub moe_offsets: Option<C::DeviceArray>,
    pub moe_sumk: Option<C::DeviceArray>,
    pub moe_bucketed_token_ids: Option<C::DeviceArray>,
    pub moe_bucketed_probs: Option<C::DeviceArray>,
    pub moe_x_perm: Option<C::DeviceArray>,
    pub moe_tok2row: Option<C::DeviceArray>,
    pub moe_y_partial: Option<C::DeviceArray>,
    pub moe_hidden: Option<C::DeviceArray>,
    pub moe_two_pass_row_expert_map: Option<C::DeviceArray>,
    pub moe_tile_counts: Option<C::DeviceArray>,
    pub moe_tile_offsets: Option<C::DeviceArray>,
    pub moe_tile_map: Option<C::DeviceArray>,
    pub moe_total_tiles: Option<C::DeviceArray>,
    pub moe_dispatch_args: Option<C::DeviceArray>,
    pub moe_scatter_partials: Option<C::DeviceArray>,
    pub moe_scatter_block_bases: Option<C::DeviceArray>,
    pub moe_block_alloc: Option<C::DeviceArray>,
}

impl<C: DeviceContext> ScratchBuffers<C> {
    pub fn new(
        context: &C,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        max_prefix_len: usize,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize], dtype: DataType| -> C::DeviceArray {
            unsafe { context.array_uninitialized(shape, dtype) }
        };

        let act_ty = model_shape.activation_data_type();

        let partials_shape =
            model_shape.attention_partials_shape(max_suffix_len);
        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64),
            token_positions: alloc(&[max_suffix_len], DataType::I32),
            token_bitmask: alloc(
                &model_shape.bitmask_shape(max_suffix_len),
                DataType::U32,
            ),
            token_seeds: alloc(&[max_suffix_len], DataType::U64),
            sampling_output: alloc(&[max_suffix_len], DataType::U32),

            // 2-D
            attention_window_size_to_bias: {
                // Collect unique window sizes across all layers
                let unique_window_sizes: std::collections::HashSet<
                    Option<usize>,
                > = model_shape
                    .sliding_window_length_per_layer
                    .iter()
                    .copied()
                    .collect();

                // Create one buffer per unique window size
                unique_window_sizes
                    .into_iter()
                    .map(|window_size| {
                        (
                            window_size,
                            alloc(
                                &[
                                    max_suffix_len,
                                    max_suffix_len + max_prefix_len,
                                ],
                                act_ty,
                            ),
                        )
                    })
                    .collect::<HashMap<Option<usize>, C::DeviceArray>>()
            },
            logits: alloc(&model_shape.logits_shape(max_suffix_len), act_ty),
            main: alloc(&model_shape.main_shape(max_suffix_len), act_ty),
            shortcut: alloc(&model_shape.main_shape(max_suffix_len), act_ty),
            qkv: alloc(&model_shape.qkv_shape(max_suffix_len), act_ty),
            attention_output: alloc(
                &model_shape.attention_output_shape(max_suffix_len),
                act_ty,
            ),
            mlp_fused_up: alloc(
                &model_shape.mlp_fused_up_shape(max_suffix_len),
                act_ty,
            ),
            mlp_hidden: alloc(
                &model_shape.mlp_hidden_shape(max_suffix_len),
                act_ty,
            ),
            ssm_inproj: model_shape
                .ssm_inproj_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_packed: model_shape
                .ssm_packed_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_conv_padded: model_shape
                .ssm_conv_padded_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_x: model_shape
                .ssm_x_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_b: model_shape
                .ssm_bc_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_c: model_shape
                .ssm_bc_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_dt: model_shape
                .ssm_dt_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            ssm_z: model_shape
                .ssm_z_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty)),
            // 3-D
            rotated_queries: alloc(
                &model_shape.rotated_queries_shape(max_suffix_len),
                act_ty,
            ),
            rotated_keys: alloc(
                &model_shape.rotated_keys_shape(max_suffix_len),
                act_ty,
            ),
            extracted_values: alloc(
                &model_shape.extracted_values_shape(max_suffix_len),
                act_ty,
            ),

            attention_partials: alloc(&partials_shape, act_ty),
            attention_sums: alloc(&sums_maxs_shape, act_ty),
            attention_maxs: alloc(&sums_maxs_shape, act_ty),

            moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_topk_ids_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_topk_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_topk_probs_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, act_ty))
                },
                _ => None,
            },
            moe_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_offsets_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_sumk: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_sumk_shape();
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_bucketed_token_ids: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape =
                        model_shape.moe_bucketed_token_ids_shape(max_routed);
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_bucketed_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape =
                        model_shape.moe_bucketed_probs_shape(max_routed);
                    Some(alloc(&shape, act_ty))
                },
                _ => None,
            },
            moe_x_perm: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_x_perm_shape(max_routed);
                    Some(alloc(&shape, DataType::F16))
                },
                _ => None,
            },
            moe_tok2row: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_tok2row_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, DataType::I32))
                },
                _ => None,
            },
            moe_y_partial: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_y_partial_shape(max_routed);
                    Some(alloc(&shape, DataType::F16))
                },
                _ => None,
            },
            moe_hidden: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_hidden_shape(max_routed);
                    Some(alloc(&shape, DataType::F32))
                },
                _ => None,
            },
            moe_two_pass_row_expert_map: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    Some(alloc(&[max_routed], DataType::U32))
                },
                _ => None,
            },
            moe_tile_counts: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_counts_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_tile_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_offsets_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_tile_map: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_tile_map_shape(max_routed);
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_total_tiles: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_total_tiles_shape();
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_dispatch_args: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_dispatch_args_shape();
                    Some(alloc(&shape, DataType::U32))
                },
                _ => None,
            },
            moe_scatter_partials: match &decoder_config.layer_config.mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    Some(alloc(&[entries], DataType::U32))
                },
                _ => None,
            },
            moe_scatter_block_bases: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    Some(alloc(&[entries], DataType::U32))
                },
                _ => None,
            },
            moe_block_alloc: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    Some(alloc(&[entries], DataType::U32))
                },
                _ => None,
            },
        }
    }
}
