use std::collections::HashMap;

use metal::Buffer as MTLBuffer;

use super::{
    super::MTLContext,
    model_shape::{MOE_TWO_PASS_K_TILE, ModelShape},
};
use crate::{
    DataType,
    array::array_size_in_bytes,
    config::{DecoderConfig, MLPConfig},
};

/// Holds one shared Metal buffer for every temporary tensor that can appear during a forward pass.
/// Each buffer is large enough for the worst-case shape: `max_suffix_len` / `max_prefix_len`.
///
/// Per-iteration code merely *wraps* the buffer into a `MetalArray` with the runtime shape.
#[derive(Debug)]
pub struct ForwardPassBuffers {
    // 1-D
    pub token_ids: MTLBuffer,
    pub token_positions: MTLBuffer,
    pub sampling_output: MTLBuffer,

    // 2-D
    pub attention_window_size_to_bias: HashMap<Option<usize>, MTLBuffer>,
    pub logits: MTLBuffer,
    pub main: MTLBuffer,
    pub shortcut: MTLBuffer,
    pub qkv: MTLBuffer,
    pub attention_output: MTLBuffer,
    pub mlp_fused_up: MTLBuffer,
    pub mlp_hidden: MTLBuffer,

    // 3-D
    pub rotated_queries: MTLBuffer,
    pub rotated_keys: MTLBuffer,

    // 2-pass attention intermediate buffers
    pub attention_partials: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: MTLBuffer, // [num_heads * max_suffix_len * total_blocks_count]

    pub moe_topk_ids: Option<MTLBuffer>,
    pub moe_topk_probs: Option<MTLBuffer>,
    pub moe_offsets: Option<MTLBuffer>,
    pub moe_sumk: Option<MTLBuffer>,
    pub moe_bucketed_token_ids: Option<MTLBuffer>,
    pub moe_bucketed_probs: Option<MTLBuffer>,
    pub moe_x_perm: Option<MTLBuffer>,
    pub moe_tok2row: Option<MTLBuffer>,
    pub moe_y_partial: Option<MTLBuffer>,
    pub moe_hidden: Option<MTLBuffer>,
    pub moe_two_pass_partial: Option<MTLBuffer>,
    pub moe_two_pass_row_expert_map: Option<MTLBuffer>,
    pub moe_tile_counts: Option<MTLBuffer>,
    pub moe_tile_offsets: Option<MTLBuffer>,
    pub moe_tile_map: Option<MTLBuffer>,
    pub moe_total_tiles: Option<MTLBuffer>,
    pub moe_dispatch_args: Option<MTLBuffer>,
    pub moe_scatter_partials: Option<MTLBuffer>,
    pub moe_scatter_block_bases: Option<MTLBuffer>,
    pub moe_block_alloc: Option<MTLBuffer>,
}

impl ForwardPassBuffers {
    // TODO: use device arrays instead of MTLBuffers
    /// Allocate the buffers with `StorageModeShared` so that they are CPU-accessible as well.
    pub fn new(
        context: &MTLContext,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        max_prefix_len: usize,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize], dtype: DataType| -> MTLBuffer {
            let bytes = array_size_in_bytes(shape, dtype);
            context.device.new_buffer(
                bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };

        let act_ty = model_shape.activation_data_type();

        let partials_shape =
            model_shape.attention_partials_shape(max_suffix_len);
        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64),
            token_positions: alloc(&[max_suffix_len], DataType::I32),
            sampling_output: alloc(&[max_suffix_len], DataType::U32),

            // 2-D
            attention_window_size_to_bias: (0..model_shape.num_layers)
                .map(|layer| {
                    (
                        model_shape.sliding_window_length_per_layer
                            [layer as usize],
                        alloc(
                            &[max_suffix_len, max_suffix_len + max_prefix_len],
                            act_ty,
                        ),
                    )
                })
                .collect::<HashMap<Option<usize>, MTLBuffer>>(),
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

            // 3-D
            rotated_queries: alloc(
                &model_shape.rotated_queries_shape(max_suffix_len),
                act_ty,
            ),
            rotated_keys: alloc(
                &model_shape.rotated_keys_shape(max_suffix_len),
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
                    Some(alloc(&shape, DataType::F16))
                },
                _ => None,
            },
            moe_two_pass_partial: match &decoder_config.layer_config.mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_two_pass_partial_shape(
                        max_routed,
                        MOE_TWO_PASS_K_TILE,
                    );
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
                    let bytes = (entries * std::mem::size_of::<u32>()) as u64;
                    Some(context.device.new_buffer(
                        bytes,
                        metal::MTLResourceOptions::StorageModeShared,
                    ))
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
                    let bytes = (entries * std::mem::size_of::<u32>()) as u64;
                    Some(context.device.new_buffer(
                        bytes,
                        metal::MTLResourceOptions::StorageModeShared,
                    ))
                },
                _ => None,
            },
            moe_block_alloc: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    let bytes = (entries * std::mem::size_of::<u32>()) as u64;
                    Some(context.device.new_buffer(
                        bytes,
                        metal::MTLResourceOptions::StorageModeShared,
                    ))
                },
                _ => None,
            },
        }
    }
}
