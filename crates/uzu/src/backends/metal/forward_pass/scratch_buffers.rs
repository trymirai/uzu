use std::{cell::RefCell, collections::HashMap};

use super::model_shape::ModelShape;
use crate::{
    DataType, DeviceContext,
    config::{DecoderConfig, MLPConfig},
};

type Array<Context> = RefCell<<Context as DeviceContext>::DeviceArray>;

pub struct ScratchBuffers<Context: DeviceContext> {
    // 1-D
    pub token_ids: Array<Context>,
    pub token_positions: Array<Context>,
    pub token_parents: Array<Context>,
    pub token_bitmask: Array<Context>,
    pub token_seeds: Array<Context>,
    pub sampling_output: Array<Context>,

    // 2-D
    pub attention_window_size_to_bias: HashMap<Option<usize>, Array<Context>>,
    pub logits: Array<Context>,
    pub main: Array<Context>,
    pub shortcut: Array<Context>,
    pub qkv: Array<Context>,
    pub attention_output: Array<Context>,
    pub mlp_fused_up: Array<Context>,
    pub mlp_hidden: Array<Context>,
    pub ssm_inproj: Option<Array<Context>>,
    pub ssm_packed: Option<Array<Context>>,
    pub ssm_conv_padded: Option<Array<Context>>,
    pub short_conv_padded: Option<Array<Context>>,
    pub ssm_x: Option<Array<Context>>,
    pub ssm_b: Option<Array<Context>>,
    pub ssm_c: Option<Array<Context>>,
    pub ssm_dt: Option<Array<Context>>,
    pub ssm_z: Option<Array<Context>>,

    // 3-D
    pub rotated_queries: Array<Context>,
    pub rotated_keys: Array<Context>,
    pub extracted_values: Array<Context>,

    // 2-pass attention intermediate buffers
    pub attention_partials: Array<Context>, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: Array<Context>, // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: Array<Context>, // [num_heads * max_suffix_len * total_blocks_count]

    pub moe_topk_ids: Option<Array<Context>>,
    pub moe_topk_probs: Option<Array<Context>>,
    pub moe_offsets: Option<Array<Context>>,
    pub moe_sumk: Option<Array<Context>>,
    pub moe_bucketed_token_ids: Option<Array<Context>>,
    pub moe_bucketed_probs: Option<Array<Context>>,
    pub moe_x_perm: Option<Array<Context>>,
    pub moe_tok2row: Option<Array<Context>>,
    pub moe_y_partial: Option<Array<Context>>,
    pub moe_hidden: Option<Array<Context>>,
    pub moe_two_pass_row_expert_map: Option<Array<Context>>,
    pub moe_tile_counts: Option<Array<Context>>,
    pub moe_tile_offsets: Option<Array<Context>>,
    pub moe_tile_map: Option<Array<Context>>,
    pub moe_total_tiles: Option<Array<Context>>,
    pub moe_dispatch_args: Option<Array<Context>>,
    pub moe_scatter_partials: Option<Array<Context>>,
    pub moe_scatter_block_bases: Option<Array<Context>>,
    pub moe_block_alloc: Option<Array<Context>>,
}

impl<Context: DeviceContext> ScratchBuffers<Context> {
    // TODO: use device arrays instead of MTLBuffers
    /// Allocate the buffers with `StorageModeShared` so that they are CPU-accessible as well.
    pub fn new(
        context: &Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        max_prefix_len: usize,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize],
                     dtype: DataType,
                     label: &str|
         -> Array<Context> {
            let array =
                context.array(shape, dtype, format!("scratch_buffers_{label}"));
            RefCell::new(array)
        };

        let act_ty = model_shape.activation_data_type();

        let partials_shape =
            model_shape.attention_partials_shape(max_suffix_len);
        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64, "token_ids"),
            token_positions: alloc(
                &[max_suffix_len],
                DataType::I32,
                "token_positions",
            ),
            token_parents: alloc(
                &[max_suffix_len],
                DataType::I32,
                "token_parents",
            ),
            token_bitmask: alloc(
                &model_shape.bitmask_shape(max_suffix_len),
                DataType::U32,
                "token_bitmask",
            ),
            token_seeds: alloc(&[max_suffix_len], DataType::U64, "token_seeds"),
            sampling_output: alloc(
                &[max_suffix_len],
                DataType::U32,
                "sampling_output",
            ),

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
                        (window_size, {
                            let label = match window_size {
                                Some(window_size) => format!(
                                    "attention_bias_for_window_size_{window_size}"
                                ),
                                None => "attention_bias_for_window_size_none".to_string(),
                            };
                            alloc(
                                &[
                                    max_suffix_len,
                                    max_suffix_len + max_prefix_len,
                                ],
                                act_ty,
                                label
                                .as_str(),
                            )
                        })
                    })
                    .collect::<HashMap<Option<usize>, Array<Context>>>()
            },
            logits: alloc(
                &model_shape.logits_shape(max_suffix_len),
                act_ty,
                "logits",
            ),
            main: alloc(
                &model_shape.main_shape(max_suffix_len),
                act_ty,
                "main",
            ),
            shortcut: alloc(
                &model_shape.main_shape(max_suffix_len),
                act_ty,
                "shortcut",
            ),
            qkv: alloc(&model_shape.qkv_shape(max_suffix_len), act_ty, "qkv"),
            attention_output: alloc(
                &model_shape.attention_output_shape(max_suffix_len),
                act_ty,
                "attention_output",
            ),
            mlp_fused_up: alloc(
                &model_shape.mlp_fused_up_shape(max_suffix_len),
                act_ty,
                "mlp_fused_up",
            ),
            mlp_hidden: alloc(
                &model_shape.mlp_hidden_shape(max_suffix_len),
                act_ty,
                "mlp_hidden",
            ),
            ssm_inproj: model_shape
                .ssm_inproj_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_inproj")),
            ssm_packed: model_shape
                .ssm_packed_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_packed")),
            ssm_conv_padded: model_shape
                .ssm_conv_padded_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_conv_padded")),
            short_conv_padded: model_shape
                .short_conv_padded_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "short_conv_padded")),
            ssm_x: model_shape
                .ssm_x_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_x")),
            ssm_b: model_shape
                .ssm_bc_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_b")),
            ssm_c: model_shape
                .ssm_bc_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_c")),
            ssm_dt: model_shape
                .ssm_dt_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_dt")),
            ssm_z: model_shape
                .ssm_z_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_z")),
            // 3-D
            rotated_queries: alloc(
                &model_shape.rotated_queries_shape(max_suffix_len),
                act_ty,
                "rotated_queries",
            ),
            rotated_keys: alloc(
                &model_shape.rotated_keys_shape(max_suffix_len),
                act_ty,
                "rotated_keys",
            ),
            extracted_values: alloc(
                &model_shape.extracted_values_shape(max_suffix_len),
                act_ty,
                "extracted_values",
            ),

            attention_partials: alloc(
                &partials_shape,
                act_ty,
                "attention_partials",
            ),
            attention_sums: alloc(&sums_maxs_shape, act_ty, "attention_sums"),
            attention_maxs: alloc(&sums_maxs_shape, act_ty, "attention_maxs"),

            moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_topk_ids_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, DataType::U32, "moe_topk_ids"))
                },
                _ => None,
            },
            moe_topk_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_topk_probs_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, act_ty, "moe_topk_probs"))
                },
                _ => None,
            },
            moe_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_offsets_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32, "moe_offsets"))
                },
                _ => None,
            },
            moe_sumk: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_sumk_shape();
                    Some(alloc(&shape, DataType::U32, "moe_sumk"))
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
                    Some(alloc(&shape, DataType::U32, "moe_bucketed_token_ids"))
                },
                _ => None,
            },
            moe_bucketed_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape =
                        model_shape.moe_bucketed_probs_shape(max_routed);
                    Some(alloc(&shape, act_ty, "moe_bucketed_probs"))
                },
                _ => None,
            },
            moe_x_perm: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_x_perm_shape(max_routed);
                    Some(alloc(&shape, DataType::F16, "moe_x_perm"))
                },
                _ => None,
            },
            moe_tok2row: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_tok2row_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    );
                    Some(alloc(&shape, DataType::I32, "moe_tok2row"))
                },
                _ => None,
            },
            moe_y_partial: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_y_partial_shape(max_routed);
                    Some(alloc(&shape, DataType::F16, "moe_y_partial"))
                },
                _ => None,
            },
            moe_hidden: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_hidden_shape(max_routed);
                    Some(alloc(&shape, DataType::F32, "moe_hidden"))
                },
                _ => None,
            },
            moe_two_pass_row_expert_map: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    Some(alloc(
                        &[max_routed],
                        DataType::U32,
                        "moe_two_pass_row_expert_map",
                    ))
                },
                _ => None,
            },
            moe_tile_counts: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_counts_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32, "moe_tile_counts"))
                },
                _ => None,
            },
            moe_tile_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let shape = model_shape.moe_offsets_shape(moe.mixture_size);
                    Some(alloc(&shape, DataType::U32, "moe_tile_offsets"))
                },
                _ => None,
            },
            moe_tile_map: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = max_suffix_len * moe.num_experts_per_token;
                    let shape = model_shape.moe_tile_map_shape(max_routed);
                    Some(alloc(&shape, DataType::U32, "moe_tile_map"))
                },
                _ => None,
            },
            moe_total_tiles: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_total_tiles_shape();
                    Some(alloc(&shape, DataType::U32, "moe_total_tiles"))
                },
                _ => None,
            },
            moe_dispatch_args: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    let shape = model_shape.moe_dispatch_args_shape();
                    Some(alloc(&shape, DataType::U32, "moe_dispatch_args"))
                },
                _ => None,
            },
            moe_scatter_partials: match &decoder_config.layer_config.mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    Some(alloc(
                        &[entries],
                        DataType::U32,
                        "moe_scatter_partials",
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
                    Some(alloc(
                        &[entries],
                        DataType::U32,
                        "moe_scatter_block_bases",
                    ))
                },
                _ => None,
            },
            moe_block_alloc: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let num_blocks = ((max_suffix_len + 255) / 256).max(1);
                    let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
                    let entries = num_blocks * num_tiles * 512;
                    Some(alloc(&[entries], DataType::U32, "moe_block_alloc"))
                },
                _ => None,
            },
        }
    }
}
