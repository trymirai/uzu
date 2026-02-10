use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use crate::{
    DataType,
    array::{ArrayCell, ArrayContextExt},
    backends::common::Backend,
    config::{DecoderConfig, MLPConfig},
    forward_pass::model_shape::ModelShape,
};

pub struct ScratchBuffers<B: Backend> {
    // 1-D
    pub token_ids: ArrayCell<B>,
    pub token_positions: ArrayCell<B>,
    pub token_bitmask: ArrayCell<B>,
    pub token_seeds: ArrayCell<B>,
    pub sampling_output: ArrayCell<B>,

    // 2-D
    pub attention_window_size_to_bias: HashMap<Option<usize>, ArrayCell<B>>,
    pub logits: ArrayCell<B>,
    pub main: ArrayCell<B>,
    pub shortcut: ArrayCell<B>,
    pub qkv: ArrayCell<B>,
    pub attention_output: ArrayCell<B>,
    pub mlp_fused_up: ArrayCell<B>,
    pub mlp_hidden: ArrayCell<B>,
    pub ssm_inproj: Option<ArrayCell<B>>,
    pub ssm_packed: Option<ArrayCell<B>>,
    pub ssm_conv_padded: Option<ArrayCell<B>>,
    pub short_conv_padded: Option<ArrayCell<B>>,
    pub ssm_x: Option<ArrayCell<B>>,
    pub ssm_b: Option<ArrayCell<B>>,
    pub ssm_c: Option<ArrayCell<B>>,
    pub ssm_dt: Option<ArrayCell<B>>,
    pub ssm_z: Option<ArrayCell<B>>,

    // 3-D
    pub rotated_queries: ArrayCell<B>,
    pub rotated_keys: ArrayCell<B>,
    pub extracted_values: ArrayCell<B>,

    // 2-pass attention intermediate buffers
    pub attention_partials: ArrayCell<B>, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: ArrayCell<B>, // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: ArrayCell<B>, // [num_heads * max_suffix_len * total_blocks_count]

    pub moe_topk_ids: Option<ArrayCell<B>>,
    pub moe_topk_probs: Option<ArrayCell<B>>,
    pub moe_offsets: Option<ArrayCell<B>>,
    pub moe_sumk: Option<ArrayCell<B>>,
    pub moe_bucketed_token_ids: Option<ArrayCell<B>>,
    pub moe_bucketed_probs: Option<ArrayCell<B>>,
    pub moe_x_perm: Option<ArrayCell<B>>,
    pub moe_tok2row: Option<ArrayCell<B>>,
    pub moe_y_partial: Option<ArrayCell<B>>,
    pub moe_hidden: Option<ArrayCell<B>>,
    pub moe_two_pass_row_expert_map: Option<ArrayCell<B>>,
    pub moe_tile_counts: Option<ArrayCell<B>>,
    pub moe_tile_offsets: Option<ArrayCell<B>>,
    pub moe_tile_map: Option<ArrayCell<B>>,
    pub moe_total_tiles: Option<ArrayCell<B>>,
    pub moe_dispatch_args: Option<ArrayCell<B>>,
    pub moe_scatter_partials: Option<ArrayCell<B>>,
    pub moe_scatter_block_bases: Option<ArrayCell<B>>,
    pub moe_block_alloc: Option<ArrayCell<B>>,
}

impl<B: Backend> ScratchBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        max_prefix_len: usize,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc =
            |shape: &[usize], dtype: DataType, label: &str| -> ArrayCell<B> {
                let array = context.create_array(
                    shape,
                    dtype,
                    &format!("scratch_buffers_{label}"),
                );
                RefCell::new(array)
            };

        let act_ty = model_shape.activation_data_type();

        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);
        let moe = match &decoder_config.layer_config.mlp_config {
            MLPConfig::MixtureOfExperts(moe) => Some(moe),
            _ => None,
        };
        let moe_max_routed =
            moe.map(|moe| max_suffix_len * moe.num_experts_per_token);
        let moe_scatter_entries = moe.map(|moe| {
            let num_blocks = ((max_suffix_len + 255) / 256).max(1);
            let num_tiles = ((moe.mixture_size + 512 - 1) / 512).max(1);
            num_blocks * num_tiles * 512
        });

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64, "token_ids"),
            token_positions: alloc(
                &[max_suffix_len],
                DataType::I32,
                "token_positions",
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
            attention_window_size_to_bias: model_shape
                .sliding_window_length_per_layer
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .into_iter()
                .map(|window_size| {
                    let label = match window_size {
                        Some(ws) => {
                            format!("attention_bias_for_window_size_{ws}")
                        },
                        None => {
                            "attention_bias_for_window_size_none".to_string()
                        },
                    };
                    let shape =
                        [max_suffix_len, max_suffix_len + max_prefix_len];
                    (window_size, alloc(&shape, act_ty, &label))
                })
                .collect(),
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
                &model_shape.attention_partials_shape(max_suffix_len),
                act_ty,
                "attention_partials",
            ),
            attention_sums: alloc(&sums_maxs_shape, act_ty, "attention_sums"),
            attention_maxs: alloc(&sums_maxs_shape, act_ty, "attention_maxs"),

            moe_topk_ids: moe.map(|moe| {
                alloc(
                    &model_shape.moe_topk_ids_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    ),
                    DataType::U32,
                    "moe_topk_ids",
                )
            }),
            moe_topk_probs: moe.map(|moe| {
                alloc(
                    &model_shape.moe_topk_probs_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    ),
                    act_ty,
                    "moe_topk_probs",
                )
            }),
            moe_offsets: moe.map(|moe| {
                alloc(
                    &model_shape.moe_offsets_shape(moe.mixture_size),
                    DataType::U32,
                    "moe_offsets",
                )
            }),
            moe_sumk: moe.map(|_| {
                alloc(&model_shape.moe_sumk_shape(), DataType::U32, "moe_sumk")
            }),
            moe_bucketed_token_ids: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_bucketed_token_ids_shape(max_routed),
                    DataType::U32,
                    "moe_bucketed_token_ids",
                )
            }),
            moe_bucketed_probs: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_bucketed_probs_shape(max_routed),
                    act_ty,
                    "moe_bucketed_probs",
                )
            }),
            moe_x_perm: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_x_perm_shape(max_routed),
                    DataType::F16,
                    "moe_x_perm",
                )
            }),
            moe_tok2row: moe.map(|moe| {
                alloc(
                    &model_shape.moe_tok2row_shape(
                        max_suffix_len,
                        moe.num_experts_per_token,
                    ),
                    DataType::I32,
                    "moe_tok2row",
                )
            }),
            moe_y_partial: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_y_partial_shape(max_routed),
                    DataType::F16,
                    "moe_y_partial",
                )
            }),
            moe_hidden: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_hidden_shape(max_routed),
                    DataType::F32,
                    "moe_hidden",
                )
            }),
            moe_two_pass_row_expert_map: moe_max_routed.map(|max_routed| {
                alloc(
                    &[max_routed],
                    DataType::U32,
                    "moe_two_pass_row_expert_map",
                )
            }),
            moe_tile_counts: moe.map(|moe| {
                alloc(
                    &model_shape.moe_counts_shape(moe.mixture_size),
                    DataType::U32,
                    "moe_tile_counts",
                )
            }),
            moe_tile_offsets: moe.map(|moe| {
                alloc(
                    &model_shape.moe_offsets_shape(moe.mixture_size),
                    DataType::U32,
                    "moe_tile_offsets",
                )
            }),
            moe_tile_map: moe_max_routed.map(|max_routed| {
                alloc(
                    &model_shape.moe_tile_map_shape(max_routed),
                    DataType::U32,
                    "moe_tile_map",
                )
            }),
            moe_total_tiles: moe.map(|_| {
                alloc(
                    &model_shape.moe_total_tiles_shape(),
                    DataType::U32,
                    "moe_total_tiles",
                )
            }),
            moe_dispatch_args: moe.map(|_| {
                alloc(
                    &model_shape.moe_dispatch_args_shape(),
                    DataType::U32,
                    "moe_dispatch_args",
                )
            }),
            moe_scatter_partials: moe_scatter_entries.map(|entries| {
                alloc(&[entries], DataType::U32, "moe_scatter_partials")
            }),
            moe_scatter_block_bases: moe_scatter_entries.map(|entries| {
                alloc(&[entries], DataType::U32, "moe_scatter_block_bases")
            }),
            moe_block_alloc: moe_scatter_entries.map(|entries| {
                alloc(&[entries], DataType::U32, "moe_block_alloc")
            }),
        }
    }
}
