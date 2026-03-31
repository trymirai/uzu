use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    config::{DecoderConfig, MLPConfig},
    forward_pass::model_shape::ModelShape,
};

pub struct ScratchBuffers<B: Backend> {
    // 1-D
    pub token_ids: Array<B>,
    pub token_subtrie_ranges: Array<B>,
    pub token_positions: Array<B>,
    pub token_parents: Array<B>,
    pub token_bitmask: Array<B>,
    pub token_seeds: Array<B>,
    pub sampling_output: Array<B>,

    // 2-D
    pub logits: Array<B>,
    pub main: Array<B>,
    pub shortcut: Array<B>,
    pub qkv: Array<B>,
    pub gate: Option<Array<B>>,
    pub attention_output: Array<B>,
    pub mlp_fused_up: Array<B>,
    pub mlp_hidden: Array<B>,
    pub lora_intermediate: Option<Array<B>>,
    pub ssm_inproj: Option<Array<B>>,
    pub ssm_packed: Option<Array<B>>,
    pub ssm_conv_padded: Option<Array<B>>,
    pub ssm_x: Option<Array<B>>,
    pub ssm_b: Option<Array<B>>,
    pub ssm_c: Option<Array<B>>,
    pub ssm_dt: Option<Array<B>>,
    pub ssm_z: Option<Array<B>>,

    // 3-D
    pub rotated_queries: Array<B>,
    pub rotated_keys: Array<B>,
    pub extracted_values: Array<B>,

    // 2-pass attention intermediate buffers
    pub attention_partials: Array<B>, // [num_heads * max_suffix_len * total_blocks_count * head_dim]
    pub attention_sums: Array<B>,     // [num_heads * max_suffix_len * total_blocks_count]
    pub attention_maxs: Array<B>,     // [num_heads * max_suffix_len * total_blocks_count]

    pub moe_topk_ids: Option<Array<B>>,
    pub moe_topk_probs: Option<Array<B>>,
    pub moe_offsets: Option<Array<B>>,
    pub moe_sumk: Option<Array<B>>,
    pub moe_bucketed_token_ids: Option<Array<B>>,
    pub moe_bucketed_probs: Option<Array<B>>,
    pub moe_x_perm: Option<Array<B>>,
    pub moe_tok2row: Option<Array<B>>,
    pub moe_y_partial: Option<Array<B>>,
    pub moe_hidden: Option<Array<B>>,
    pub moe_two_pass_row_expert_map: Option<Array<B>>,
    pub moe_tile_counts: Option<Array<B>>,
    pub moe_tile_offsets: Option<Array<B>>,
    pub moe_tile_map: Option<Array<B>>,
    pub moe_total_tiles: Option<Array<B>>,
    pub moe_dispatch_args: Option<Array<B>>,
    pub moe_scatter_partials: Option<Array<B>>,
    pub moe_scatter_block_bases: Option<Array<B>>,
    pub moe_block_alloc: Option<Array<B>>,
}

impl<B: Backend> ScratchBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        max_suffix_len: usize,
    ) -> Self {
        // Helper closure for allocation
        let alloc = |shape: &[usize], dtype: DataType, label: &str| -> Array<B> {
            context.create_array_uninitialized(shape, dtype, &format!("scratch_buffers_{label}"))
        };

        let act_ty = model_shape.activation_data_type();

        let sums_maxs_shape = model_shape.attention_sums_shape(max_suffix_len);
        let moe = match &decoder_config.layer_config.mlp_config {
            MLPConfig::MixtureOfExperts(moe) => Some(moe),
            _ => None,
        };
        let moe_max_routed = moe.map(|moe| max_suffix_len * moe.num_active_routed_experts);
        let moe_scatter_entries = moe.map(|moe| {
            let num_blocks = ((max_suffix_len + 255) / 256).max(1);
            let num_tiles = ((moe.num_routed_experts + 512 - 1) / 512).max(1);
            num_blocks * num_tiles * 512
        });

        Self {
            // 1-D
            token_ids: alloc(&[max_suffix_len], DataType::U64, "token_ids"),
            token_subtrie_ranges: alloc(
                &model_shape.subtrie_ranges_shape(max_suffix_len),
                DataType::U32,
                "token_subtrie_ranges",
            ),
            token_positions: alloc(&[max_suffix_len], DataType::I32, "token_positions"),
            token_parents: alloc(&[max_suffix_len], DataType::I32, "token_parents"),
            token_bitmask: alloc(&model_shape.bitmask_shape(max_suffix_len), DataType::U32, "token_bitmask"),
            token_seeds: alloc(&[max_suffix_len], DataType::U64, "token_seeds"),
            sampling_output: alloc(&[max_suffix_len], DataType::U32, "sampling_output"),

            // 2-D
            logits: alloc(&model_shape.logits_shape(max_suffix_len), act_ty, "logits"),
            main: alloc(&model_shape.main_shape(max_suffix_len), act_ty, "main"),
            shortcut: alloc(&model_shape.main_shape(max_suffix_len), act_ty, "shortcut"),
            qkv: alloc(&model_shape.qkv_shape(max_suffix_len), act_ty, "qkv"),
            gate: model_shape.gate_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "gate")),
            attention_output: alloc(&model_shape.attention_output_shape(max_suffix_len), act_ty, "attention_output"),
            mlp_fused_up: alloc(&model_shape.mlp_fused_up_shape(max_suffix_len), act_ty, "mlp_fused_up"),
            mlp_hidden: alloc(&model_shape.mlp_hidden_shape(max_suffix_len), act_ty, "mlp_hidden"),
            lora_intermediate: model_shape
                .lora_intermediate(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "lora_intermediate")),
            ssm_inproj: model_shape.ssm_inproj_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_inproj")),
            ssm_packed: model_shape.ssm_packed_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_packed")),
            ssm_conv_padded: model_shape
                .ssm_conv_padded_shape(max_suffix_len)
                .map(|shape| alloc(&shape, act_ty, "ssm_conv_padded")),
            ssm_x: model_shape.ssm_x_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_x")),
            ssm_b: model_shape.ssm_bc_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_b")),
            ssm_c: model_shape.ssm_bc_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_c")),
            ssm_dt: model_shape.ssm_dt_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_dt")),
            ssm_z: model_shape.ssm_z_shape(max_suffix_len).map(|shape| alloc(&shape, act_ty, "ssm_z")),
            // 3-D
            rotated_queries: alloc(&model_shape.rotated_queries_shape(max_suffix_len), act_ty, "rotated_queries"),
            rotated_keys: alloc(&model_shape.rotated_keys_shape(max_suffix_len), act_ty, "rotated_keys"),
            extracted_values: alloc(&model_shape.extracted_values_shape(max_suffix_len), act_ty, "extracted_values"),

            attention_partials: alloc(
                &model_shape.attention_partials_shape(max_suffix_len),
                act_ty,
                "attention_partials",
            ),
            attention_sums: alloc(&sums_maxs_shape, act_ty, "attention_sums"),
            attention_maxs: alloc(&sums_maxs_shape, act_ty, "attention_maxs"),

            moe_topk_ids: moe.map(|moe| {
                alloc(
                    &model_shape.moe_topk_ids_shape(max_suffix_len, moe.num_active_routed_experts),
                    DataType::U32,
                    "moe_topk_ids",
                )
            }),
            moe_topk_probs: moe.map(|moe| {
                alloc(
                    &model_shape.moe_topk_probs_shape(max_suffix_len, moe.num_active_routed_experts),
                    act_ty,
                    "moe_topk_probs",
                )
            }),
            moe_offsets: moe
                .map(|moe| alloc(&model_shape.moe_offsets_shape(moe.num_routed_experts), DataType::U32, "moe_offsets")),
            moe_sumk: moe.map(|_| alloc(&model_shape.moe_sumk_shape(), DataType::U32, "moe_sumk")),
            moe_bucketed_token_ids: moe_max_routed.map(|max_routed| {
                alloc(&model_shape.moe_bucketed_token_ids_shape(max_routed), DataType::U32, "moe_bucketed_token_ids")
            }),
            moe_bucketed_probs: moe_max_routed.map(|max_routed| {
                alloc(&model_shape.moe_bucketed_probs_shape(max_routed), act_ty, "moe_bucketed_probs")
            }),
            moe_x_perm: moe_max_routed
                .map(|max_routed| alloc(&model_shape.moe_x_perm_shape(max_routed), DataType::F16, "moe_x_perm")),
            moe_tok2row: moe.map(|moe| {
                alloc(
                    &model_shape.moe_tok2row_shape(max_suffix_len, moe.num_active_routed_experts),
                    DataType::I32,
                    "moe_tok2row",
                )
            }),
            moe_y_partial: moe_max_routed
                .map(|max_routed| alloc(&model_shape.moe_y_partial_shape(max_routed), DataType::F16, "moe_y_partial")),
            moe_hidden: moe_max_routed
                .map(|max_routed| alloc(&model_shape.moe_hidden_shape(max_routed), DataType::F32, "moe_hidden")),
            moe_two_pass_row_expert_map: moe_max_routed
                .map(|max_routed| alloc(&[max_routed], DataType::U32, "moe_two_pass_row_expert_map")),
            moe_tile_counts: moe.map(|moe| {
                alloc(&model_shape.moe_counts_shape(moe.num_routed_experts), DataType::U32, "moe_tile_counts")
            }),
            moe_tile_offsets: moe.map(|moe| {
                alloc(&model_shape.moe_offsets_shape(moe.num_routed_experts), DataType::U32, "moe_tile_offsets")
            }),
            moe_tile_map: moe_max_routed
                .map(|max_routed| alloc(&model_shape.moe_tile_map_shape(max_routed), DataType::U32, "moe_tile_map")),
            moe_total_tiles: moe.map(|_| alloc(&model_shape.moe_total_tiles_shape(), DataType::U32, "moe_total_tiles")),
            moe_dispatch_args: moe
                .map(|_| alloc(&model_shape.moe_dispatch_args_shape(), DataType::U32, "moe_dispatch_args")),
            moe_scatter_partials: moe_scatter_entries
                .map(|entries| alloc(&[entries], DataType::U32, "moe_scatter_partials")),
            moe_scatter_block_bases: moe_scatter_entries
                .map(|entries| alloc(&[entries], DataType::U32, "moe_scatter_block_bases")),
            moe_block_alloc: moe_scatter_entries.map(|entries| alloc(&[entries], DataType::U32, "moe_block_alloc")),
        }
    }
}
