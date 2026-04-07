use crate::{
    array::Array,
    backends::common::Backend,
    config::{DecoderConfig, MLPConfig},
    forward_pass::{model_shape::ModelShape, scratch_buffers::ScratchBuffers},
};

pub struct LanguageModelGeneratorAuxBuffers<B: Backend> {
    pub ssm_inproj: Option<Array<B>>,
    pub ssm_packed: Option<Array<B>>,
    pub ssm_conv_padded: Option<Array<B>>,
    pub ssm_x: Option<Array<B>>,
    pub ssm_b: Option<Array<B>>,
    pub ssm_c: Option<Array<B>>,
    pub ssm_dt: Option<Array<B>>,
    pub ssm_z: Option<Array<B>>,
    // DeltaNet prep buffers
    pub delta_net_prep_q_norm: Option<Array<B>>,
    pub delta_net_prep_k_norm: Option<Array<B>>,
    pub delta_net_prep_beta: Option<Array<B>>,
    pub delta_net_prep_decay: Option<Array<B>>,
    // PLE buffers
    pub ple_embeddings: Option<Array<B>>,
    pub ple_projection: Option<Array<B>>,
    pub ple_combined: Option<Array<B>>,
    pub ple_gate: Option<Array<B>>,
    // MoE buffers
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

impl<B: Backend> LanguageModelGeneratorAuxBuffers<B> {
    pub fn new(
        scratch: &ScratchBuffers<B>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let moe = match &decoder_config.layer_config.mlp_config {
            MLPConfig::MixtureOfExperts(moe) => Some(moe),
            _ => None,
        };
        let max_routed = moe.map(|moe| suffix_length * moe.num_active_routed_experts);
        let num_blocks = suffix_length.div_ceil(256).max(1);
        let scatter_entries = moe.map(|moe| num_blocks * moe.num_routed_experts.div_ceil(512).max(1) * 512);
        let block_alloc_entries = moe.map(|moe| num_blocks * moe.num_routed_experts.div_ceil(512).max(1));

        Self {
            ssm_inproj: scratch
                .ssm_inproj
                .as_ref()
                .zip(model_shape.ssm_inproj_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_packed: scratch
                .ssm_packed
                .as_ref()
                .zip(model_shape.ssm_packed_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_conv_padded: scratch
                .ssm_conv_padded
                .as_ref()
                .zip(model_shape.ssm_conv_padded_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_x: scratch
                .ssm_x
                .as_ref()
                .zip(model_shape.ssm_x_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_b: scratch
                .ssm_b
                .as_ref()
                .zip(model_shape.ssm_bc_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_c: scratch
                .ssm_c
                .as_ref()
                .zip(model_shape.ssm_bc_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_dt: scratch
                .ssm_dt
                .as_ref()
                .zip(model_shape.ssm_dt_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ssm_z: scratch
                .ssm_z
                .as_ref()
                .zip(model_shape.ssm_z_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            delta_net_prep_q_norm: scratch
                .delta_net_prep_q_norm
                .as_ref()
                .zip(model_shape.delta_net_prep_qk_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            delta_net_prep_k_norm: scratch
                .delta_net_prep_k_norm
                .as_ref()
                .zip(model_shape.delta_net_prep_qk_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            delta_net_prep_beta: scratch
                .delta_net_prep_beta
                .as_ref()
                .zip(model_shape.delta_net_prep_beta_decay_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            delta_net_prep_decay: scratch
                .delta_net_prep_decay
                .as_ref()
                .zip(model_shape.delta_net_prep_beta_decay_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ple_embeddings: scratch
                .ple_embeddings
                .as_ref()
                .zip(model_shape.ple_embeddings_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ple_projection: scratch
                .ple_projection
                .as_ref()
                .zip(model_shape.ple_projection_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ple_combined: scratch
                .ple_combined
                .as_ref()
                .zip(model_shape.ple_combined_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            ple_gate: scratch
                .ple_gate
                .as_ref()
                .zip(model_shape.ple_gate_shape(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            moe_topk_ids: moe.zip(scratch.moe_topk_ids.as_ref()).map(|(moe, buf)| {
                buf.view(&model_shape.moe_topk_ids_shape(suffix_length, moe.num_active_routed_experts))
            }),
            moe_topk_probs: moe.zip(scratch.moe_topk_probs.as_ref()).map(|(moe, buf)| {
                buf.view(&model_shape.moe_topk_probs_shape(suffix_length, moe.num_active_routed_experts))
            }),
            moe_offsets: moe
                .zip(scratch.moe_offsets.as_ref())
                .map(|(moe, buf)| buf.view(&model_shape.moe_offsets_shape(moe.num_routed_experts))),
            moe_sumk: moe.and(scratch.moe_sumk.as_ref()).map(|buf| buf.view(&model_shape.moe_sumk_shape())),
            moe_bucketed_token_ids: max_routed
                .zip(scratch.moe_bucketed_token_ids.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_bucketed_token_ids_shape(max_routed))),
            moe_bucketed_probs: max_routed
                .zip(scratch.moe_bucketed_probs.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_bucketed_probs_shape(max_routed))),
            moe_x_perm: max_routed
                .zip(scratch.moe_x_perm.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_x_perm_shape(max_routed))),
            moe_tok2row: moe.zip(scratch.moe_tok2row.as_ref()).map(|(moe, buf)| {
                buf.view(&model_shape.moe_tok2row_shape(suffix_length, moe.num_active_routed_experts))
            }),
            moe_y_partial: max_routed
                .zip(scratch.moe_y_partial.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_y_partial_shape(max_routed))),
            moe_hidden: max_routed
                .zip(scratch.moe_hidden.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_hidden_shape(max_routed))),
            moe_two_pass_row_expert_map: max_routed
                .zip(scratch.moe_two_pass_row_expert_map.as_ref())
                .map(|(max_routed, buf)| buf.view(&[max_routed])),
            moe_tile_counts: moe
                .zip(scratch.moe_tile_counts.as_ref())
                .map(|(moe, buf)| buf.view(&model_shape.moe_counts_shape(moe.num_routed_experts))),
            moe_tile_offsets: moe
                .zip(scratch.moe_tile_offsets.as_ref())
                .map(|(moe, buf)| buf.view(&model_shape.moe_offsets_shape(moe.num_routed_experts))),
            moe_tile_map: max_routed
                .zip(scratch.moe_tile_map.as_ref())
                .map(|(max_routed, buf)| buf.view(&model_shape.moe_tile_map_shape(max_routed))),
            moe_total_tiles: moe
                .and(scratch.moe_total_tiles.as_ref())
                .map(|buf| buf.view(&model_shape.moe_total_tiles_shape())),
            moe_dispatch_args: moe
                .and(scratch.moe_dispatch_args.as_ref())
                .map(|buf| buf.view(&model_shape.moe_dispatch_args_shape())),
            moe_scatter_partials: scatter_entries
                .zip(scratch.moe_scatter_partials.as_ref())
                .map(|(entries, buf)| buf.view(&[entries])),
            moe_scatter_block_bases: scatter_entries
                .zip(scratch.moe_scatter_block_bases.as_ref())
                .map(|(entries, buf)| buf.view(&[entries])),
            moe_block_alloc: block_alloc_entries
                .zip(scratch.moe_block_alloc.as_ref())
                .map(|(entries, buf)| buf.view(&[entries])),
        }
    }
}
