use crate::{
    DecoderConfig,
    array::{ArrayCell, ArrayCellExt},
    backends::common::Backend,
    config::MLPConfig,
    forward_pass::{model_shape::ModelShape, scratch_buffers::ScratchBuffers},
};

pub struct LanguageModelGeneratorAuxBuffers<B: Backend> {
    pub ssm_inproj: Option<ArrayCell<B>>,
    pub ssm_packed: Option<ArrayCell<B>>,
    pub ssm_conv_padded: Option<ArrayCell<B>>,
    pub short_conv_padded: Option<ArrayCell<B>>,
    pub ssm_x: Option<ArrayCell<B>>,
    pub ssm_b: Option<ArrayCell<B>>,
    pub ssm_c: Option<ArrayCell<B>>,
    pub ssm_dt: Option<ArrayCell<B>>,
    pub ssm_z: Option<ArrayCell<B>>,
    // MoE buffers
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
        let max_routed =
            moe.map(|moe| suffix_length * moe.num_experts_per_token);
        let num_blocks = suffix_length.div_ceil(256).max(1);
        let scatter_entries = moe.map(|moe| {
            num_blocks * moe.mixture_size.div_ceil(512).max(1) * 512
        });
        let block_alloc_entries =
            moe.map(|moe| num_blocks * moe.mixture_size.div_ceil(512).max(1));

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
            short_conv_padded: scratch
                .short_conv_padded
                .as_ref()
                .zip(model_shape.short_conv_padded_shape(suffix_length))
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
            moe_topk_ids: moe.zip(scratch.moe_topk_ids.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_topk_ids_shape(
                        suffix_length,
                        moe.num_experts_per_token,
                    ))
                },
            ),
            moe_topk_probs: moe.zip(scratch.moe_topk_probs.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_topk_probs_shape(
                        suffix_length,
                        moe.num_experts_per_token,
                    ))
                },
            ),
            moe_offsets: moe.zip(scratch.moe_offsets.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_offsets_shape(moe.mixture_size))
                },
            ),
            moe_sumk: moe
                .and(scratch.moe_sumk.as_ref())
                .map(|buf| buf.view(&model_shape.moe_sumk_shape())),
            moe_bucketed_token_ids: max_routed
                .zip(scratch.moe_bucketed_token_ids.as_ref())
                .map(|(max_routed, buf)| {
                    buf.view(
                        &model_shape.moe_bucketed_token_ids_shape(max_routed),
                    )
                }),
            moe_bucketed_probs: max_routed
                .zip(scratch.moe_bucketed_probs.as_ref())
                .map(|(max_routed, buf)| {
                    buf.view(&model_shape.moe_bucketed_probs_shape(max_routed))
                }),
            moe_x_perm: max_routed.zip(scratch.moe_x_perm.as_ref()).map(
                |(max_routed, buf)| {
                    buf.view(&model_shape.moe_x_perm_shape(max_routed))
                },
            ),
            moe_tok2row: moe.zip(scratch.moe_tok2row.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_tok2row_shape(
                        suffix_length,
                        moe.num_experts_per_token,
                    ))
                },
            ),
            moe_y_partial: max_routed.zip(scratch.moe_y_partial.as_ref()).map(
                |(max_routed, buf)| {
                    buf.view(&model_shape.moe_y_partial_shape(max_routed))
                },
            ),
            moe_hidden: max_routed.zip(scratch.moe_hidden.as_ref()).map(
                |(max_routed, buf)| {
                    buf.view(&model_shape.moe_hidden_shape(max_routed))
                },
            ),
            moe_two_pass_row_expert_map: max_routed
                .zip(scratch.moe_two_pass_row_expert_map.as_ref())
                .map(|(max_routed, buf)| buf.view(&[max_routed])),
            moe_tile_counts: moe.zip(scratch.moe_tile_counts.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_counts_shape(moe.mixture_size))
                },
            ),
            moe_tile_offsets: moe.zip(scratch.moe_tile_offsets.as_ref()).map(
                |(moe, buf)| {
                    buf.view(&model_shape.moe_offsets_shape(moe.mixture_size))
                },
            ),
            moe_tile_map: max_routed.zip(scratch.moe_tile_map.as_ref()).map(
                |(max_routed, buf)| {
                    buf.view(&model_shape.moe_tile_map_shape(max_routed))
                },
            ),
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
