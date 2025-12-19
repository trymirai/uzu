use std::cell::RefCell;

use super::super::{ModelShape, ScratchBuffers};
use crate::{Array, DecoderConfig, DeviceContext, config::MLPConfig};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub struct LanguageModelGeneratorAuxBuffers<C: DeviceContext> {
    pub ssm_inproj: Option<ArrayCell<C>>,
    pub ssm_packed: Option<ArrayCell<C>>,
    pub ssm_conv_padded: Option<ArrayCell<C>>,
    pub ssm_x: Option<ArrayCell<C>>,
    pub ssm_b: Option<ArrayCell<C>>,
    pub ssm_c: Option<ArrayCell<C>>,
    pub ssm_dt: Option<ArrayCell<C>>,
    pub ssm_z: Option<ArrayCell<C>>,
    // MoE buffers
    pub moe_topk_ids: Option<ArrayCell<C>>,
    pub moe_topk_probs: Option<ArrayCell<C>>,
    pub moe_offsets: Option<ArrayCell<C>>,
    pub moe_sumk: Option<ArrayCell<C>>,
    pub moe_bucketed_token_ids: Option<ArrayCell<C>>,
    pub moe_bucketed_probs: Option<ArrayCell<C>>,
    pub moe_x_perm: Option<ArrayCell<C>>,
    pub moe_tok2row: Option<ArrayCell<C>>,
    pub moe_y_partial: Option<ArrayCell<C>>,
    pub moe_hidden: Option<ArrayCell<C>>,
    pub moe_two_pass_row_expert_map: Option<ArrayCell<C>>,
    pub moe_tile_counts: Option<ArrayCell<C>>,
    pub moe_tile_offsets: Option<ArrayCell<C>>,
    pub moe_tile_map: Option<ArrayCell<C>>,
    pub moe_total_tiles: Option<ArrayCell<C>>,
    pub moe_dispatch_args: Option<ArrayCell<C>>,
    pub moe_scatter_partials: Option<ArrayCell<C>>,
    pub moe_scatter_block_bases: Option<ArrayCell<C>>,
    pub moe_block_alloc: Option<ArrayCell<C>>,
}

impl<C: DeviceContext> LanguageModelGeneratorAuxBuffers<C> {
    pub fn new(
        scratch: &ScratchBuffers<C>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        Self {
            ssm_inproj: match (
                scratch.ssm_inproj.as_ref(),
                model_shape.ssm_inproj_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_packed: match (
                scratch.ssm_packed.as_ref(),
                model_shape.ssm_packed_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_conv_padded: match (
                scratch.ssm_conv_padded.as_ref(),
                model_shape.ssm_conv_padded_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_x: match (
                scratch.ssm_x.as_ref(),
                model_shape.ssm_x_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_b: match (
                scratch.ssm_b.as_ref(),
                model_shape.ssm_bc_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_c: match (
                scratch.ssm_c.as_ref(),
                model_shape.ssm_bc_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_dt: match (
                scratch.ssm_dt.as_ref(),
                model_shape.ssm_dt_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            ssm_z: match (
                scratch.ssm_z.as_ref(),
                model_shape.ssm_z_shape(suffix_length),
            ) {
                (Some(buf), Some(shape)) => {
                    Some(RefCell::new(buf.reshape(&shape)))
                },
                _ => None,
            },
            moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_topk_ids.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_topk_ids_shape(
                                suffix_length,
                                moe.num_experts_per_token,
                            ),
                        ))
                    })
                },
                _ => None,
            },
            moe_topk_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_topk_probs.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_topk_probs_shape(
                                suffix_length,
                                moe.num_experts_per_token,
                            ),
                        ))
                    })
                },
                _ => None,
            },
            moe_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_offsets.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_offsets_shape(moe.mixture_size),
                        ))
                    })
                },
                _ => None,
            },
            moe_sumk: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    scratch.moe_sumk.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(&model_shape.moe_sumk_shape()))
                    })
                },
                _ => None,
            },
            moe_bucketed_token_ids: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_bucketed_token_ids.as_ref().map(|buf| {
                        RefCell::new(
                            buf.reshape(
                                &model_shape
                                    .moe_bucketed_token_ids_shape(max_routed),
                            ),
                        )
                    })
                },
                _ => None,
            },
            moe_bucketed_probs: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_bucketed_probs.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_bucketed_probs_shape(max_routed),
                        ))
                    })
                },
                _ => None,
            },
            moe_x_perm: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_x_perm.as_ref().map(|buf| {
                        RefCell::new(
                            buf.reshape(
                                &model_shape.moe_x_perm_shape(max_routed),
                            ),
                        )
                    })
                },
                _ => None,
            },
            moe_tok2row: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_tok2row.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_tok2row_shape(
                                suffix_length,
                                moe.num_experts_per_token,
                            ),
                        ))
                    })
                },
                _ => None,
            },
            moe_y_partial: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_y_partial.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_y_partial_shape(max_routed),
                        ))
                    })
                },
                _ => None,
            },
            moe_hidden: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_hidden.as_ref().map(|buf| {
                        RefCell::new(
                            buf.reshape(
                                &model_shape.moe_hidden_shape(max_routed),
                            ),
                        )
                    })
                },
                _ => None,
            },
            moe_two_pass_row_expert_map: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch
                        .moe_two_pass_row_expert_map
                        .as_ref()
                        .map(|buf| RefCell::new(buf.reshape(&[max_routed])))
                },
                _ => None,
            },
            moe_tile_counts: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_tile_counts.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_counts_shape(moe.mixture_size),
                        ))
                    })
                },
                _ => None,
            },
            moe_tile_offsets: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_tile_offsets.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_offsets_shape(moe.mixture_size),
                        ))
                    })
                },
                _ => None,
            },
            moe_tile_map: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    let max_routed = suffix_length * moe.num_experts_per_token;
                    scratch.moe_tile_map.as_ref().map(|buf| {
                        RefCell::new(buf.reshape(
                            &model_shape.moe_tile_map_shape(max_routed),
                        ))
                    })
                },
                _ => None,
            },
            moe_total_tiles: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    scratch.moe_total_tiles.as_ref().map(|buf| {
                        RefCell::new(
                            buf.reshape(&model_shape.moe_total_tiles_shape()),
                        )
                    })
                },
                _ => None,
            },
            moe_dispatch_args: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(_) => {
                    scratch.moe_dispatch_args.as_ref().map(|buf| {
                        RefCell::new(
                            buf.reshape(&model_shape.moe_dispatch_args_shape()),
                        )
                    })
                },
                _ => None,
            },
            moe_scatter_partials: match &decoder_config.layer_config.mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_scatter_partials.as_ref().map(|buf| {
                        let num_blocks = ((suffix_length + 255) / 256).max(1);
                        let num_tiles =
                            ((moe.mixture_size + 512 - 1) / 512).max(1);
                        let entries = num_blocks * num_tiles * 512;
                        RefCell::new(buf.reshape(&[entries]))
                    })
                },
                _ => None,
            },
            moe_scatter_block_bases: match &decoder_config
                .layer_config
                .mlp_config
            {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_scatter_block_bases.as_ref().map(|buf| {
                        let num_blocks = ((suffix_length + 255) / 256).max(1);
                        let num_tiles =
                            ((moe.mixture_size + 512 - 1) / 512).max(1);
                        let entries = num_blocks * num_tiles * 512;
                        RefCell::new(buf.reshape(&[entries]))
                    })
                },
                _ => None,
            },
            moe_block_alloc: match &decoder_config.layer_config.mlp_config {
                MLPConfig::MixtureOfExperts(moe) => {
                    scratch.moe_block_alloc.as_ref().map(|buf| {
                        let num_blocks = ((suffix_length + 255) / 256).max(1);
                        let num_tiles =
                            ((moe.mixture_size + 512 - 1) / 512).max(1);
                        let entries = num_blocks * num_tiles;
                        RefCell::new(buf.reshape(&[entries]))
                    })
                },
                _ => None,
            },
        }
    }
}
