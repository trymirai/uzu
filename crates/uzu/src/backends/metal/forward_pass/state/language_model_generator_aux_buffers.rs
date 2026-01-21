use std::{cell::RefCell, rc::Rc};

use super::super::{ModelShape, ScratchBuffers};
use crate::{
    Array, DataType, DecoderConfig,
    backends::metal::{MTLContext, MetalArray},
    config::MLPConfig,
};

type ArrayCell = RefCell<MetalArray>;

pub struct LanguageModelGeneratorAuxBuffers {
    pub ssm_inproj: Option<ArrayCell>,
    pub ssm_packed: Option<ArrayCell>,
    pub ssm_conv_padded: Option<ArrayCell>,
    pub short_conv_padded: Option<ArrayCell>,
    pub ssm_x: Option<ArrayCell>,
    pub ssm_b: Option<ArrayCell>,
    pub ssm_c: Option<ArrayCell>,
    pub ssm_dt: Option<ArrayCell>,
    pub ssm_z: Option<ArrayCell>,
    // MoE buffers
    pub moe_topk_ids: Option<ArrayCell>,
    pub moe_topk_probs: Option<ArrayCell>,
    pub moe_offsets: Option<ArrayCell>,
    pub moe_sumk: Option<ArrayCell>,
    pub moe_bucketed_token_ids: Option<ArrayCell>,
    pub moe_bucketed_probs: Option<ArrayCell>,
    pub moe_x_perm: Option<ArrayCell>,
    pub moe_tok2row: Option<ArrayCell>,
    pub moe_y_partial: Option<ArrayCell>,
    pub moe_hidden: Option<ArrayCell>,
    pub moe_two_pass_row_expert_map: Option<ArrayCell>,
    pub moe_tile_counts: Option<ArrayCell>,
    pub moe_tile_offsets: Option<ArrayCell>,
    pub moe_tile_map: Option<ArrayCell>,
    pub moe_total_tiles: Option<ArrayCell>,
    pub moe_dispatch_args: Option<ArrayCell>,
    pub moe_scatter_partials: Option<ArrayCell>,
    pub moe_scatter_block_bases: Option<ArrayCell>,
    pub moe_block_alloc: Option<ArrayCell>,
}

impl LanguageModelGeneratorAuxBuffers {
    pub fn new(
        scratch: &ScratchBuffers<Rc<MTLContext>>,
        decoder_config: &DecoderConfig,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let act_dtype = model_shape.activation_data_type();

        unsafe {
            Self {
                ssm_inproj: match (
                    scratch.ssm_inproj.as_ref(),
                    model_shape.ssm_inproj_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_packed: match (
                    scratch.ssm_packed.as_ref(),
                    model_shape.ssm_packed_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_conv_padded: match (
                    scratch.ssm_conv_padded.as_ref(),
                    model_shape.ssm_conv_padded_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                short_conv_padded: match (
                    scratch.short_conv_padded.as_ref(),
                    model_shape.short_conv_padded_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_x: match (
                    scratch.ssm_x.as_ref(),
                    model_shape.ssm_x_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_b: match (
                    scratch.ssm_b.as_ref(),
                    model_shape.ssm_bc_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_c: match (
                    scratch.ssm_c.as_ref(),
                    model_shape.ssm_bc_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_dt: match (
                    scratch.ssm_dt.as_ref(),
                    model_shape.ssm_dt_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                ssm_z: match (
                    scratch.ssm_z.as_ref(),
                    model_shape.ssm_z_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => {
                        Some(RefCell::new(MetalArray::new(
                            buf.borrow().mtl_buffer_cloned(),
                            &shape,
                            act_dtype,
                        )))
                    },
                    _ => None,
                },
                moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_topk_ids.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_topk_ids_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_topk_probs: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_topk_probs.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_topk_probs_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                act_dtype,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_offsets: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_offsets.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape
                                    .moe_offsets_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_sumk: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_sumk.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_sumk_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_bucketed_token_ids: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_bucketed_token_ids.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape
                                    .moe_bucketed_token_ids_shape(max_routed),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_bucketed_probs: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_bucketed_probs.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape
                                    .moe_bucketed_probs_shape(max_routed),
                                act_dtype,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_x_perm: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_x_perm.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_x_perm_shape(max_routed),
                                DataType::F16,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tok2row: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tok2row.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_tok2row_shape(
                                    suffix_length,
                                    moe.num_experts_per_token,
                                ),
                                DataType::I32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_y_partial: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_y_partial.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_y_partial_shape(max_routed),
                                DataType::F16,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_hidden: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_hidden.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_hidden_shape(max_routed),
                                DataType::F32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_two_pass_row_expert_map: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_two_pass_row_expert_map.as_ref().map(
                            |buf| {
                                RefCell::new(MetalArray::new(
                                    buf.borrow().mtl_buffer_cloned(),
                                    &[max_routed],
                                    DataType::U32,
                                ))
                            },
                        )
                    },
                    _ => None,
                },
                moe_tile_counts: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tile_counts.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_counts_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tile_offsets: match &decoder_config.layer_config.mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_tile_offsets.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape
                                    .moe_offsets_shape(moe.mixture_size),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_tile_map: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        let max_routed =
                            suffix_length * moe.num_experts_per_token;
                        scratch.moe_tile_map.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_tile_map_shape(max_routed),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_total_tiles: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_total_tiles.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_total_tiles_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_dispatch_args: match &decoder_config.layer_config.mlp_config
                {
                    MLPConfig::MixtureOfExperts(_) => {
                        scratch.moe_dispatch_args.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &model_shape.moe_dispatch_args_shape(),
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_scatter_partials: match &decoder_config
                    .layer_config
                    .mlp_config
                {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_scatter_partials.as_ref().map(|buf| {
                            let num_blocks =
                                ((suffix_length + 255) / 256).max(1);
                            let num_tiles =
                                ((moe.mixture_size + 512 - 1) / 512).max(1);
                            let entries = num_blocks * num_tiles * 512;
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &[entries],
                                DataType::U32,
                            ))
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
                            let num_blocks =
                                ((suffix_length + 255) / 256).max(1);
                            let num_tiles =
                                ((moe.mixture_size + 512 - 1) / 512).max(1);
                            let entries = num_blocks * num_tiles * 512;
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &[entries],
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
                moe_block_alloc: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_block_alloc.as_ref().map(|buf| {
                            let num_blocks =
                                ((suffix_length + 255) / 256).max(1);
                            let num_tiles =
                                ((moe.mixture_size + 512 - 1) / 512).max(1);
                            let entries = num_blocks * num_tiles;
                            RefCell::new(MetalArray::new(
                                buf.borrow().mtl_buffer_cloned(),
                                &[entries],
                                DataType::U32,
                            ))
                        })
                    },
                    _ => None,
                },
            }
        }
    }
}
