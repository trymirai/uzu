//! LLM-specific auxiliary buffers (SSM, MoE).

use std::cell::RefCell;

use super::super::{ModelShape, ScratchBuffers};
use crate::{
    DataType, DecoderConfig, backends::metal::MetalArray, config::MLPConfig,
};

type ArrayCell = RefCell<MetalArray>;

/// LLM-specific auxiliary buffers (SSM, MoE).
pub struct LLMAuxBuffers {
    pub ssm_inproj: Option<ArrayCell>,
    pub ssm_packed: Option<ArrayCell>,
    pub ssm_conv_padded: Option<ArrayCell>,
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

impl LLMAuxBuffers {
    pub fn new(
        scratch: &ScratchBuffers,
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
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_packed: match (
                    scratch.ssm_packed.as_ref(),
                    model_shape.ssm_packed_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_conv_padded: match (
                    scratch.ssm_conv_padded.as_ref(),
                    model_shape.ssm_conv_padded_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_x: match (
                    scratch.ssm_x.as_ref(),
                    model_shape.ssm_x_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_b: match (
                    scratch.ssm_b.as_ref(),
                    model_shape.ssm_bc_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_c: match (
                    scratch.ssm_c.as_ref(),
                    model_shape.ssm_bc_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_dt: match (
                    scratch.ssm_dt.as_ref(),
                    model_shape.ssm_dt_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                ssm_z: match (
                    scratch.ssm_z.as_ref(),
                    model_shape.ssm_z_shape(suffix_length),
                ) {
                    (Some(buf), Some(shape)) => Some(RefCell::new(
                        MetalArray::new(buf.clone(), &shape, act_dtype),
                    )),
                    _ => None,
                },
                // MoE buffers - simplified initialization
                moe_topk_ids: match &decoder_config.layer_config.mlp_config {
                    MLPConfig::MixtureOfExperts(moe) => {
                        scratch.moe_topk_ids.as_ref().map(|buf| {
                            RefCell::new(MetalArray::new(
                                buf.clone(),
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
                                buf.clone(),
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
                // Remaining MoE buffers (simplified - can expand as needed)
                moe_offsets: None,
                moe_sumk: None,
                moe_bucketed_token_ids: None,
                moe_bucketed_probs: None,
                moe_x_perm: None,
                moe_tok2row: None,
                moe_y_partial: None,
                moe_hidden: None,
                moe_two_pass_row_expert_map: None,
                moe_tile_counts: None,
                moe_tile_offsets: None,
                moe_tile_map: None,
                moe_total_tiles: None,
                moe_dispatch_args: None,
                moe_scatter_partials: None,
                moe_scatter_block_bases: None,
                moe_block_alloc: None,
            }
        }
    }
}
