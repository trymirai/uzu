//! Common auxiliary buffers shared by both LLM and classifier modes.

use std::cell::RefCell;

use super::super::{ModelShape, ScratchBuffers};
use crate::backends::metal::MetalArray;

type ArrayCell = RefCell<MetalArray>;

/// Common auxiliary buffers shared by both LLM and classifier modes.
pub struct CommonAuxBuffers {
    pub suffix_length: usize,
    pub main: ArrayCell,
    pub shortcut: ArrayCell,
    pub qkv: ArrayCell,
    pub attention_output: ArrayCell,
    pub mlp_fused_up: ArrayCell,
    pub mlp_hidden: ArrayCell,
    pub rotated_queries: ArrayCell,
    pub rotated_keys: ArrayCell,
    pub extracted_values: ArrayCell,
    pub attention_partials: ArrayCell,
    pub attention_sums: ArrayCell,
    pub attention_maxs: ArrayCell,
}

impl CommonAuxBuffers {
    /// Create common auxiliary buffers from scratch buffers.
    pub fn new(
        scratch: &ScratchBuffers,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let act_dtype = model_shape.activation_data_type();
        unsafe {
            Self {
                suffix_length,
                main: RefCell::new(MetalArray::new(
                    scratch.main.clone(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                shortcut: RefCell::new(MetalArray::new(
                    scratch.shortcut.clone(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                qkv: RefCell::new(MetalArray::new(
                    scratch.qkv.clone(),
                    &model_shape.qkv_shape(suffix_length),
                    act_dtype,
                )),
                attention_output: RefCell::new(MetalArray::new(
                    scratch.attention_output.clone(),
                    &model_shape.attention_output_shape(suffix_length),
                    act_dtype,
                )),
                mlp_fused_up: RefCell::new(MetalArray::new(
                    scratch.mlp_fused_up.clone(),
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    act_dtype,
                )),
                mlp_hidden: RefCell::new(MetalArray::new(
                    scratch.mlp_hidden.clone(),
                    &model_shape.mlp_hidden_shape(suffix_length),
                    act_dtype,
                )),
                rotated_queries: RefCell::new(MetalArray::new(
                    scratch.rotated_queries.clone(),
                    &model_shape.rotated_queries_shape(suffix_length),
                    act_dtype,
                )),
                rotated_keys: RefCell::new(MetalArray::new(
                    scratch.rotated_keys.clone(),
                    &model_shape.rotated_keys_shape(suffix_length),
                    act_dtype,
                )),
                extracted_values: RefCell::new(MetalArray::new(
                    scratch.extracted_values.clone(),
                    &model_shape.extracted_values_shape(suffix_length),
                    act_dtype,
                )),
                attention_partials: RefCell::new(MetalArray::new(
                    scratch.attention_partials.clone(),
                    &model_shape.attention_partials_shape(suffix_length),
                    act_dtype,
                )),
                attention_sums: RefCell::new(MetalArray::new(
                    scratch.attention_sums.clone(),
                    &model_shape.attention_sums_shape(suffix_length),
                    act_dtype,
                )),
                attention_maxs: RefCell::new(MetalArray::new(
                    scratch.attention_maxs.clone(),
                    &model_shape.attention_maxs_shape(suffix_length),
                    act_dtype,
                )),
            }
        }
    }

    pub fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}
