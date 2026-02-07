use std::{cell::RefCell, rc::Rc};

use super::super::{ModelShape, ScratchBuffers};
use crate::{
    array::ArrayCellExt,
    backends::metal::{MTLContext, MetalArray},
};

type ArrayCell = RefCell<MetalArray>;

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
    pub fn new(
        scratch: &ScratchBuffers<Rc<MTLContext>>,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        Self {
            suffix_length,
            main: scratch.main.view(&model_shape.main_shape(suffix_length)),
            shortcut: scratch
                .shortcut
                .view(&model_shape.main_shape(suffix_length)),
            qkv: scratch.qkv.view(&model_shape.qkv_shape(suffix_length)),
            attention_output: scratch
                .attention_output
                .view(&model_shape.attention_output_shape(suffix_length)),
            mlp_fused_up: scratch
                .mlp_fused_up
                .view(&model_shape.mlp_fused_up_shape(suffix_length)),
            mlp_hidden: scratch
                .mlp_hidden
                .view(&model_shape.mlp_hidden_shape(suffix_length)),
            rotated_queries: scratch
                .rotated_queries
                .view(&model_shape.rotated_queries_shape(suffix_length)),
            rotated_keys: scratch
                .rotated_keys
                .view(&model_shape.rotated_keys_shape(suffix_length)),
            extracted_values: scratch
                .extracted_values
                .view(&model_shape.extracted_values_shape(suffix_length)),
            attention_partials: scratch
                .attention_partials
                .view(&model_shape.attention_partials_shape(suffix_length)),
            attention_sums: scratch
                .attention_sums
                .view(&model_shape.attention_sums_shape(suffix_length)),
            attention_maxs: scratch
                .attention_maxs
                .view(&model_shape.attention_maxs_shape(suffix_length)),
        }
    }

    pub fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}
