use crate::{
    array::{ArrayCell, ArrayCellExt},
    backends::common::Backend,
    forward_pass::{model_shape::ModelShape, scratch_buffers::ScratchBuffers},
};

pub struct CommonAuxBuffers<B: Backend> {
    pub suffix_length: usize,
    pub main: ArrayCell<B>,
    pub shortcut: ArrayCell<B>,
    pub qkv: ArrayCell<B>,
    pub attention_output: ArrayCell<B>,
    pub mlp_fused_up: ArrayCell<B>,
    pub mlp_hidden: ArrayCell<B>,
    pub rotated_queries: ArrayCell<B>,
    pub rotated_keys: ArrayCell<B>,
    pub extracted_values: ArrayCell<B>,
    pub attention_partials: ArrayCell<B>,
    pub attention_sums: ArrayCell<B>,
    pub attention_maxs: ArrayCell<B>,
}

impl<B: Backend> CommonAuxBuffers<B> {
    pub fn new(
        scratch: &ScratchBuffers<B>,
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
}
