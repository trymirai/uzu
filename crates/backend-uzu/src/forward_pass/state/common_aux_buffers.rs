use crate::{
    array::Array,
    backends::common::Backend,
    forward_pass::{model_shape::ModelShape, scratch_buffers::ScratchBuffers},
};

pub struct CommonAuxBuffers<B: Backend> {
    pub suffix_length: usize,
    pub main: Array<B>,
    pub shortcut: Array<B>,
    pub qkv: Array<B>,
    pub gate: Option<Array<B>>,
    pub attention_output: Array<B>,
    pub mlp_fused_up: Array<B>,
    pub mlp_hidden: Array<B>,
    pub lora_intermediate: Option<Array<B>>,
    pub rotated_queries: Array<B>,
    pub rotated_keys: Array<B>,
    pub extracted_values: Array<B>,
    pub attention_partials: Array<B>,
    pub attention_sums: Array<B>,
    pub attention_maxs: Array<B>,
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
            shortcut: scratch.shortcut.view(&model_shape.main_shape(suffix_length)),
            qkv: scratch.qkv.view(&model_shape.qkv_shape(suffix_length)),
            gate: scratch.gate.as_ref().zip(model_shape.gate_shape(suffix_length)).map(|(buf, shape)| buf.view(&shape)),
            attention_output: scratch.attention_output.view(&model_shape.attention_output_shape(suffix_length)),
            mlp_fused_up: scratch.mlp_fused_up.view(&model_shape.mlp_fused_up_shape(suffix_length)),
            mlp_hidden: scratch.mlp_hidden.view(&model_shape.mlp_hidden_shape(suffix_length)),
            lora_intermediate: scratch
                .lora_intermediate
                .as_ref()
                .zip(model_shape.lora_intermediate(suffix_length))
                .map(|(buf, shape)| buf.view(&shape)),
            rotated_queries: scratch.rotated_queries.view(&model_shape.rotated_queries_shape(suffix_length)),
            rotated_keys: scratch.rotated_keys.view(&model_shape.rotated_keys_shape(suffix_length)),
            extracted_values: scratch.extracted_values.view(&model_shape.extracted_values_shape(suffix_length)),
            attention_partials: scratch.attention_partials.view(&model_shape.attention_partials_shape(suffix_length)),
            attention_sums: scratch.attention_sums.view(&model_shape.attention_sums_shape(suffix_length)),
            attention_maxs: scratch.attention_maxs.view(&model_shape.attention_maxs_shape(suffix_length)),
        }
    }
}
