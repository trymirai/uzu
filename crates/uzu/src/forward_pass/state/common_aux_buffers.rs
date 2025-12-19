use std::cell::RefCell;

use super::super::{ModelShape, ScratchBuffers};
use crate::{Array, DeviceContext};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub struct CommonAuxBuffers<C: DeviceContext> {
    pub suffix_length: usize,
    pub main: ArrayCell<C>,
    pub shortcut: ArrayCell<C>,
    pub qkv: ArrayCell<C>,
    pub attention_output: ArrayCell<C>,
    pub mlp_fused_up: ArrayCell<C>,
    pub mlp_hidden: ArrayCell<C>,
    pub rotated_queries: ArrayCell<C>,
    pub rotated_keys: ArrayCell<C>,
    pub extracted_values: ArrayCell<C>,
    pub attention_partials: ArrayCell<C>,
    pub attention_sums: ArrayCell<C>,
    pub attention_maxs: ArrayCell<C>,
}

impl<C: DeviceContext> CommonAuxBuffers<C> {
    pub fn new(
        scratch: &ScratchBuffers<C>,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        Self {
            suffix_length,
            main: RefCell::new(
                scratch.main.reshape(&model_shape.main_shape(suffix_length)),
            ),
            shortcut: RefCell::new(
                scratch
                    .shortcut
                    .reshape(&model_shape.main_shape(suffix_length)),
            ),
            qkv: RefCell::new(
                scratch.qkv.reshape(&model_shape.qkv_shape(suffix_length)),
            ),
            attention_output: RefCell::new(
                scratch.attention_output.reshape(
                    &model_shape.attention_output_shape(suffix_length),
                ),
            ),
            mlp_fused_up: RefCell::new(
                scratch
                    .mlp_fused_up
                    .reshape(&model_shape.mlp_fused_up_shape(suffix_length)),
            ),
            mlp_hidden: RefCell::new(
                scratch
                    .mlp_hidden
                    .reshape(&model_shape.mlp_hidden_shape(suffix_length)),
            ),
            rotated_queries: RefCell::new(
                scratch
                    .rotated_queries
                    .reshape(&model_shape.rotated_queries_shape(suffix_length)),
            ),
            rotated_keys: RefCell::new(
                scratch
                    .rotated_keys
                    .reshape(&model_shape.rotated_keys_shape(suffix_length)),
            ),
            extracted_values: RefCell::new(
                scratch.extracted_values.reshape(
                    &model_shape.extracted_values_shape(suffix_length),
                ),
            ),
            attention_partials: RefCell::new(
                scratch.attention_partials.reshape(
                    &model_shape.attention_partials_shape(suffix_length),
                ),
            ),
            attention_sums: RefCell::new(
                scratch
                    .attention_sums
                    .reshape(&model_shape.attention_sums_shape(suffix_length)),
            ),
            attention_maxs: RefCell::new(
                scratch
                    .attention_maxs
                    .reshape(&model_shape.attention_maxs_shape(suffix_length)),
            ),
        }
    }

    pub fn suffix_length(&self) -> usize {
        self.suffix_length
    }
}
