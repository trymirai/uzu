use std::{cell::RefCell, rc::Rc};

use super::super::{ModelShape, ScratchBuffers};
use crate::{
    Array,
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
        let act_dtype = model_shape.activation_data_type();
        unsafe {
            Self {
                suffix_length,
                main: RefCell::new(MetalArray::new(
                    scratch.main.borrow_mut().mtl_buffer().into(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                shortcut: RefCell::new(MetalArray::new(
                    scratch.shortcut.borrow_mut().mtl_buffer().into(),
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                qkv: RefCell::new(MetalArray::new(
                    scratch.qkv.borrow_mut().mtl_buffer().into(),
                    &model_shape.qkv_shape(suffix_length),
                    act_dtype,
                )),
                attention_output: RefCell::new(MetalArray::new(
                    scratch.attention_output.borrow().mtl_buffer_cloned(),
                    &model_shape.attention_output_shape(suffix_length),
                    act_dtype,
                )),
                mlp_fused_up: RefCell::new(MetalArray::new(
                    scratch.mlp_fused_up.borrow().mtl_buffer_cloned(),
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    act_dtype,
                )),
                mlp_hidden: RefCell::new(MetalArray::new(
                    scratch.mlp_hidden.borrow().mtl_buffer_cloned(),
                    &model_shape.mlp_hidden_shape(suffix_length),
                    act_dtype,
                )),
                rotated_queries: RefCell::new(MetalArray::new(
                    scratch.rotated_queries.borrow().mtl_buffer_cloned(),
                    &model_shape.rotated_queries_shape(suffix_length),
                    act_dtype,
                )),
                rotated_keys: RefCell::new(MetalArray::new(
                    scratch.rotated_keys.borrow().mtl_buffer_cloned(),
                    &model_shape.rotated_keys_shape(suffix_length),
                    act_dtype,
                )),
                extracted_values: RefCell::new(MetalArray::new(
                    scratch.extracted_values.borrow().mtl_buffer_cloned(),
                    &model_shape.extracted_values_shape(suffix_length),
                    act_dtype,
                )),
                attention_partials: RefCell::new(MetalArray::new(
                    scratch
                        .attention_partials
                        .borrow_mut()
                        .backend_buffer()
                        .to_owned()
                        .into(),
                    &model_shape.attention_partials_shape(suffix_length),
                    act_dtype,
                )),
                attention_sums: RefCell::new(MetalArray::new(
                    scratch.attention_sums.borrow().mtl_buffer_cloned(),
                    &model_shape.attention_sums_shape(suffix_length),
                    act_dtype,
                )),
                attention_maxs: RefCell::new(MetalArray::new(
                    scratch.attention_maxs.borrow().mtl_buffer_cloned(),
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
