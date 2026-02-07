use std::{cell::RefCell, rc::Rc};

use super::super::{ModelShape, ScratchBuffers};
use crate::backends::metal::{MTLContext, MetalArray};

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
                main: RefCell::new(MetalArray::from_parts(
                    scratch.main.borrow_mut().buffer().into(),
                    0,
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                shortcut: RefCell::new(MetalArray::from_parts(
                    scratch.shortcut.borrow_mut().buffer().into(),
                    0,
                    &model_shape.main_shape(suffix_length),
                    act_dtype,
                )),
                qkv: RefCell::new(MetalArray::from_parts(
                    scratch.qkv.borrow_mut().buffer().into(),
                    0,
                    &model_shape.qkv_shape(suffix_length),
                    act_dtype,
                )),
                attention_output: RefCell::new(MetalArray::from_parts(
                    scratch.attention_output.borrow().buffer().clone(),
                    0,
                    &model_shape.attention_output_shape(suffix_length),
                    act_dtype,
                )),
                mlp_fused_up: RefCell::new(MetalArray::from_parts(
                    scratch.mlp_fused_up.borrow().buffer().clone(),
                    0,
                    &model_shape.mlp_fused_up_shape(suffix_length),
                    act_dtype,
                )),
                mlp_hidden: RefCell::new(MetalArray::from_parts(
                    scratch.mlp_hidden.borrow().buffer().clone(),
                    0,
                    &model_shape.mlp_hidden_shape(suffix_length),
                    act_dtype,
                )),
                rotated_queries: RefCell::new(MetalArray::from_parts(
                    scratch.rotated_queries.borrow().buffer().clone(),
                    0,
                    &model_shape.rotated_queries_shape(suffix_length),
                    act_dtype,
                )),
                rotated_keys: RefCell::new(MetalArray::from_parts(
                    scratch.rotated_keys.borrow().buffer().clone(),
                    0,
                    &model_shape.rotated_keys_shape(suffix_length),
                    act_dtype,
                )),
                extracted_values: RefCell::new(MetalArray::from_parts(
                    scratch.extracted_values.borrow().buffer().clone(),
                    0,
                    &model_shape.extracted_values_shape(suffix_length),
                    act_dtype,
                )),
                attention_partials: RefCell::new(MetalArray::from_parts(
                    scratch
                        .attention_partials
                        .borrow_mut()
                        .buffer()
                        .to_owned()
                        .into(),
                    0,
                    &model_shape.attention_partials_shape(suffix_length),
                    act_dtype,
                )),
                attention_sums: RefCell::new(MetalArray::from_parts(
                    scratch.attention_sums.borrow().buffer().clone(),
                    0,
                    &model_shape.attention_sums_shape(suffix_length),
                    act_dtype,
                )),
                attention_maxs: RefCell::new(MetalArray::from_parts(
                    scratch.attention_maxs.borrow().buffer().clone(),
                    0,
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
