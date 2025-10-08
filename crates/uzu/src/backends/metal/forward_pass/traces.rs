use std::{cell::RefCell, rc::Rc};

use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray, ModelShape},
};

type ArrayCell = RefCell<MetalArray>;

pub struct DecoderLayerActivationTrace {
    pub inputs: ArrayCell,
    pub pre_attention_norm: ArrayCell,
    pub attention: ArrayCell,
    pub post_attention_norm: ArrayCell,
    pub mlp_inputs: ArrayCell,
    pub pre_mlp_norm: ArrayCell,
    pub mlp: ArrayCell,
    pub post_mlp_norm: ArrayCell,
    pub outputs: ArrayCell,
}

impl DecoderLayerActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        unsafe {
            Self {
                inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                attention: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_attention_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp_inputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                pre_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                mlp: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                post_mlp_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                outputs: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}

pub struct DecoderActivationTrace {
    pub layer_results: Vec<Rc<RefCell<DecoderLayerActivationTrace>>>,
    pub output_norm: ArrayCell,
    pub logits: ArrayCell,
}

impl DecoderActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(DecoderLayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        unsafe {
            Self {
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                logits: RefCell::new(context.array_uninitialized(
                    &model_shape.logits_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}
