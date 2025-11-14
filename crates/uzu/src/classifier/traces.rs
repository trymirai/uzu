use std::{cell::RefCell, rc::Rc};

use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray, forward_pass::ModelShape},
};

type ArrayCell = RefCell<MetalArray>;

pub struct ClassifierLayerActivationTrace {
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

impl ClassifierLayerActivationTrace {
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

pub struct ClassifierActivationTrace {
    pub embedding_norm: ArrayCell,
    pub layer_results: Vec<Rc<RefCell<ClassifierLayerActivationTrace>>>,
    pub output_norm: ArrayCell,
    pub output_pooling: ArrayCell,
    pub prediction_dense_output: ArrayCell,
    pub prediction_gelu_output: ArrayCell,
    pub prediction_norm_output: ArrayCell,
    pub logits: ArrayCell,
}

impl ClassifierActivationTrace {
    pub fn new(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(ClassifierLayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        let model_dim = model_shape.main_shape(1)[1];
        unsafe {
            Self {
                embedding_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                output_pooling: RefCell::new(context.array_uninitialized(
                    &[1, model_dim],
                    model_shape.activation_data_type(),
                )),
                prediction_dense_output: RefCell::new(
                    context.array_uninitialized(
                        &[1, model_dim],
                        model_shape.activation_data_type(),
                    ),
                ),
                prediction_gelu_output: RefCell::new(
                    context.array_uninitialized(
                        &[1, model_dim],
                        model_shape.activation_data_type(),
                    ),
                ),
                prediction_norm_output: RefCell::new(
                    context.array_uninitialized(
                        &[1, model_dim],
                        model_shape.activation_data_type(),
                    ),
                ),
                logits: RefCell::new(context.array_uninitialized(
                    &[1, num_labels],
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}
