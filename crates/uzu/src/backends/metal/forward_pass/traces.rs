use std::{cell::RefCell, rc::Rc};

use crate::{
    DeviceContext,
    backends::metal::{MTLContext, MetalArray, ModelShape},
};

type ArrayCell = RefCell<MetalArray>;

pub struct LayerActivationTrace {
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

impl LayerActivationTrace {
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

pub struct ActivationTrace {
    pub embedding_norm: Option<ArrayCell>,
    pub layer_results: Vec<Rc<RefCell<LayerActivationTrace>>>,
    pub output_norm: ArrayCell,
    pub output_pooling: Option<ArrayCell>,
    pub logits: ArrayCell,
}

impl ActivationTrace {
    pub fn new_llm(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(LayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        unsafe {
            Self {
                embedding_norm: None,
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                output_pooling: None,
                logits: RefCell::new(context.array_uninitialized(
                    &model_shape.logits_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
            }
        }
    }

    pub fn new_classifier(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let layer_results = (0..model_shape.num_layers)
            .map(|_| {
                Rc::new(RefCell::new(LayerActivationTrace::new(
                    context,
                    model_shape,
                    suffix_length,
                )))
            })
            .collect();
        let model_dim = model_shape.main_shape(1)[1];
        unsafe {
            Self {
                embedding_norm: Some(RefCell::new(
                    context.array_uninitialized(
                        &model_shape.main_shape(suffix_length),
                        model_shape.activation_data_type(),
                    ),
                )),
                layer_results,
                output_norm: RefCell::new(context.array_uninitialized(
                    &model_shape.main_shape(suffix_length),
                    model_shape.activation_data_type(),
                )),
                output_pooling: Some(RefCell::new(
                    context.array_uninitialized(
                        &[1, model_dim],
                        model_shape.activation_data_type(),
                    ),
                )),
                logits: RefCell::new(context.array_uninitialized(
                    &[1, num_labels],
                    model_shape.activation_data_type(),
                )),
            }
        }
    }
}

impl ActivationTrace {
    pub fn embedding_norm(&self) -> &ArrayCell {
        self.embedding_norm
            .as_ref()
            .expect("embedding_norm is only available for classifier traces")
    }

    pub fn output_pooling(&self) -> &ArrayCell {
        self.output_pooling
            .as_ref()
            .expect("output_pooling is only available for classifier traces")
    }
}
