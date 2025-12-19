use std::{cell::RefCell, rc::Rc};

use crate::{DeviceContext, forward_pass::model_shape::ModelShape};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub struct LayerActivationTrace<C: DeviceContext> {
    pub inputs: ArrayCell<C>,
    pub pre_attention_norm: ArrayCell<C>,
    pub attention: ArrayCell<C>,
    pub post_attention_norm: ArrayCell<C>,
    pub mlp_inputs: ArrayCell<C>,
    pub pre_mlp_norm: ArrayCell<C>,
    pub mlp: ArrayCell<C>,
    pub post_mlp_norm: ArrayCell<C>,
    pub outputs: ArrayCell<C>,
}

impl<C: DeviceContext> LayerActivationTrace<C> {
    pub fn new(
        context: &C,
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

pub struct ActivationTrace<C: DeviceContext> {
    pub embedding_norm: Option<ArrayCell<C>>,
    pub layer_results: Vec<Rc<RefCell<LayerActivationTrace<C>>>>,
    pub output_norm: ArrayCell<C>,
    pub output_pooling: Option<ArrayCell<C>>,
    pub logits: ArrayCell<C>,
}

impl<C: DeviceContext> ActivationTrace<C> {
    pub fn new_llm(
        context: &C,
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
        context: &C,
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

impl<C: DeviceContext> ActivationTrace<C> {
    pub fn embedding_norm(&self) -> &ArrayCell<C> {
        self.embedding_norm
            .as_ref()
            .expect("embedding_norm is only available for classifier traces")
    }

    pub fn output_pooling(&self) -> &ArrayCell<C> {
        self.output_pooling
            .as_ref()
            .expect("output_pooling is only available for classifier traces")
    }
}
