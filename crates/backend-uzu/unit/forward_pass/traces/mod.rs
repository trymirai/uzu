mod trace_validator;
mod trace_validator_test;

use crate::{
    array::ArrayContextExt,
    backends::common::Backend,
    classifier::ActivationTrace,
    forward_pass::{model_shape::ModelShape, traces::LayerActivationTrace},
};

fn create_layer_results<B: Backend>(
    context: &B::Context,
    model_shape: &ModelShape,
    suffix_length: usize,
) -> Box<[LayerActivationTrace<B>]> {
    (0..model_shape.num_layers).map(|_| LayerActivationTrace::new(context, model_shape, suffix_length)).collect()
}

impl<B: Backend> LayerActivationTrace<B> {
    #[cfg(all(test, feature = "tracing"))]
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let main_shape = model_shape.main_shape(suffix_length);
        let main = || context.create_array_uninitialized(&main_shape, model_shape.data_type);

        Self {
            inputs: main(),
            pre_attention_norm: main(),
            attention: main(),
            post_attention_norm: main(),
            mlp_inputs: main(),
            pre_mlp_norm: main(),
            mlp: main(),
            post_mlp_norm: main(),
            outputs: main(),
        }
    }
}

impl<B: Backend> ActivationTrace<B> {
    pub fn new_llm(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results = create_layer_results(context, model_shape, suffix_length);

        Self {
            embedding_norm: None,
            layer_results,
            output_norm: context.create_array_uninitialized(&main_shape, model_shape.data_type),
            output_pooling: None,
            logits: context.create_array_uninitialized(&model_shape.logits_shape(suffix_length), model_shape.data_type),
        }
    }

    pub fn new_classifier(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results = create_layer_results(context, model_shape, suffix_length);
        let model_dim = model_shape.main_shape(1)[1];

        Self {
            embedding_norm: Some(context.create_array_uninitialized(&main_shape, model_shape.data_type)),
            layer_results,
            output_norm: context.create_array_uninitialized(&main_shape, model_shape.data_type),
            output_pooling: Some(context.create_array_uninitialized(&[1, model_dim], model_shape.data_type)),
            logits: context.create_array_uninitialized(&[1, num_labels], model_shape.data_type),
        }
    }
}
