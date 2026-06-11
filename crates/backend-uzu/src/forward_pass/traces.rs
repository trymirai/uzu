use crate::{
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    forward_pass::model_shape::ModelShape,
};

fn create_layer_results<B: Backend>(
    context: &B::Context,
    model_shape: &ModelShape,
    suffix_length: usize,
) -> Box<[LayerActivationTrace<B>]> {
    (0..model_shape.num_layers).map(|_| LayerActivationTrace::new(context, model_shape, suffix_length)).collect()
}

pub struct LayerActivationTrace<B: Backend> {
    pub inputs: Array<B>,
    pub pre_attention_norm: Array<B>,
    pub attention: Array<B>,
    pub post_attention_norm: Array<B>,
    pub mlp_inputs: Array<B>,
    pub pre_mlp_norm: Array<B>,
    pub mlp: Array<B>,
    pub post_mlp_norm: Array<B>,
    pub outputs: Array<B>,
}

impl<B: Backend> LayerActivationTrace<B> {
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

pub struct ActivationTrace<B: Backend> {
    pub embedding_norm: Option<Array<B>>,
    pub layer_results: Box<[LayerActivationTrace<B>]>,
    pub output_norm: Array<B>,
    pub output_pooling: Option<Array<B>>,
    pub logits: Array<B>,
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

    pub fn embedding_norm_mut(&mut self) -> &mut Array<B> {
        self.embedding_norm.as_mut().expect("embedding_norm is only available for classifier traces")
    }

    pub fn output_pooling_mut(&mut self) -> &mut Array<B> {
        self.output_pooling.as_mut().expect("output_pooling is only available for classifier traces")
    }
}

#[cfg(all(test, feature = "tracing"))]
#[path = "../../unit/forward_pass/traces/mod.rs"]
mod tests;
