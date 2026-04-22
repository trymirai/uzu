use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, Backend},
    forward_pass::model_shape::ModelShape,
};

fn create_trace_allocation<B: Backend>(
    context: &B::Context,
    shape: &[usize],
    data_type: DataType,
) -> Allocation<B> {
    context.create_array_uninitialized(shape, data_type, "activation_trace").into_allocation()
}

fn create_layer_results<B: Backend>(
    context: &B::Context,
    model_shape: &ModelShape,
    suffix_length: usize,
) -> Box<[LayerActivationTrace<B>]> {
    (0..model_shape.num_layers).map(|_| LayerActivationTrace::new(context, model_shape, suffix_length)).collect()
}

pub struct LayerActivationTrace<B: Backend> {
    pub inputs: Allocation<B>,
    pub pre_attention_norm: Allocation<B>,
    pub attention: Allocation<B>,
    pub post_attention_norm: Allocation<B>,
    pub mlp_inputs: Allocation<B>,
    pub pre_mlp_norm: Allocation<B>,
    pub mlp: Allocation<B>,
    pub post_mlp_norm: Allocation<B>,
    pub outputs: Allocation<B>,
}

impl<B: Backend> LayerActivationTrace<B> {
    pub fn new(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let main_shape = model_shape.main_shape(suffix_length);
        let activation_data_type = model_shape.activation_data_type();
        let main = || create_trace_allocation(context, &main_shape, activation_data_type);

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
    pub embedding_norm: Option<Allocation<B>>,
    pub layer_results: Box<[LayerActivationTrace<B>]>,
    pub output_norm: Allocation<B>,
    pub output_pooling: Option<Allocation<B>>,
    pub logits: Allocation<B>,
}

impl<B: Backend> ActivationTrace<B> {
    pub fn new_llm(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
    ) -> Self {
        let activation_data_type = model_shape.activation_data_type();
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results = create_layer_results(context, model_shape, suffix_length);

        Self {
            embedding_norm: None,
            layer_results,
            output_norm: create_trace_allocation(context, &main_shape, activation_data_type),
            output_pooling: None,
            logits: create_trace_allocation(context, &model_shape.logits_shape(suffix_length), activation_data_type),
        }
    }

    pub fn new_classifier(
        context: &B::Context,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let activation_data_type = model_shape.activation_data_type();
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results = create_layer_results(context, model_shape, suffix_length);
        let model_dim = model_shape.main_shape(1)[1];

        Self {
            embedding_norm: Some(create_trace_allocation(context, &main_shape, activation_data_type)),
            layer_results,
            output_norm: create_trace_allocation(context, &main_shape, activation_data_type),
            output_pooling: Some(create_trace_allocation(context, &[1, model_dim], activation_data_type)),
            logits: create_trace_allocation(context, &[1, num_labels], activation_data_type),
        }
    }

    pub fn embedding_norm_mut(&mut self) -> &mut Allocation<B> {
        self.embedding_norm.as_mut().expect("embedding_norm is only available for classifier traces")
    }

    pub fn output_pooling_mut(&mut self) -> &mut Allocation<B> {
        self.output_pooling.as_mut().expect("output_pooling is only available for classifier traces")
    }
}
