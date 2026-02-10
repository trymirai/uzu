use std::{cell::RefCell, rc::Rc};

use crate::{
    DataType,
    array::ArrayContextExt,
    backends::metal::{MTLContext, MetalArray},
    forward_pass::model_shape::ModelShape,
};

type ArrayCell = RefCell<MetalArray>;

fn create_trace_array(
    context: &MTLContext,
    shape: &[usize],
    data_type: DataType,
    label: &str,
) -> ArrayCell {
    RefCell::new(context.create_array_uninitialized(shape, data_type, label))
}

fn create_layer_results(
    context: &MTLContext,
    model_shape: &ModelShape,
    suffix_length: usize,
) -> Vec<Rc<RefCell<LayerActivationTrace>>> {
    (0..model_shape.num_layers)
        .map(|_| {
            Rc::new(RefCell::new(LayerActivationTrace::new(
                context,
                model_shape,
                suffix_length,
            )))
        })
        .collect()
}

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
        let main_shape = model_shape.main_shape(suffix_length);
        let activation_data_type = model_shape.activation_data_type();
        let main = |label| {
            create_trace_array(
                context,
                &main_shape,
                activation_data_type,
                label,
            )
        };

        Self {
            inputs: main("layer_activation_trace_inputs"),
            pre_attention_norm: main(
                "layer_activation_trace_pre_attention_norm",
            ),
            attention: main("layer_activation_trace_attention"),
            post_attention_norm: main(
                "layer_activation_trace_post_attention_norm",
            ),
            mlp_inputs: main("layer_activation_trace_mlp_inputs"),
            pre_mlp_norm: main("layer_activation_trace_pre_mlp_norm"),
            mlp: main("layer_activation_trace_mlp"),
            post_mlp_norm: main("layer_activation_trace_post_mlp_norm"),
            outputs: main("layer_activation_trace_outputs"),
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
        let activation_data_type = model_shape.activation_data_type();
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results =
            create_layer_results(context, model_shape, suffix_length);

        Self {
            embedding_norm: None,
            layer_results,
            output_norm: create_trace_array(
                context,
                &main_shape,
                activation_data_type,
                "activation_trace_output_norm",
            ),
            output_pooling: None,
            logits: create_trace_array(
                context,
                &model_shape.logits_shape(suffix_length),
                activation_data_type,
                "activation_trace_logits",
            ),
        }
    }

    pub fn new_classifier(
        context: &MTLContext,
        model_shape: &ModelShape,
        suffix_length: usize,
        num_labels: usize,
    ) -> Self {
        let activation_data_type = model_shape.activation_data_type();
        let main_shape = model_shape.main_shape(suffix_length);
        let layer_results =
            create_layer_results(context, model_shape, suffix_length);
        let model_dim = model_shape.main_shape(1)[1];

        Self {
            embedding_norm: Some(create_trace_array(
                context,
                &main_shape,
                activation_data_type,
                "activation_trace_embedding_norm",
            )),
            layer_results,
            output_norm: create_trace_array(
                context,
                &main_shape,
                activation_data_type,
                "activation_trace_output_norm",
            ),
            output_pooling: Some(create_trace_array(
                context,
                &[1, model_dim],
                activation_data_type,
                "activation_trace_output_pooling",
            )),
            logits: create_trace_array(
                context,
                &[1, num_labels],
                activation_data_type,
                "activation_trace_logits",
            ),
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
