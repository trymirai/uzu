use std::{collections::HashMap, rc::Rc};

use mpsgraph::{Device as MPSDevice, Graph};
use objc2::rc::autoreleasepool;

use crate::{
    DataType,
    backends::metal::{
        MTLContext,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ArrayId, IOArrays, MPSGraphBlock,
            encodable_with_state::EncodableWithState,
            transformer_layer::linear_block,
        },
        graph::{common::activation, placeholder, shaped_type},
        kernel::create_normalization_encodable,
    },
    config::PredictionHeadConfig,
    parameters::ParameterTree,
};

pub struct PredictionHeadExecutables {
    pub dense: Box<dyn EncodableWithState>,
    pub activation: Box<dyn EncodableWithState>,
    pub norm: Box<dyn EncodableWithState>,
    pub final_linear: Box<dyn EncodableWithState>,
}

impl PredictionHeadExecutables {
    pub fn new(
        mtl_context: Rc<MTLContext>,
        config: &PredictionHeadConfig,
        model_dim: usize,
        num_labels: usize,
        parameter_tree: &ParameterTree<Rc<MTLContext>>,
        compilation_config: Rc<CompilationConfig>,
    ) -> Self {
        let data_type: DataType =
            config.dense_config.activation_precision().into();

        eprintln!(
            "[DEBUG] PredictionHeadExecutables::new - Creating dense layer: input_dim={}, output_dim={}, has_bias={}",
            model_dim, model_dim, config.use_dense_bias
        );
        let dense = linear_block::<1>(
            &config.dense_config,
            config.use_dense_bias,
            model_dim,
            [model_dim],
            &mtl_context,
            &parameter_tree.subtree("dense").unwrap(),
            ArrayId::Main,
            ArrayId::Main,
            &compilation_config.descriptor_general,
        );
        eprintln!(
            "[DEBUG] PredictionHeadExecutables::new - Dense layer created"
        );

        let activation_block = Self::create_activation_block(
            &mtl_context,
            &config.activation,
            data_type,
            model_dim,
            &compilation_config,
        );

        let norm = create_normalization_encodable(
            &mtl_context,
            data_type,
            config.normalization_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &parameter_tree.subtree("norm").unwrap(),
        )
        .expect("Failed to create prediction head norm kernel");

        let final_linear = linear_block::<1>(
            &config.final_linear_config,
            true,
            model_dim,
            [num_labels],
            &mtl_context,
            &parameter_tree.subtree("final_linear").unwrap(),
            ArrayId::Main, // Input from Main [batch, model_dim]
            ArrayId::Main, // Output to Main [batch, num_labels] - MPSGraph writes num_labels elements
            &compilation_config.descriptor_general,
        );

        Self {
            dense,
            activation: activation_block,
            norm,
            final_linear,
        }
    }

    fn create_activation_block(
        mtl_context: &MTLContext,
        activation_config: &crate::config::Activation,
        data_type: DataType,
        model_dim: usize,
        compilation_config: &CompilationConfig,
    ) -> Box<dyn EncodableWithState> {
        autoreleasepool(|_| {
            let graph = Graph::new();

            let input_shape = [-1, model_dim as isize];
            let input_placeholder =
                placeholder(&graph, &input_shape, data_type);

            let output = activation(
                &graph,
                activation_config,
                &input_placeholder,
                data_type,
            );

            let shaped_type_obj = shaped_type(&input_shape, data_type);
            let feeds =
                HashMap::from([(&*input_placeholder, &*shaped_type_obj)]);

            let executable = graph.compile(
                &MPSDevice::with_device(&mtl_context.device),
                &feeds,
                &[&output],
                None,
                Some(&compilation_config.descriptor_general),
            );

            let arguments = IOArrays::new(
                vec![ArrayId::Main].into_boxed_slice(),
                vec![ArrayId::Main].into_boxed_slice(),
            );

            let execution_descriptor =
                mpsgraph::ExecutableExecutionDescriptor::new();
            execution_descriptor.set_enable_commit_and_continue(true);

            Box::new(MPSGraphBlock::new(
                executable,
                execution_descriptor,
                arguments,
            ))
        })
    }
}
