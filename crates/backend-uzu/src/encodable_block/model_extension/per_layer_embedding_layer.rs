use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    DataType,
    backends::common::{Backend, Encoder, Kernels, kernel::TensorMulSliceKernel},
    config::PerLayerEmbeddingLayerConfig,
    encodable_block::{
        Activation, Linear, RMSNorm, TensorAddSwap,
        model_extension::{ModelExtensionError, validate_tensor},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::ParameterTree,
};

pub struct PerLayerEmbeddingLayer<B: Backend> {
    materialize_residual: TensorAddSwap<B>,
    gate_projection: Box<dyn Linear<B>>,
    activation: Activation<B>,
    multiply_by_embedding: <B::Kernels as Kernels>::TensorMulSliceKernel,
    output_projection: Box<dyn Linear<B>>,
    output_norm: RMSNorm<B>,
    layer_index: usize,
    per_layer_embedding_dim: usize,
    total_per_layer_embedding_dim: usize,
}

impl<B: Backend> PerLayerEmbeddingLayer<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        layer_config: &PerLayerEmbeddingLayerConfig,
        layer_index: usize,
        model_dim: usize,
        num_layers: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, ModelExtensionError<B>> {
        let materialize_residual = TensorAddSwap::new(context, data_type, ArrayId::Shortcut, ArrayId::Main)
            .map_err(ModelExtensionError::BackendError)?;
        let gate_projection = <dyn Linear<B>>::new(
            &layer_config.linear_config,
            false,
            model_dim,
            [layer_config.ple_dim],
            context,
            &parameter_tree.subtree("gate")?,
            ArrayId::Main,
            ArrayId::PleGate,
        )?;
        let activation =
            Activation::new(context, data_type, layer_config.activation, ArrayId::PleGate, ArrayId::PleGate)
                .map_err(ModelExtensionError::BackendError)?;
        let multiply_by_embedding = <B::Kernels as Kernels>::TensorMulSliceKernel::new(context, data_type)
            .map_err(ModelExtensionError::BackendError)?;
        let output_projection = <dyn Linear<B>>::new(
            &layer_config.linear_config,
            false,
            layer_config.ple_dim,
            [model_dim],
            context,
            &parameter_tree.subtree("projection")?,
            ArrayId::PleGate,
            ArrayId::Main,
        )?;
        let norm_tree = parameter_tree.subtree("norm")?;
        let output_norm = RMSNorm::new(
            context,
            data_type,
            layer_config.norm_config.clone(),
            ArrayId::Main,
            ArrayId::Main,
            &norm_tree,
            None,
            None,
            false,
        )?;

        Ok(Self {
            materialize_residual,
            gate_projection,
            activation,
            multiply_by_embedding,
            output_projection,
            output_norm,
            layer_index,
            per_layer_embedding_dim: layer_config.ple_dim,
            total_per_layer_embedding_dim: num_layers * layer_config.ple_dim,
        })
    }

    pub fn validate_post_layer_scalar(
        data_type: DataType,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Rc<RefCell<B::DenseBuffer>>, ModelExtensionError<B>> {
        let scalar = parameter_tree.leaf_array("post_layer_scalar")?;
        validate_tensor::<B>(scalar.shape(), scalar.data_type(), &[1], data_type)?;
        Ok(scalar.buffer())
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.materialize_residual.encode(state, encoder)?;
        self.gate_projection.encode(state, encoder)?;
        self.activation.encode(state, encoder)?;

        let ple_gate = state.array(ArrayId::PleGate);
        let ple_combined = state.array(ArrayId::PleCombined);
        self.multiply_by_embedding.encode(
            ple_gate.buffer().borrow_mut().deref_mut(),
            ple_combined.buffer().borrow().deref(),
            state.active_row_count() as u32,
            self.total_per_layer_embedding_dim as u32,
            self.per_layer_embedding_dim as u32,
            self.layer_index as u32,
            encoder,
        );

        self.output_projection.encode(state, encoder)?;
        self.output_norm.encode(state, encoder)?;
        Ok(())
    }
}
