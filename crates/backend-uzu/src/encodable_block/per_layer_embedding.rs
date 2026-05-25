use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        ActivationConfig, Allocation, Backend, Encoder,
        kernel::{
            FullPrecisionEmbeddingLookupKernel, Kernels, PleGateActMulKernel, TensorAddBiasKernel, TensorAddScaleKernel,
        },
    },
    config::{PLELayerConfig, PLEModelConfig},
    encodable_block::{Linear, PostLayerScalar, RMSNorm, RMSNormError, linear::LinearBlockError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum PerLayerEmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("RMSNorm error: {0}")]
    RMSNormError(#[from] RMSNormError<B>),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
}

pub struct PerLayerEmbedding<B: Backend> {
    token_embedding: Allocation<B>,
    token_embedding_lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    model_projection: Box<dyn Linear<B>>,
    projection_norm: RMSNorm<B>,
    add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    ple_dim: usize,
    num_layers: usize,
    ple_vocab_size: usize,
    model_dim: usize,
    fused_token_scale: f32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbedding<B> {
    pub fn new(
        context: &B::Context,
        config: &PLEModelConfig,
        model_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let total_ple_dim = config.num_layers * config.ple_dim;
        let activation_data_type: DataType = config.linear_config.activation_precision().into();

        let token_embedding_leaf = parameter_tree.leaf("token_embedding")?;
        token_embedding_leaf.validate_shape(&[config.ple_vocab_size, total_ple_dim], activation_data_type)?;

        let token_embedding = token_embedding_leaf.read_allocation()?;

        let token_embedding_lookup =
            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, activation_data_type)
                .map_err(PerLayerEmbeddingError::BackendError)?;

        let model_projection = <dyn Linear<B>>::new(
            &config.linear_config,
            model_dim,
            [total_ple_dim],
            context,
            &parameter_tree.subtree("model_projection")?,
        )?;

        let scale_squared = config.model_projection_scale * config.model_projection_scale;
        let projection_norm_config = {
            let mut adjusted = config.norm_config.clone();
            adjusted.epsilon /= scale_squared;
            adjusted
        };
        let projection_norm = RMSNorm::new(
            context,
            activation_data_type,
            projection_norm_config,
            &parameter_tree.subtree("projection_norm")?,
            None,
            false,
            false,
            PostLayerScalar::ScaleOutput(config.input_scale),
        )?;

        let add_scale = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, activation_data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        Ok(Self {
            token_embedding,
            token_embedding_lookup,
            model_projection,
            projection_norm,
            add_scale,
            ple_dim: config.ple_dim,
            num_layers: config.num_layers,
            ple_vocab_size: config.ple_vocab_size,
            model_dim,
            fused_token_scale: config.ple_embed_scale * config.input_scale,
            data_type: activation_data_type,
        })
    }

    pub fn encode(
        &self,
        token_ids: &Allocation<B>,
        inner_features: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let total_ple_dim = self.num_layers * self.ple_dim;
        let total_rows = batch_dim * self.num_layers;
        let total_elements = batch_dim * total_ple_dim;

        let mut token_ple = encoder.allocate_scratch(size_for_shape(&[batch_dim, total_ple_dim], self.data_type))?;
        self.token_embedding_lookup.encode(
            token_ids,
            &self.token_embedding,
            &mut token_ple,
            batch_dim as u32,
            self.ple_vocab_size as u32,
            total_ple_dim as u32,
            self.fused_token_scale,
            encoder,
        );

        let mut model_projection_input =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        encoder.encode_copy(inner_features, .., &mut model_projection_input, ..);
        let model_projected = self.model_projection.encode(model_projection_input, batch_dim, encoder)?;

        let model_normed = self.projection_norm.encode(&model_projected, 0, total_rows, None, encoder)?;

        let mut per_layer_inputs =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.num_layers, self.ple_dim], self.data_type))?;
        self.add_scale.encode(
            &token_ple,
            &model_normed,
            &mut per_layer_inputs,
            total_elements as u32,
            total_elements as u32,
            1.0,
            encoder,
        );

        Ok(per_layer_inputs)
    }
}

pub struct PerLayerEmbeddingProjection<B: Backend> {
    gate: Box<dyn Linear<B>>,
    projection: Box<dyn Linear<B>>,
    norm: RMSNorm<B>,
    gate_act_mul: <B::Kernels as Kernels>::PleGateActMulKernel,
    residual_finalize: <B::Kernels as Kernels>::TensorAddBiasKernel,
    residual_combine: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model_dim: usize,
    ple_dim: usize,
    num_layers: usize,
    activation: ActivationConfig,
    post_layer_scalar: f32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbeddingProjection<B> {
    pub fn new(
        context: &B::Context,
        config: &PLELayerConfig,
        model_dim: usize,
        num_layers: usize,
        post_layer_scalar: f32,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let activation_data_type: DataType = config.linear_config.activation_precision().into();

        let gate = <dyn Linear<B>>::new(
            &config.linear_config,
            model_dim,
            [config.ple_dim],
            context,
            &parameter_tree.subtree("gate")?,
        )?;
        let projection = <dyn Linear<B>>::new(
            &config.linear_config,
            config.ple_dim,
            [model_dim],
            context,
            &parameter_tree.subtree("projection")?,
        )?;
        let norm = RMSNorm::new(
            context,
            activation_data_type,
            config.norm_config.clone(),
            &parameter_tree.subtree("norm")?,
            None,
            false,
            false,
            PostLayerScalar::None,
        )?;

        let gate_act_mul = <B::Kernels as Kernels>::PleGateActMulKernel::new(context, activation_data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_finalize = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, activation_data_type, true)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_combine = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, activation_data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        Ok(Self {
            gate,
            projection,
            norm,
            gate_act_mul,
            residual_finalize,
            residual_combine,
            model_dim,
            ple_dim: config.ple_dim,
            num_layers,
            activation: config.activation,
            post_layer_scalar,
            data_type: activation_data_type,
        })
    }

    pub fn encode(
        &self,
        layer_index: usize,
        per_layer_input: &Allocation<B>,
        outputs: &mut Allocation<B>,
        hidden: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let length = batch_dim * self.model_dim;

        self.residual_finalize.encode(
            None::<&Allocation<B>>,
            hidden,
            &mut *outputs,
            length as u32,
            length as u32,
            encoder,
        );

        let mut gate_input = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        encoder.encode_copy(outputs, .., &mut gate_input, ..);
        let gate_out = self.gate.encode(gate_input, batch_dim, encoder)?;

        let mut activated = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.ple_dim], self.data_type))?;
        self.gate_act_mul.encode(
            &gate_out,
            per_layer_input,
            &mut activated,
            self.ple_dim as i32,
            batch_dim as i32,
            self.num_layers as i32,
            (layer_index * self.ple_dim) as i32,
            self.activation.act_type(),
            encoder,
        );

        let projected = self.projection.encode(activated, batch_dim, encoder)?;
        let normed = self.norm.encode(&projected, 0, batch_dim, None, encoder)?;

        let mut combined = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        self.residual_combine.encode(
            &*outputs,
            &normed,
            &mut combined,
            length as u32,
            length as u32,
            self.post_layer_scalar,
            encoder,
        );
        encoder.encode_copy(&combined, .., &mut *outputs, ..);

        Ok(())
    }
}
