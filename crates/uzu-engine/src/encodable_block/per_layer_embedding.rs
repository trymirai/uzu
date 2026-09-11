use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{GatedActMul, GatedActMulSettings, Kernels, TensorAddBiasKernel, TensorAddScaleKernel},
    },
    config::{
        activation::AnyActivation,
        per_layer_embedding::{PLELayerConfig, PLEModelConfig},
    },
    data_type::DataType,
    encodable_block::{
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum PerLayerEmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
    #[error("Embedding table error: {0}")]
    EmbeddingTable(#[from] EmbeddingTableError<B>),
}

pub struct PerLayerEmbedding<B: Backend> {
    token_embedding: EmbeddingTable<B>,
    model_projection: Box<dyn Linear<B>>,
    projection_norm: Normalization<B>,
    add_scale: <B::Kernels as Kernels>::TensorAddScaleKernel,
    ple_dim: u32,
    num_layers: u32,
    model_dim: u32,
    fused_token_scale: f32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbedding<B> {
    pub fn new(
        context: &B::Context,
        config: &PLEModelConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let total_ple_dim = config.num_layers * config.ple_dim;

        let token_embedding = EmbeddingTable::load(
            context,
            &parameter_tree.subtree("token_embedding"),
            config.ple_vocab_size,
            total_ple_dim,
            data_type,
        )?;

        let model_projection = <dyn Linear<B>>::new(
            model_dim,
            [total_ple_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("model_projection"),
        )?;

        let scale_squared = config.model_projection_scale * config.model_projection_scale;
        let projection_norm_config = {
            let mut adjusted = config.norm_config.clone();
            adjusted.epsilon /= scale_squared;
            adjusted
        };
        let projection_norm = Normalization::new(
            config.ple_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::ScaleOutput(config.input_scale),
            data_type,
            &projection_norm_config,
            &parameter_tree.subtree("projection_norm"),
            context,
        )?;

        let add_scale = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        Ok(Self {
            token_embedding,
            model_projection,
            projection_norm,
            add_scale,
            ple_dim: config.ple_dim,
            num_layers: config.num_layers,
            model_dim,
            fused_token_scale: config.ple_embed_scale * config.input_scale,
            data_type,
        })
    }

    pub fn encode(
        &self,
        token_ids: &Allocation<B>,
        inner_features: &Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("per layer embedding");

        let total_ple_dim = self.num_layers * self.ple_dim;
        let total_rows = batch_dim * self.num_layers;
        let total_elements = batch_dim * total_ple_dim;

        let mut token_ple = encoder.allocate_scratch(size_for_shape(&[batch_dim, total_ple_dim], self.data_type))?;
        self.token_embedding.encode_lookup(token_ids, &mut token_ple, batch_dim, self.fused_token_scale, encoder);

        let mut model_projection_input =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        encoder.encode_copy(inner_features, .., &mut model_projection_input, ..);
        let model_projected = self.model_projection.encode(model_projection_input, batch_dim, encoder)?;

        let model_normed = self.projection_norm.encode(&model_projected, 0, total_rows, None, encoder)?;

        let mut per_layer_inputs =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.num_layers, self.ple_dim], self.data_type))?;
        self.add_scale.encode(
            Some(&token_ple),
            &model_normed,
            &mut per_layer_inputs,
            total_elements,
            total_elements,
            1.0,
            encoder,
        );

        encoder.pop_debug_group();

        Ok(per_layer_inputs)
    }
}

pub struct PerLayerEmbeddingProjection<B: Backend> {
    gate: Box<dyn Linear<B>>,
    projection: Box<dyn Linear<B>>,
    norm: Normalization<B>,
    gate_act_mul: GatedActMul<B>,
    residual_finalize: <B::Kernels as Kernels>::TensorAddBiasKernel,
    residual_combine: <B::Kernels as Kernels>::TensorAddScaleKernel,
    model_dim: u32,
    ple_dim: u32,
    num_layers: u32,
    activation: AnyActivation,
    post_layer_scalar: f32,
    data_type: DataType,
}

impl<B: Backend> PerLayerEmbeddingProjection<B> {
    pub fn new(
        context: &B::Context,
        config: &PLELayerConfig,
        model_dim: u32,
        num_layers: u32,
        post_layer_scalar: f32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let gate = <dyn Linear<B>>::new(
            model_dim,
            [config.ple_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("gate"),
        )?;
        let projection = <dyn Linear<B>>::new(
            config.ple_dim,
            [model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("projection"),
        )?;
        let norm = Normalization::new(
            model_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            data_type,
            &config.norm_config,
            &parameter_tree.subtree("norm"),
            context,
        )?;

        let activation_alpha = config.activation.alpha();
        let settings = GatedActMulSettings {
            activation_alpha: (activation_alpha != 1.0).then_some(activation_alpha),
            ..Default::default()
        };
        let gate_act_mul = GatedActMul::full_precision(context, data_type, false, false, settings)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_finalize = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, data_type, true)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        let residual_combine = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, true)
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
            activation: config.activation.clone(),
            post_layer_scalar,
            data_type,
        })
    }

    pub fn encode(
        &self,
        layer_index: u32,
        per_layer_input: &Allocation<B>,
        outputs: &mut Allocation<B>,
        hidden: &Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        encoder.push_debug_group("per layer embedding projection");

        let length = batch_dim * self.model_dim;

        self.residual_finalize.encode(None::<&Allocation<B>>, hidden, &mut *outputs, length, length, encoder);

        let mut gate_input = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.model_dim], self.data_type))?;
        encoder.encode_copy(outputs, .., &mut gate_input, ..);
        let gate_out = self.gate.encode(gate_input, batch_dim, encoder)?;

        let mut activated = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.ple_dim], self.data_type))?;
        self.gate_act_mul.encode_fp(
            &gate_out,
            Some(per_layer_input),
            &mut activated,
            None,
            self.ple_dim,
            batch_dim,
            layer_index * self.ple_dim,
            self.num_layers * self.ple_dim,
            self.activation.act_type(),
            encoder,
        );

        let projected = self.projection.encode(activated, batch_dim, encoder)?;
        let normed = self.norm.encode(&projected, 0, batch_dim, None, encoder)?;

        self.residual_combine.encode(
            None::<&Allocation<B>>,
            &normed,
            &mut *outputs,
            length,
            length,
            self.post_layer_scalar,
            encoder,
        );

        encoder.pop_debug_group();

        Ok(())
    }
}
