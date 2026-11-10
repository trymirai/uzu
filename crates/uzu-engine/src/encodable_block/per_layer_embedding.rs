use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{Kernels, TensorAddScaleKernel},
    },
    config::per_layer_embedding::PLEModelConfig,
    data_type::DataType,
    encodable_block::{
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum PerLayerEmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Linear error: {0}")]
    LinearError(#[from] LinearBlockError<B>),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
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
    pub async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &PLEModelConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, PerLayerEmbeddingError<B>> {
        let total_ple_dim = config.num_layers * config.ple_dim;

        let token_embedding = EmbeddingTable::load(
            config.ple_vocab_size,
            total_ple_dim,
            data_type,
            &parameter_tree.subtree("token_embedding"),
            context,
        )
        .await?;

        let model_projection = <dyn Linear<B>>::new(
            model_dim,
            total_ple_dim,
            data_type,
            false,
            &parameter_tree.subtree("model_projection"),
            context,
        )
        .await?;

        let scale_squared = config.model_projection_scale * config.model_projection_scale;
        let projection_norm_config = {
            let mut adjusted = config.norm_config.clone();
            adjusted.epsilon /= scale_squared;
            adjusted
        };
        let projection_norm = Normalization::new(
            config.ple_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            PostLayerScalar::ScaleOutput(config.input_scale),
            &projection_norm_config,
            &parameter_tree.subtree("projection_norm"),
            context,
        )
        .await?;

        let add_scale = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false, true)
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
        token_ids: impl BufferRef<Backend = B>,
        inner_features: impl BufferRef<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, PerLayerEmbeddingError<B>> {
        let command_buffer = &mut *command_buffer.debug_group("per layer embedding");

        let total_ple_dim = self.num_layers * self.ple_dim;
        let total_rows = batch_dim * self.num_layers;
        let total_elements = batch_dim * total_ple_dim;

        let mut token_ple = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, total_ple_dim], self.data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        self.token_embedding.encode_lookup(
            token_ids,
            &mut token_ple,
            batch_dim,
            self.fused_token_scale,
            command_buffer,
        );

        let mut model_projection_input = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.model_dim], self.data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        command_buffer.encode_copy(inner_features, &mut model_projection_input);
        let model_projected = self.model_projection.encode(model_projection_input, batch_dim, command_buffer)?;

        let model_normed = self
            .projection_norm
            .encode(&model_projected, 0, total_rows, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(PerLayerEmbeddingError::BackendError)?;

        let mut per_layer_inputs = command_buffer
            .allocate_scratch_for_shape(&[batch_dim, self.num_layers, self.ple_dim], self.data_type)
            .map_err(PerLayerEmbeddingError::BackendError)?;
        self.add_scale.encode(
            Some(&token_ple),
            Some(&model_normed),
            &mut per_layer_inputs,
            total_elements,
            total_elements,
            1.0,
            command_buffer,
        );

        Ok(per_layer_inputs)
    }
}
