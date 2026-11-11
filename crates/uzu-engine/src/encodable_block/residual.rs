use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        kernel::{
            GatedResidualExpandKernel, GatedResidualLowRankActivationKernel, GatedResidualMixKernel,
            GatedResidualUpdateKernel, TensorAddScaleKernel,
        },
    },
    config::residual::{AnyResidualConfig, gated::GatedResidualConfig},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError, LinearEncodeError},
        normalization::{Normalization, NormalizationNewError, ShortcutMode},
    },
    safetensors::ParameterTree,
    utils::maybe_owned::MaybeOwned,
};

#[derive(Debug, Error)]
pub enum ResidualNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

#[derive(Debug, Error)]
pub enum ResidualEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Linear encoding error: {0}")]
    LinearEncode(#[from] LinearEncodeError<B>),
}

struct GatedStreamReader<B: Backend> {
    model_dim: u32,
    stream_count: u32,
    low_rank_dim: u32,
    norm: Normalization<B>,
    down_projection: Box<dyn Linear<B>>,
    up_projection: Box<dyn Linear<B>>,
    low_rank_activation: <B::Kernels as Kernels>::GatedResidualLowRankActivationKernel,
    mix: <B::Kernels as Kernels>::GatedResidualMixKernel,
}

impl<B: Backend> GatedStreamReader<B> {
    async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &GatedResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        let stream_dim = config.stream_count * model_dim;
        let norm = Normalization::new_mixture(
            Some(config.stream_count),
            model_dim,
            data_type,
            data_type,
            None,
            ShortcutMode::None,
            &config.norm_config,
            &parameter_tree.subtree("norm"),
            context,
        )
        .await?;
        let down_projection = <dyn Linear<B>>::new(
            stream_dim,
            config.low_rank_dim,
            data_type,
            false,
            &parameter_tree.subtree("down_projection"),
            context,
        )
        .await?;
        let up_projection = <dyn Linear<B>>::new(
            config.low_rank_dim,
            stream_dim,
            data_type,
            false,
            &parameter_tree.subtree("up_projection"),
            context,
        )
        .await?;

        Ok(Self {
            model_dim,
            stream_count: config.stream_count,
            low_rank_dim: config.low_rank_dim,
            norm,
            down_projection,
            up_projection,
            low_rank_activation: <B::Kernels as Kernels>::GatedResidualLowRankActivationKernel::new(context, data_type)
                .map_err(ResidualNewError::Backend)?,
            mix: <B::Kernels as Kernels>::GatedResidualMixKernel::new(context, data_type)
                .map_err(ResidualNewError::Backend)?,
        })
    }

    /// Returns the normalized streams (`[row_count, stream_count, model_dim]`) and their mix (`[row_count, model_dim]`).
    fn encode(
        &self,
        stream: impl BufferRef<Backend = B>,
        row_offset: u32,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), ResidualEncodeError<B>> {
        let normalized = self
            .norm
            .encode(
                stream,
                row_offset * self.stream_count,
                row_count * self.stream_count,
                None::<&mut B::ScratchBuffer>,
                command_buffer,
            )
            .map_err(ResidualEncodeError::Backend)?;

        // TODO: We could avoid this copy if we make a method on `Linear` that
        // doesn't modify the input.
        let mut down_input =
            command_buffer.allocate_scratch(normalized.size()).map_err(ResidualEncodeError::Backend)?;
        command_buffer.encode_copy(&normalized, &mut down_input);
        let mut low_rank = self.down_projection.encode(down_input, row_count, command_buffer)?;

        self.low_rank_activation.encode(
            &mut low_rank,
            self.stream_count,
            row_count * self.low_rank_dim,
            command_buffer,
        );
        let mix_logits = self.up_projection.encode(low_rank, row_count, command_buffer)?;

        let mut mixed = command_buffer
            .allocate_scratch(normalized.size() / self.stream_count as usize)
            .map_err(ResidualEncodeError::Backend)?;
        self.mix.encode(
            &normalized,
            &mix_logits,
            &mut mixed,
            self.model_dim,
            self.stream_count,
            row_count * self.model_dim,
            command_buffer,
        );

        Ok((normalized, mixed))
    }
}

pub enum ResidualConnection<B: Backend> {
    Additive(AdditiveResidualConnection<B>),
    Gated(GatedResidualConnection<B>),
}

pub struct AdditiveResidualConnection<B: Backend> {
    model_dim: u32,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> AdditiveResidualConnection<B> {
    fn new(
        model_dim: u32,
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            model_dim,
            residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false, true)?,
        })
    }

    fn encode<E: From<ResidualEncodeError<B>>>(
        &self,
        stream: &B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            MaybeOwned<'_, B::ScratchBuffer>,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, E>,
    ) -> Result<B::ScratchBuffer, E> {
        let update = body(MaybeOwned::Borrowed(stream), command_buffer)?;

        let mut output = command_buffer.allocate_scratch(stream.size()).map_err(ResidualEncodeError::Backend)?;
        let elements = row_count * self.model_dim;
        self.residual_add.encode(Some(stream), Some(&update), &mut output, elements, elements, 1.0, command_buffer);
        Ok(output)
    }
}

pub struct GatedResidualConnection<B: Backend> {
    reader: GatedStreamReader<B>,
    injection: Box<dyn Linear<B>>,
    update: <B::Kernels as Kernels>::GatedResidualUpdateKernel,
}

impl<B: Backend> GatedResidualConnection<B> {
    async fn new(
        model_dim: u32,
        data_type: DataType,
        config: &GatedResidualConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        let reader =
            GatedStreamReader::new(model_dim, data_type, config, &parameter_tree.subtree("reader"), context).await?;
        let injection = <dyn Linear<B>>::new(
            config.stream_count * model_dim,
            config.stream_count,
            data_type,
            false,
            &parameter_tree.subtree("injection"),
            context,
        )
        .await?;
        let update = <B::Kernels as Kernels>::GatedResidualUpdateKernel::new(context, data_type)
            .map_err(ResidualNewError::Backend)?;

        Ok(Self {
            reader,
            injection,
            update,
        })
    }

    fn encode<E: From<ResidualEncodeError<B>>>(
        &self,
        stream: &B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        sublayer_body: impl FnOnce(
            MaybeOwned<'_, B::ScratchBuffer>,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, E>,
    ) -> Result<B::ScratchBuffer, E> {
        let (normalized, mixed) = self.reader.encode(stream, 0, row_count, command_buffer)?;
        let injection_logits =
            self.injection.encode(normalized, row_count, command_buffer).map_err(ResidualEncodeError::LinearEncode)?;

        let update = sublayer_body(MaybeOwned::Owned(mixed), command_buffer)?;

        let mut output = command_buffer.allocate_scratch(stream.size()).map_err(ResidualEncodeError::Backend)?;
        let model_dim = self.reader.model_dim;
        let stream_count = self.reader.stream_count;
        self.update.encode(
            stream,
            &injection_logits,
            &update,
            &mut output,
            model_dim,
            stream_count,
            row_count * stream_count * model_dim,
            command_buffer,
        );
        Ok(output)
    }
}

impl<B: Backend> ResidualConnection<B> {
    pub async fn new(
        config: &AnyResidualConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        Ok(match config {
            AnyResidualConfig::AdditiveResidualConfig(_) => Self::Additive(
                AdditiveResidualConnection::new(model_dim, context, data_type).map_err(ResidualNewError::Backend)?,
            ),
            AnyResidualConfig::GatedResidualConfig(config) => {
                Self::Gated(GatedResidualConnection::new(model_dim, data_type, config, parameter_tree, context).await?)
            },
        })
    }

    pub fn encode<E: From<ResidualEncodeError<B>>>(
        &self,
        stream: &B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        body: impl FnOnce(
            MaybeOwned<'_, B::ScratchBuffer>,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ) -> Result<B::ScratchBuffer, E>,
    ) -> Result<B::ScratchBuffer, E> {
        match self {
            Self::Additive(connection) => connection.encode(stream, row_count, command_buffer, body),
            Self::Gated(connection) => connection.encode(stream, row_count, command_buffer, body),
        }
    }
}

pub enum ResidualHead<B: Backend> {
    Additive,
    Gated(GatedResidualHead<B>),
}

pub struct GatedResidualHead<B: Backend> {
    reader: GatedStreamReader<B>,
    expand: <B::Kernels as Kernels>::GatedResidualExpandKernel,
}

impl<B: Backend> ResidualHead<B> {
    pub async fn new(
        config: &AnyResidualConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        match config {
            AnyResidualConfig::AdditiveResidualConfig(_) => Ok(Self::Additive),
            AnyResidualConfig::GatedResidualConfig(config) => Ok(Self::Gated(GatedResidualHead {
                reader: GatedStreamReader::new(
                    model_dim,
                    data_type,
                    config,
                    &parameter_tree.subtree("reader"),
                    context,
                )
                .await?,
                expand: <B::Kernels as Kernels>::GatedResidualExpandKernel::new(context, data_type)
                    .map_err(ResidualNewError::Backend)?,
            })),
        }
    }

    pub fn expand(
        &self,
        embeddings: B::ScratchBuffer,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        match self {
            Self::Additive => Ok(embeddings),
            Self::Gated(head) => {
                let model_dim = head.reader.model_dim;
                let stream_count = head.reader.stream_count;
                let mut stream = command_buffer.allocate_scratch(embeddings.size() * stream_count as usize)?;
                head.expand.encode(
                    &embeddings,
                    &mut stream,
                    model_dim,
                    stream_count,
                    row_count * stream_count * model_dim,
                    command_buffer,
                );
                Ok(stream)
            },
        }
    }

    pub fn readout(
        &self,
        stream: impl BufferRef<Backend = B>,
        row_offset: u32,
        row_count: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<B::ScratchBuffer>, ResidualEncodeError<B>> {
        match self {
            Self::Additive => Ok(None),
            Self::Gated(head) => {
                let (_, mixed) = head.reader.encode(stream, row_offset, row_count, command_buffer)?;
                Ok(Some(mixed))
            },
        }
    }
}
