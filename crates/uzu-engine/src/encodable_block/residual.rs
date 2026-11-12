use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferRef, CommandBuffer, Kernels, kernel::TensorAddScaleKernel},
    config::residual::AnyResidualConfig,
    data_type::DataType,
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum ResidualNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Gated residual connections are not supported yet")]
    GatedUnsupported,
}

pub enum ResidualConnection<B: Backend> {
    Additive(AdditiveResidualConnection<B>),
    // TODO: Gated(GatedResidualConnection<B>)
}

pub struct AdditiveResidualConnection<B: Backend> {
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> AdditiveResidualConnection<B> {
    fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)?,
        })
    }

    fn encode_update(
        &self,
        stream: impl BufferRef<Backend = B>,
        update: impl BufferRef<Backend = B>,
        output: &mut B::ScratchBuffer,
        elements: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.residual_add.encode(Some(stream), update, output, elements, elements, 1.0, command_buffer);
        Ok(())
    }
}

impl<B: Backend> ResidualConnection<B> {
    pub fn new(
        config: &AnyResidualConfig,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        match config {
            AnyResidualConfig::AdditiveResidualConfig(_) => {
                // TODO: Load weights for gated residuals.
                let _ = parameter_tree;
                Ok(Self::Additive(
                    AdditiveResidualConnection::new(context, data_type).map_err(ResidualNewError::Backend)?,
                ))
            },
            AnyResidualConfig::GatedResidualConfig(_) => Err(ResidualNewError::GatedUnsupported),
        }
    }

    pub fn encode_update(
        &self,
        stream: impl BufferRef<Backend = B>,
        update: impl BufferRef<Backend = B>,
        output: &mut B::ScratchBuffer,
        elements: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        match self {
            Self::Additive(connection) => connection.encode_update(stream, update, output, elements, command_buffer),
        }
    }
}

pub struct ResidualHead<B: Backend> {
    connection: ResidualConnection<B>,
}

impl<B: Backend> ResidualHead<B> {
    pub fn new(
        config: &AnyResidualConfig,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        Ok(Self {
            connection: ResidualConnection::new(config, data_type, parameter_tree, context)?,
        })
    }

    // TODO: gated repeats the embeddings `stream_count` times
    pub fn expand(
        &self,
        embeddings: B::ScratchBuffer,
    ) -> B::ScratchBuffer {
        embeddings
    }

    pub fn encode_update(
        &self,
        stream: impl BufferRef<Backend = B>,
        update: impl BufferRef<Backend = B>,
        output: &mut B::ScratchBuffer,
        elements: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.connection.encode_update(stream, update, output, elements, command_buffer)
    }
}
