use std::marker::PhantomData;

use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, kernel::TensorAddScaleKernel,
    },
    config::residual::AnyResidualConfig,
    data_type::DataType,
    safetensors::ParameterTree,
};

#[derive(Debug, Error)]
pub enum ResidualNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
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
            residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false, true)?,
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
        self.residual_add.encode(Some(stream), Some(update), output, elements, elements, 1.0, command_buffer);
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
        Ok(match config {
            AnyResidualConfig::AdditiveResidualConfig(_) => {
                Self::Additive(AdditiveResidualConnection::new(context, data_type).map_err(ResidualNewError::Backend)?)
            },
            AnyResidualConfig::GatedResidualConfig(_) => {
                // TODO: Load weights for gated residuals.
                let _ = parameter_tree;
                todo!("Implement gated residual support.")
            },
        })
    }

    pub fn encode_update(
        &self,
        stream: impl BufferRef<Backend = B>,
        update: impl BufferRef<Backend = B>,
        elements: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch(stream.size())?;
        match self {
            Self::Additive(connection) => {
                connection.encode_update(stream, update, &mut output, elements, command_buffer)?
            },
        };

        Ok(output)
    }
}

pub struct ResidualHead<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> ResidualHead<B> {
    pub fn new(
        config: &AnyResidualConfig,
        data_type: DataType,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ResidualNewError<B>> {
        match config {
            AnyResidualConfig::AdditiveResidualConfig(_) => Ok(Self {
                backend: PhantomData,
            }),
            AnyResidualConfig::GatedResidualConfig(_) => {
                // TODO: Load weights for gated residuals.
                let _ = (data_type, parameter_tree, context);
                todo!("Implement gated residual support.")
            },
        }
    }

    // TODO: gated repeats the embeddings `stream_count` times
    pub fn expand(
        &self,
        embeddings: B::ScratchBuffer,
    ) -> B::ScratchBuffer {
        embeddings
    }

    // TODO: gated mixes the `stream_count` streams down to `model_dim`
    pub fn readout(
        &self,
        stream: B::ScratchBuffer,
    ) -> B::ScratchBuffer {
        stream
    }
}
