use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding,
        kernel::{Kernels, SeparableCausalConvKernel},
    },
    config::token_mixer::convolutions::SeparableCausalConvConfig,
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
    utils::size_for_shape::size_for_shape,
};

#[derive(Debug, Error)]
pub enum ConvolutionNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    Parameter(#[from] ParameterLoaderError<B>),
}

pub struct SeparableCausalConv<B: Backend> {
    model_dim: u32,
    data_type: DataType,
    weights: B::GlobalGpuBuffer,
    biases: Option<B::GlobalGpuBuffer>,
    kernel: <B::Kernels as Kernels>::SeparableCausalConvKernel,
}

impl<B: Backend> SeparableCausalConv<B> {
    pub async fn new(
        model_dim: u32,
        kernel_size: u32,
        group_size: u32,
        data_type: DataType,
        config: &SeparableCausalConvConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ConvolutionNewError<B>> {
        assert!(model_dim.is_multiple_of(4));
        assert!(group_size.is_multiple_of(4));
        assert!(model_dim.is_multiple_of(group_size));
        assert_eq!(data_type, DataType::BF16, "SeparableCausalConv only supports BF16");

        let weights = parameter_tree.read_buffer("weights", &[model_dim, kernel_size], DataType::BF16).await?;

        let biases = if config.has_biases {
            Some(parameter_tree.read_buffer("biases", &[model_dim], DataType::BF16).await?)
        } else {
            None
        };

        let kernel = <B::Kernels as Kernels>::SeparableCausalConvKernel::new(
            context,
            data_type,
            model_dim,
            kernel_size,
            group_size,
            config.has_biases,
        )
        .map_err(ConvolutionNewError::Backend)?;

        Ok(Self {
            model_dim,
            data_type,
            weights,
            biases,
            kernel,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        coefficient_deltas: impl BufferRef<Backend = B>,
        coefficient_row_stride: u32,
        coefficient_column_offset: u32,
        sequence_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output =
            command_buffer.allocate_scratch_for_shape(&[sequence_length, self.model_dim], self.data_type)?;
        let coefficients_offset_bytes = size_for_shape(&[coefficient_column_offset], self.data_type);
        self.kernel.encode(
            input,
            coefficient_deltas.subrange(coefficients_offset_bytes..),
            &self.weights,
            self.biases.as_ref(),
            &mut output,
            sequence_length,
            coefficient_row_stride,
            command_buffer,
        );

        Ok(output)
    }
}
