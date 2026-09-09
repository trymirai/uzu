use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, SeparableCausalConvKernel},
    },
    config::convolutions::SeparableCausalConvConfig,
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
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
    coefficient_row_stride: u32,
    data_type: DataType,
    weights: Allocation<B>,
    biases: Option<Allocation<B>>,
    kernel: <B::Kernels as Kernels>::SeparableCausalConvKernel,
}

impl<B: Backend> SeparableCausalConv<B> {
    pub fn new(
        model_dim: u32,
        kernel_size: u32,
        group_size: u32,
        data_type: DataType,
        config: &SeparableCausalConvConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, ConvolutionNewError<B>> {
        assert!(model_dim.is_multiple_of(4));
        assert!(group_size.is_multiple_of(4));
        assert!(model_dim.is_multiple_of(group_size));
        assert_eq!(data_type, DataType::BF16, "SeparableCausalConv only supports BF16");

        let weights =
            parameter_tree.leaf("weights")?.validate(&[model_dim, kernel_size], DataType::BF16)?.read_allocation()?;

        let biases = config
            .has_biases
            .then(|| parameter_tree.leaf("biases")?.validate(&[model_dim], DataType::BF16)?.read_allocation())
            .transpose()?;

        let kernel = <B::Kernels as Kernels>::SeparableCausalConvKernel::new(
            context,
            data_type,
            model_dim,
            kernel_size,
            group_size,
            config.has_biases,
        )
        .map_err(ConvolutionNewError::Backend)?;

        let coefficient_row_stride = 2 * kernel_size * (model_dim / group_size);

        Ok(Self {
            model_dim,
            coefficient_row_stride,
            data_type,
            weights,
            biases,
            kernel,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        coefficient_deltas: &Allocation<B>,
        coefficient_column_offset: u32,
        sequence_length: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("SeparableCausalConv");

        let mut output = encoder.allocate_scratch_for_shape(&[sequence_length, self.model_dim], self.data_type)?;

        let coefficients_offset_bytes = size_for_shape(&[coefficient_column_offset], self.data_type);
        self.kernel.encode(
            input,
            (coefficient_deltas, coefficients_offset_bytes),
            &self.weights,
            self.biases.as_ref(),
            &mut output,
            sequence_length,
            self.coefficient_row_stride,
            encoder,
        );

        encoder.pop_debug_group();
        Ok(output)
    }
}
