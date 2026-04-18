//! RMS Normalization encodable.

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, RMSNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RMSNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
}

pub struct RMSNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::RMSNormKernel,
    config: NormalizationConfig,
    scales: Allocation<B>,
    element_count: usize,
    input_data_type: DataType,
    output_data_type: DataType,
    hadamard_factors: Option<Allocation<B>>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        parameter_tree: &ParameterTree<B::Context>,
        hadamard_factors: Option<Allocation<B>>,
        use_shortcut: bool,
        residual_add: bool,
    ) -> Result<Self, RMSNormError<B>> {
        let scales_leaf = parameter_tree.leaf("scales").map_err(RMSNormError::ParameterError)?;
        let element_count = scales_leaf.shape()[0];
        let scales = scales_leaf.read_allocation().map_err(RMSNormError::ParameterError)?;

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let (input_type, scales_type, output_type) = match config.upcast_mode {
            UpcastMode::OnlyNormalization => (intermediate_data_type, scale_data_type, scale_data_type),
            UpcastMode::FullLayer => (intermediate_data_type, scale_data_type, scale_data_type),
        };

        let kernel = <B::Kernels as Kernels>::RMSNormKernel::new(
            context,
            input_type,
            scales_type,
            output_type,
            accumulation_data_type,
            false,
            config.upcast_mode == UpcastMode::FullLayer,
            use_shortcut,
            residual_add,
            hadamard_factors.is_some(),
        )
        .map_err(RMSNormError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            scales,
            element_count,
            input_data_type: input_type,
            output_data_type: output_type,
            hadamard_factors,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        row_offset: usize,
        row_count: usize,
        shortcut: Option<&mut Allocation<B>>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let row_size = self.element_count * self.input_data_type.size_in_bytes();
        let input = input.view_at_offset(row_offset * row_size, row_count * row_size);
        let mut shortcut =
            shortcut.map(|shortcut| shortcut.view_at_offset(row_offset * row_size, row_count * row_size));
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[row_count, self.element_count], self.output_data_type))?;
        self.kernel.encode(
            Some(&input),
            &self.scales,
            &mut output,
            shortcut.as_mut(),
            self.hadamard_factors.as_ref(),
            row_count as u32,
            self.element_count as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            encoder,
        );
        Ok(output)
    }
}
