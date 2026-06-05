//! LayerNorm encodable.

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, LayerNormKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LayerNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
}

pub struct LayerNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::LayerNormKernel,
    config: NormalizationConfig,
    scales: Allocation<B>,
    element_count: usize,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        element_count: usize,
        config: NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, LayerNormError<B>> {
        let scale_data_type = super::normalization::NORMALIZATION_SCALE_DATA_TYPE;
        let scales = parameter_tree.leaf("scales")?.validate(&[element_count], scale_data_type)?.read_allocation()?;

        let accumulation_data_type = super::normalization::NORMALIZATION_ACCUMULATION_DATA_TYPE;
        let output_data_type = intermediate_data_type;
        let kernel = <B::Kernels as Kernels>::LayerNormKernel::new(
            context,
            intermediate_data_type,
            scale_data_type,
            output_data_type,
            accumulation_data_type,
            false,
        )
        .map_err(LayerNormError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            scales,
            element_count,
            input_data_type: intermediate_data_type,
            output_data_type,
        })
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        row_offset: usize,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let full_layer = if self.config.upcast_mode == UpcastMode::FullLayer {
            1u32
        } else {
            0u32
        };
        let row_size = self.element_count * self.input_data_type.size_in_bytes();
        let input_offset = row_offset * row_size;
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[row_count, self.element_count], self.output_data_type))?;
        self.kernel.encode(
            Some((input, input_offset)),
            &self.scales,
            &mut output,
            row_count as u32,
            self.element_count as u32,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            full_layer,
            encoder,
        );
        Ok(output)
    }
}
