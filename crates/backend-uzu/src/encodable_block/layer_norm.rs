//! LayerNorm encodable.

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, LayerNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LayerNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
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
        config: NormalizationConfig,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, LayerNormError<B>> {
        let scales_leaf = parameter_tree.leaf("scales").map_err(LayerNormError::ParameterError)?;
        let element_count = scales_leaf.shape()[0];
        let scales = scales_leaf.read_allocation().map_err(LayerNormError::ParameterError)?;

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();
        let full_layer = config.upcast_mode == UpcastMode::FullLayer;

        let kernel = <B::Kernels as Kernels>::LayerNormKernel::new(
            context,
            intermediate_data_type,
            scale_data_type,
            scale_data_type,
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
            output_data_type: if full_layer {
                scale_data_type
            } else {
                scale_data_type
            },
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
        let input = input.view_at_offset(row_offset * row_size, row_count * row_size);
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[row_count, self.element_count], self.output_data_type))?;
        self.kernel.encode(
            Some(&input),
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
