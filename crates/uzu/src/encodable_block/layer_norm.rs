//! LayerNorm encodable.

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{Kernels, LayerNormKernel},
    },
    config::{NormalizationConfig, UpcastMode},
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LayerNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
}

pub struct LayerNorm<B: Backend> {
    kernel: <B::Kernels as Kernels>::LayerNormKernel,
    config: NormalizationConfig,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    scales_buffer: B::NativeBuffer,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, LayerNormError<B>> {
        let scales = parameter_tree.leaf("scales").map_err(LayerNormError::ParameterError)?;
        let scales_buffer = scales.buffer().clone();

        let accumulation_data_type: DataType = config.accumulation_precision.into();
        let scale_data_type: DataType = config.scale_precision.into();

        let kernel = <B::Kernels as Kernels>::LayerNormKernel::new(
            context,
            intermediate_data_type,
            scale_data_type,
            scale_data_type,
            accumulation_data_type,
        )
        .map_err(LayerNormError::BackendError)?;

        Ok(Self {
            kernel,
            config,
            input_array_id,
            output_array_id,
            scales_buffer,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for LayerNorm<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let input_binding = state.arrays(&[self.input_array_id]);
        let output_binding = state.arrays(&[self.output_array_id]);

        let input_shape = {
            let input_array = input_binding[0].borrow();
            input_array.shape().to_vec()
        };

        let input_array = input_binding[0].borrow_mut();
        let output_array = output_binding[0].borrow_mut();

        let batch_size = input_shape[0] as u32;
        let model_dim = input_shape[1] as u32;
        let full_layer = if self.config.upcast_mode == UpcastMode::FullLayer {
            1u32
        } else {
            0u32
        };

        self.kernel.encode(
            input_array.buffer(),
            &self.scales_buffer,
            output_array.buffer(),
            batch_size,
            model_dim,
            self.config.epsilon,
            self.config.scale_offset.unwrap_or(0.0),
            full_layer,
            encoder,
        );
    }
}
