use std::cell::RefCell;

use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::matmul::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel, MatmulKernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum FullPrecisionLinearError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error("Unsupported data type for full precision linear kernel: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unexpected weights shape: got {got:?}, expected [{expected_output_dim}, {expected_input_dim}]")]
    InvalidWeightsShape {
        got: Box<[usize]>,
        expected_output_dim: usize,
        expected_input_dim: usize,
    },
    #[error("Weights dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidWeightsDataType {
        expected: DataType,
        got: DataType,
    },
    #[error("Bias shape mismatch: got {got:?}, expected [{expected_output_dim}]")]
    InvalidBiasShape {
        got: Box<[usize]>,
        expected_output_dim: usize,
    },
    #[error("Bias dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidBiasDataType {
        expected: DataType,
        got: DataType,
    },
}

pub struct FullPrecisionLinear<B: Backend>
where
    B::Kernels: MatmulKernels,
{
    kernel: RefCell<<B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel>,
    bias_buffer: Option<B::NativeBuffer>,
    weights_buffer: B::NativeBuffer,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl<B: Backend> FullPrecisionLinear<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, FullPrecisionLinearError<B>> {
        if !matches!(precision, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(FullPrecisionLinearError::UnsupportedDataType(precision));
        }

        let weights = parameter_tree.leaf("weights").map_err(FullPrecisionLinearError::ParameterError)?;
        let weights_shape = weights.shape().to_vec();
        if weights_shape != [output_dim, input_dim] {
            return Err(FullPrecisionLinearError::InvalidWeightsShape {
                got: weights_shape.into_boxed_slice(),
                expected_output_dim: output_dim,
                expected_input_dim: input_dim,
            });
        }

        if weights.data_type() != precision {
            return Err(FullPrecisionLinearError::InvalidWeightsDataType {
                expected: precision,
                got: weights.data_type(),
            });
        }

        let bias_buffer = match parameter_tree.leaf("biases") {
            Ok(biases) => {
                let bias_shape = biases.shape().to_vec();
                if bias_shape != [output_dim] {
                    return Err(FullPrecisionLinearError::InvalidBiasShape {
                        got: bias_shape.into_boxed_slice(),
                        expected_output_dim: output_dim,
                    });
                }

                if biases.data_type() != precision {
                    return Err(FullPrecisionLinearError::InvalidBiasDataType {
                        expected: precision,
                        got: biases.data_type(),
                    });
                }

                Some(biases.buffer().clone())
            },
            Err(_) => None,
        };

        let kernel = <B::Kernels as MatmulKernels>::FullPrecisionMatmulKernel::new(context, precision)
            .map_err(FullPrecisionLinearError::BackendError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias_buffer,
            weights_buffer: weights.buffer().clone(),
            input_dim,
            output_dim,
            input_array_id,
            output_array_id,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for FullPrecisionLinear<B>
where
    B::Kernels: MatmulKernels,
{
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();
        let input_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();

        self.kernel.borrow_mut().encode(
            state.mtl_context(),
            encoder,
            FullPrecisionMatmulArguments {
                a: input_array.buffer(),
                a_offset: 0,
                b: &self.weights_buffer,
                output: output_array.buffer(),
                bias: self.bias_buffer.as_ref(),
                batch: batch_size,
                input_dim: self.input_dim,
                output_dim: self.output_dim,
            },
        );
    }
}
