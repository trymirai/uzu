use thiserror::Error;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{
            Kernels, TensorAddBiasKernel,
            matmul::{
                MatmulKernels, QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernel,
                QuantizedMatmulType,
            },
        },
    },
    config::QuantizationConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QuantizedLinearError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError),
    #[error("Unsupported data type for quantized kernel: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Expected weights of type {expected:?}, got {got:?}")]
    InvalidWeightsDataType {
        expected: DataType,
        got: DataType,
    },
    #[error("Scales dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidScalesDataType {
        expected: DataType,
        got: DataType,
    },
    #[error(
        "Unexpected MLX shapes. weights={weights:?}, scales={scales:?}, deq_biases={deq_biases:?}; expected [N,K/{packing_divisor}],[N,K_g],[N,K_g]"
    )]
    InvalidMlxShapes {
        weights: Box<[usize]>,
        scales: Box<[usize]>,
        deq_biases: Box<[usize]>,
        packing_divisor: usize,
    },
    #[error("deq_biases dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidDeqBiasesDataType {
        expected: DataType,
        got: DataType,
    },
    #[error(
        "Unexpected AWQ shapes. weights={weights:?}, scales={scales:?}, zero_points={zero_points:?}; expected [N,K/{packing_divisor}],[N,K_g],[N,(K_g+{packing_minus_one})/{packing_divisor}]"
    )]
    InvalidAwqShapes {
        weights: Box<[usize]>,
        scales: Box<[usize]>,
        zero_points: Box<[usize]>,
        packing_divisor: usize,
        packing_minus_one: usize,
    },
    #[error("Zero-points dtype mismatch: got {got:?}, expected {expected:?}")]
    InvalidZeroPointsDataType {
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

pub struct QuantizedLinear<B: Backend>
where
    B::Kernels: MatmulKernels,
{
    kernel: <B::Kernels as MatmulKernels>::QuantizedMatmulKernel,
    bias_add_kernel: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
    biases_buffer: Option<B::NativeBuffer>,
    weights_buffer: B::NativeBuffer,
    scales_buffer: B::NativeBuffer,
    zero_points_or_biases_buffer: B::NativeBuffer,
    quantization_type: QuantizedMatmulType,
    input_dim: usize,
    output_dim: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
}

impl<B: Backend> QuantizedLinear<B>
where
    B::Kernels: MatmulKernels,
{
    pub fn new(
        context: &B::Context,
        config: &QuantizationConfig,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, QuantizedLinearError<B>> {
        let kernel_data_type: DataType = config.activation_precision.into();
        if !matches!(kernel_data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedLinearError::UnsupportedDataType(kernel_data_type));
        }

        let weights = parameter_tree.leaf("weights").map_err(QuantizedLinearError::ParameterError)?;
        let packing_divisor = config.weight_quantization_mode.packing_divisor();
        let storage_type = config.weight_quantization_mode.storage_type();
        if weights.data_type() != storage_type {
            return Err(QuantizedLinearError::InvalidWeightsDataType {
                expected: storage_type,
                got: weights.data_type(),
            });
        }

        let scales = parameter_tree.leaf("scales").map_err(QuantizedLinearError::ParameterError)?;
        if scales.data_type() != kernel_data_type {
            return Err(QuantizedLinearError::InvalidScalesDataType {
                expected: kernel_data_type,
                got: scales.data_type(),
            });
        }

        let k_g = (input_dim + config.group_size - 1) / config.group_size;
        let weights_shape = weights.shape().to_vec();
        let scales_shape = scales.shape().to_vec();
        let weights_transposed = weights_shape[0] == output_dim;

        let (quantization_type, zero_points_or_biases_buffer) = match parameter_tree.leaf("deq_biases") {
            Ok(deq_biases) => {
                let deq_biases_shape = deq_biases.shape().to_vec();
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && deq_biases_shape == [output_dim, k_g])
                {
                    return Err(QuantizedLinearError::InvalidMlxShapes {
                        weights: weights_shape.clone().into_boxed_slice(),
                        scales: scales_shape.clone().into_boxed_slice(),
                        deq_biases: deq_biases_shape.into_boxed_slice(),
                        packing_divisor,
                    });
                }

                if deq_biases.data_type() != kernel_data_type {
                    return Err(QuantizedLinearError::InvalidDeqBiasesDataType {
                        expected: kernel_data_type,
                        got: deq_biases.data_type(),
                    });
                }

                (QuantizedMatmulType::Mlx, deq_biases.buffer().clone())
            },
            Err(_) => {
                let zero_points = parameter_tree.leaf("zero_points").map_err(QuantizedLinearError::ParameterError)?;
                let zero_points_shape = zero_points.shape().to_vec();
                let expected_zero_points_entries = (k_g + packing_divisor - 1) / packing_divisor;
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && zero_points_shape == [output_dim, expected_zero_points_entries])
                {
                    return Err(QuantizedLinearError::InvalidAwqShapes {
                        weights: weights_shape.clone().into_boxed_slice(),
                        scales: scales_shape.clone().into_boxed_slice(),
                        zero_points: zero_points_shape.into_boxed_slice(),
                        packing_divisor,
                        packing_minus_one: packing_divisor - 1,
                    });
                }

                if zero_points.data_type() != storage_type {
                    return Err(QuantizedLinearError::InvalidZeroPointsDataType {
                        expected: storage_type,
                        got: zero_points.data_type(),
                    });
                }

                (QuantizedMatmulType::ZeroPoint, zero_points.buffer().clone())
            },
        };

        let (bias_add_kernel, biases_buffer) = match parameter_tree.leaf("biases") {
            Ok(biases) => {
                let bias_shape = biases.shape().to_vec();
                if bias_shape != [output_dim] {
                    return Err(QuantizedLinearError::InvalidBiasShape {
                        got: bias_shape.into_boxed_slice(),
                        expected_output_dim: output_dim,
                    });
                }

                if biases.data_type() != kernel_data_type {
                    return Err(QuantizedLinearError::InvalidBiasDataType {
                        expected: kernel_data_type,
                        got: biases.data_type(),
                    });
                }

                let bias_add_kernel = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, kernel_data_type)
                    .map_err(QuantizedLinearError::BackendError)?;
                (Some(bias_add_kernel), Some(biases.buffer().clone()))
            },
            Err(_) => (None, None),
        };

        let kernel = <B::Kernels as MatmulKernels>::QuantizedMatmulKernel::new(
            context,
            QuantizedMatmulConfiguration {
                data_type: kernel_data_type,
                group_size: config.group_size,
                input_dim,
                output_dim,
                mode: config.weight_quantization_mode,
                quantization_type,
                weights_transposed,
            },
        )
        .map_err(QuantizedLinearError::BackendError)?;

        Ok(Self {
            kernel,
            bias_add_kernel,
            biases_buffer,
            weights_buffer: weights.buffer().clone(),
            scales_buffer: scales.buffer().clone(),
            zero_points_or_biases_buffer,
            quantization_type,
            input_dim,
            output_dim,
            input_array_id,
            output_array_id,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for QuantizedLinear<B>
where
    B::Kernels: MatmulKernels,
{
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters<B>,
        encoder: &B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[self.input_array_id, self.output_array_id]);
        let batch_size = state.active_suffix_length();
        let input_array = arrays[0].borrow_mut();
        let output_array = arrays[1].borrow_mut();
        let output_buffer = output_array.buffer();

        self.kernel.encode(
            encoder,
            QuantizedMatmulArguments {
                a_buffer: input_array.buffer(),
                a_offset: 0,
                b_buffer: &self.weights_buffer,
                scales_buffer: &self.scales_buffer,
                zero_points_or_biases_buffer: &self.zero_points_or_biases_buffer,
                output_buffer,
                batch: batch_size,
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                quantization_type: self.quantization_type,
            },
        );

        if let (Some(bias_add_kernel), Some(biases_buffer)) = (&self.bias_add_kernel, &self.biases_buffer) {
            let total_length = batch_size * self.output_dim;
            bias_add_kernel.encode_if(
                output_buffer,
                biases_buffer,
                output_buffer,
                self.output_dim as u32,
                total_length as u32,
                encoder,
                parameters.predicate_ref(),
            );
        }
    }
}
