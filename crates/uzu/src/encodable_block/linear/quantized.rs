use thiserror::Error;

use super::Linear;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels, TensorAddBiasKernel,
            quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
                QuantizedMatmulKernelEncodable, QuantizedMatmulType,
            },
        },
    },
    config::QuantizationConfig,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QuantizedLinearError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("QuantizedMatmul error: {0}")]
    QuantizedMatmulError(#[source] QuantizedMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
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

pub struct QuantizedLinear<B: Backend> {
    kernel: QuantizedMatmulKernelEncodable<B>,
    bias_add_kernel: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
    biases: Option<Allocation<B>>,
    weights: Allocation<B>,
    scales: Allocation<B>,
    zero_points_or_biases: Allocation<B>,
    output_hadamard_factors: Option<Allocation<B>>,
    output_dim: usize,
    output_data_type: DataType,
}

impl<B: Backend> QuantizedLinear<B> {
    pub fn new(
        context: &B::Context,
        config: &QuantizationConfig,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, QuantizedLinearError<B>> {
        let kernel_data_type: DataType = config.activation_precision.into();
        if !matches!(kernel_data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedLinearError::UnsupportedDataType(kernel_data_type));
        }

        let weights_leaf = parameter_tree.leaf("weights").map_err(QuantizedLinearError::ParameterError)?;
        let packing_divisor = config.weight_quantization_mode.packing_divisor();
        let storage_type = config.weight_quantization_mode.storage_type();
        if weights_leaf.data_type() != storage_type {
            return Err(QuantizedLinearError::InvalidWeightsDataType {
                expected: storage_type,
                got: weights_leaf.data_type(),
            });
        }

        let scales_leaf = parameter_tree.leaf("scales").map_err(QuantizedLinearError::ParameterError)?;
        if scales_leaf.data_type() != kernel_data_type {
            return Err(QuantizedLinearError::InvalidScalesDataType {
                expected: kernel_data_type,
                got: scales_leaf.data_type(),
            });
        }

        let k_g = (input_dim + config.group_size - 1) / config.group_size;
        let weights_shape = weights_leaf.shape().to_vec();
        let scales_shape = scales_leaf.shape().to_vec();
        let (quantization_type, zero_points_or_biases) = match parameter_tree.leaf("deq_biases") {
            Ok(deq_biases_leaf) => {
                let deq_biases_shape = deq_biases_leaf.shape().to_vec();
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && deq_biases_shape == [output_dim, k_g])
                {
                    return Err(QuantizedLinearError::InvalidMlxShapes {
                        weights: weights_shape.into_boxed_slice(),
                        scales: scales_shape.into_boxed_slice(),
                        deq_biases: deq_biases_shape.into_boxed_slice(),
                        packing_divisor,
                    });
                }

                if deq_biases_leaf.data_type() != kernel_data_type {
                    return Err(QuantizedLinearError::InvalidDeqBiasesDataType {
                        expected: kernel_data_type,
                        got: deq_biases_leaf.data_type(),
                    });
                }

                (
                    QuantizedMatmulType::Mlx,
                    deq_biases_leaf.read_allocation().map_err(QuantizedLinearError::ParameterError)?,
                )
            },
            Err(_) => {
                let zero_points_leaf =
                    parameter_tree.leaf("zero_points").map_err(QuantizedLinearError::ParameterError)?;
                let zero_points_shape = zero_points_leaf.shape().to_vec();
                let expected_zero_points_entries = (k_g + packing_divisor - 1) / packing_divisor;
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && zero_points_shape == [output_dim, expected_zero_points_entries])
                {
                    return Err(QuantizedLinearError::InvalidAwqShapes {
                        weights: weights_shape.into_boxed_slice(),
                        scales: scales_shape.into_boxed_slice(),
                        zero_points: zero_points_shape.into_boxed_slice(),
                        packing_divisor,
                        packing_minus_one: packing_divisor - 1,
                    });
                }

                if zero_points_leaf.data_type() != storage_type {
                    return Err(QuantizedLinearError::InvalidZeroPointsDataType {
                        expected: storage_type,
                        got: zero_points_leaf.data_type(),
                    });
                }

                (
                    QuantizedMatmulType::ZeroPoint,
                    zero_points_leaf.read_allocation().map_err(QuantizedLinearError::ParameterError)?,
                )
            },
        };

        let (bias_add_kernel, biases) = match parameter_tree.leaf("biases") {
            Ok(biases_leaf) => {
                let bias_shape = biases_leaf.shape().to_vec();
                if bias_shape != [output_dim] {
                    return Err(QuantizedLinearError::InvalidBiasShape {
                        got: bias_shape.into_boxed_slice(),
                        expected_output_dim: output_dim,
                    });
                }

                if biases_leaf.data_type() != kernel_data_type {
                    return Err(QuantizedLinearError::InvalidBiasDataType {
                        expected: kernel_data_type,
                        got: biases_leaf.data_type(),
                    });
                }

                let bias_add_kernel =
                    <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, kernel_data_type, true)
                        .map_err(QuantizedLinearError::BackendError)?;
                (
                    Some(bias_add_kernel),
                    Some(biases_leaf.read_allocation().map_err(QuantizedLinearError::ParameterError)?),
                )
            },
            Err(_) => (None, None),
        };

        let kernel = QuantizedMatmulKernelEncodable::new(
            context,
            QuantizedMatmulConfiguration {
                data_type: kernel_data_type,
                group_size: config.group_size,
                input_dim,
                output_dim,
                mode: config.weight_quantization_mode,
                quantization_type,
                use_hadamard: output_hadamard_factors.is_some(),
            },
        )
        .map_err(QuantizedLinearError::QuantizedMatmulError)?;

        Ok(Self {
            kernel,
            bias_add_kernel,
            biases,
            weights: weights_leaf.read_allocation().map_err(QuantizedLinearError::ParameterError)?,
            scales: scales_leaf.read_allocation().map_err(QuantizedLinearError::ParameterError)?,
            zero_points_or_biases,
            output_hadamard_factors,
            output_dim,
            output_data_type: kernel_data_type,
        })
    }
}

impl<B: Backend> Linear<B> for QuantizedLinear<B> {
    fn encode(
        &self,
        context: &B::Context,
        input: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let _ = context;
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.output_data_type))?;

        self.kernel
            .encode(
                encoder,
                QuantizedMatmulArguments {
                    a: input,
                    b: &self.weights,
                    scales: &self.scales,
                    zero_points_or_biases: &self.zero_points_or_biases,
                    output: &mut output,
                    hadamard_factors: self.output_hadamard_factors.as_ref(),
                    batch_dim,
                },
            )
            .expect("Failed to encode quantized matmul");

        if let (Some(bias_add_kernel), Some(biases)) = (&self.bias_add_kernel, &self.biases) {
            let total_length = batch_dim * self.output_dim;
            bias_add_kernel.encode(
                None::<&Allocation<B>>,
                biases,
                &mut output,
                self.output_dim as u32,
                total_length as u32,
                encoder,
            );
        }

        Ok(output)
    }
}
