use std::cell::RefCell;

use thiserror::Error;

use super::Linear;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{
            ManualKernels,
            matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulQuantCombo},
        },
    },
    config::QuantizationConfig,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unexpected weights shape: got {got:?}, expected [{expected_output_dim}, {expected_input_dim}]")]
    InvalidFpWeightsShape {
        got: Box<[usize]>,
        expected_output_dim: usize,
        expected_input_dim: usize,
    },
    #[error("Weights dtype mismatch: got {got:?}, expected {expected:?}")]
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
        "Unexpected scale-bias shapes. weights={weights:?}, scales={scales:?}, deq_biases={deq_biases:?}; expected [N,K/{packing_divisor}],[N,K_g],[N,K_g]"
    )]
    InvalidScaleBiasShapes {
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
        "Unexpected scale-zero-point shapes. weights={weights:?}, scales={scales:?}, zero_points={zero_points:?}; expected [N,K/{packing_divisor}],[N,K_g],[N,(K_g+{packing_minus_one})/{packing_divisor}]"
    )]
    InvalidScaleZeroPointShapes {
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

enum Mode<B: Backend> {
    FullPrecision,
    Quantized {
        method: QuantizationMethod,
        mode: QuantizationMode,
        group_size: u32,
        scales: Allocation<B>,
        zero_points_or_biases: Allocation<B>,
        output_hadamard_factors: Option<Allocation<B>>,
    },
}

pub struct LinearMatmul<B: Backend> {
    kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    weights: Allocation<B>,
    biases: Option<Allocation<B>>,
    input_dim: usize,
    output_dim: usize,
    data_type: DataType,
    mode: Mode<B>,
}

impl<B: Backend> LinearMatmul<B> {
    pub fn full_precision(
        context: &B::Context,
        precision: DataType,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, LinearMatmulError<B>> {
        if !matches!(precision, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(LinearMatmulError::UnsupportedDataType(precision));
        }

        let weights_leaf = parameter_tree.leaf("weights")?;
        let weights_shape = weights_leaf.shape().to_vec();
        if weights_shape != [output_dim, input_dim] {
            return Err(LinearMatmulError::InvalidFpWeightsShape {
                got: weights_shape.into_boxed_slice(),
                expected_output_dim: output_dim,
                expected_input_dim: input_dim,
            });
        }
        if weights_leaf.data_type() != precision {
            return Err(LinearMatmulError::InvalidWeightsDataType {
                expected: precision,
                got: weights_leaf.data_type(),
            });
        }

        let biases = load_biases(precision, output_dim, parameter_tree)?;

        let kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(context, precision)
            .map_err(LinearMatmulError::BackendError)?;
        let weights = weights_leaf.read_allocation()?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights,
            biases,
            input_dim,
            output_dim,
            data_type: precision,
            mode: Mode::FullPrecision,
        })
    }

    pub fn quantized(
        context: &B::Context,
        config: &QuantizationConfig,
        input_dim: usize,
        output_dim: usize,
        parameter_tree: &ParameterTree<B::Context>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, LinearMatmulError<B>> {
        let data_type: DataType = config.activation_precision.into();
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(LinearMatmulError::UnsupportedDataType(data_type));
        }

        let packing_divisor = config.weight_quantization_mode.packing_divisor();
        let storage_type = config.weight_quantization_mode.storage_type();

        let weights_leaf = parameter_tree.leaf("weights")?;
        if weights_leaf.data_type() != storage_type {
            return Err(LinearMatmulError::InvalidWeightsDataType {
                expected: storage_type,
                got: weights_leaf.data_type(),
            });
        }

        let scales_leaf = parameter_tree.leaf("scales")?;
        if scales_leaf.data_type() != data_type {
            return Err(LinearMatmulError::InvalidScalesDataType {
                expected: data_type,
                got: scales_leaf.data_type(),
            });
        }

        let k_g = input_dim.div_ceil(config.group_size);
        let weights_shape = weights_leaf.shape().to_vec();
        let scales_shape = scales_leaf.shape().to_vec();

        let (quantization_method, zero_points_or_biases) = match parameter_tree.leaf("deq_biases") {
            Ok(deq_biases) => {
                let deq_biases_shape = deq_biases.shape().to_vec();
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && deq_biases_shape == [output_dim, k_g])
                {
                    return Err(LinearMatmulError::InvalidScaleBiasShapes {
                        weights: weights_shape.into_boxed_slice(),
                        scales: scales_shape.into_boxed_slice(),
                        deq_biases: deq_biases_shape.into_boxed_slice(),
                        packing_divisor,
                    });
                }
                if deq_biases.data_type() != data_type {
                    return Err(LinearMatmulError::InvalidDeqBiasesDataType {
                        expected: data_type,
                        got: deq_biases.data_type(),
                    });
                }
                (QuantizationMethod::ScaleBias, deq_biases.read_allocation()?)
            },
            Err(_) => {
                let zero_points_leaf = parameter_tree.leaf("zero_points")?;
                let zero_points_shape = zero_points_leaf.shape().to_vec();
                let expected_zero_points_entries = k_g.div_ceil(packing_divisor);
                if !(weights_shape == [output_dim, input_dim / packing_divisor]
                    && scales_shape == [output_dim, k_g]
                    && zero_points_shape == [output_dim, expected_zero_points_entries])
                {
                    return Err(LinearMatmulError::InvalidScaleZeroPointShapes {
                        weights: weights_shape.into_boxed_slice(),
                        scales: scales_shape.into_boxed_slice(),
                        zero_points: zero_points_shape.into_boxed_slice(),
                        packing_divisor,
                        packing_minus_one: packing_divisor - 1,
                    });
                }
                if zero_points_leaf.data_type() != storage_type {
                    return Err(LinearMatmulError::InvalidZeroPointsDataType {
                        expected: storage_type,
                        got: zero_points_leaf.data_type(),
                    });
                }
                (QuantizationMethod::ScaleZeroPoint, zero_points_leaf.read_allocation()?)
            },
        };

        let biases = load_biases(data_type, output_dim, parameter_tree)?;

        let mut kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)
            .map_err(LinearMatmulError::BackendError)?;
        kernel
            .preheat_quant_combo(
                context,
                MatmulQuantCombo {
                    method: quantization_method,
                    mode: config.weight_quantization_mode,
                    group_size: config.group_size as u32,
                },
            )
            .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights: weights_leaf.read_allocation()?,
            biases,
            input_dim,
            output_dim,
            data_type,
            mode: Mode::Quantized {
                method: quantization_method,
                mode: config.weight_quantization_mode,
                group_size: config.group_size as u32,
                scales: scales_leaf.read_allocation()?,
                zero_points_or_biases,
                output_hadamard_factors,
            },
        })
    }
}

fn load_biases<B: Backend>(
    data_type: DataType,
    output_dim: usize,
    parameter_tree: &ParameterTree<B::Context>,
) -> Result<Option<Allocation<B>>, LinearMatmulError<B>> {
    match parameter_tree.leaf("biases") {
        Ok(biases_leaf) => {
            let bias_shape = biases_leaf.shape().to_vec();
            if bias_shape != [output_dim] {
                return Err(LinearMatmulError::InvalidBiasShape {
                    got: bias_shape.into_boxed_slice(),
                    expected_output_dim: output_dim,
                });
            }
            if biases_leaf.data_type() != data_type {
                return Err(LinearMatmulError::InvalidBiasDataType {
                    expected: data_type,
                    got: biases_leaf.data_type(),
                });
            }
            Ok(Some(biases_leaf.read_allocation()?))
        },
        Err(_) => Ok(None),
    }
}

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.data_type))?;

        let b = match &self.mode {
            Mode::FullPrecision => MatmulB::FullPrecision {
                b: &self.weights,
            },
            Mode::Quantized {
                method,
                mode,
                group_size,
                scales,
                zero_points_or_biases,
                ..
            } => match method {
                QuantizationMethod::ScaleBias => MatmulB::ScaleBiasDequant {
                    b: &self.weights,
                    scales,
                    biases: zero_points_or_biases,
                    mode: *mode,
                    group_size: *group_size,
                },
                QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                    b: &self.weights,
                    scales,
                    zero_points: zero_points_or_biases,
                    mode: *mode,
                    group_size: *group_size,
                },
            },
        };

        let rht_factors = match &self.mode {
            Mode::Quantized {
                output_hadamard_factors: Some(factors),
                ..
            } => Some(factors),
            _ => None,
        };
        let d_transform = MatmulDOps {
            ab_scale: 1.0,
            accumulate: false,
            bias: self.biases.as_ref(),
            rht_factors,
        };

        self.kernel.borrow_mut().encode(
            MatmulArguments {
                a: &input,
                a_offset: 0,
                b,
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform,
                m: batch_dim as u32,
                n: self.output_dim as u32,
                k: self.input_dim as u32,
            },
            encoder,
        )?;

        Ok(output)
    }
}
