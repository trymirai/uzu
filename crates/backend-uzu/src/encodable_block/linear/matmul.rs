use std::cell::RefCell;

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Context, Encoder,
        gpu_types::{HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMethod, QuantizationMode},
        kernel::{
            Kernels,
            matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, int_spec::IntSpec, mlx_spec::MLXSpec},
    data_type::DataType,
    encodable_block::linear::Linear,
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
    #[error("Unsupported linear matmul configuration: {0}")]
    UnsupportedConfiguration(String),
}

enum Mode<B: Backend> {
    FullPrecision,
    Quantized {
        method: QuantizationMethod,
        mode: QuantizationMode,
        group_size: u32,
        scales: Allocation<B>,
        zero_points_or_biases: Option<Allocation<B>>,
        output_hadamard_factors: Option<Allocation<B>>,
    },
}

pub struct LinearMatmul<B: Backend> {
    kernel: RefCell<<B::Kernels as Kernels>::MatmulKernel>,
    weights: Allocation<B>,
    biases: Option<Allocation<B>>,
    input_dim: usize,
    output_dim: usize,
    output_data_type: DataType,
    mode: Mode<B>,
}

impl<B: Backend> LinearMatmul<B> {
    pub fn full_precision(
        context: &B::Context,
        input_dim: usize,
        output_dim: usize,
        has_biases: bool,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, LinearMatmulError<B>> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }

        let weights = parameter_tree
            .leaf("weights.weights")?
            .validate(&[output_dim, input_dim], weights_data_type)?
            .read_allocation()?;
        let biases =
            load_biases(weights_data_type, output_data_type, output_dim, has_biases.then_some(parameter_tree))?;

        let kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights,
            biases,
            input_dim,
            output_dim,
            output_data_type,
            mode: Mode::FullPrecision,
        })
    }

    pub fn quantized(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: usize,
        output_dim: usize,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, LinearMatmulError<B>> {
        let (bits, group_size, quantization_method) = match spec {
            AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                bits,
                group_size,
                layout: Layout::OutputInput,
                ..
            }) => (bits, group_size, QuantizationMethod::ScaleBias),
            AnyWeightMatrixSpec::IntSpec(IntSpec {
                bits,
                group_size,
                is_symmetric: false,
                layout: Layout::OutputInput,
                ..
            }) => (bits, group_size, QuantizationMethod::ScaleZeroPoint),
            AnyWeightMatrixSpec::IntSpec(IntSpec {
                bits,
                group_size,
                is_symmetric: true,
                layout: Layout::OutputInput,
                ..
            }) => (bits, group_size, QuantizationMethod::ScaleSymmetric),
            spec => return Err(LinearMatmulError::UnsupportedConfiguration(format!("{spec:?}"))),
        };

        let weight_quantization_mode = match bits {
            4 => QuantizationMode::U4,
            8 => QuantizationMode::U8,
            _ => {
                return Err(LinearMatmulError::UnsupportedConfiguration(format!(
                    "{quantization_method} bits={bits}, group_size={group_size}"
                )));
            },
        };

        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }

        let packing_divisor = weight_quantization_mode.packing_divisor();
        let storage_type = weight_quantization_mode.storage_type();
        let k_g = input_dim.div_ceil(group_size);

        let weights = weights_tree
            .leaf("weights")?
            .validate(&[output_dim, input_dim / packing_divisor], storage_type)?
            .read_allocation()?;
        let scales = weights_tree.leaf("scales")?.validate(&[output_dim, k_g], weights_data_type)?.read_allocation()?;
        let zero_points_or_biases = match quantization_method {
            QuantizationMethod::ScaleBias => {
                Some(weights_tree.leaf("biases")?.validate(&[output_dim, k_g], weights_data_type)?.read_allocation()?)
            },
            QuantizationMethod::ScaleZeroPoint => {
                let expected_zero_points_entries = k_g.div_ceil(packing_divisor);
                Some(
                    weights_tree
                        .leaf("zero_points")?
                        .validate(&[output_dim, expected_zero_points_entries], storage_type)?
                        .read_allocation()?,
                )
            },
            QuantizationMethod::ScaleSymmetric => None,
        };

        let biases = load_biases(weights_data_type, output_data_type, output_dim, bias_tree)?;

        let kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            weights,
            biases,
            input_dim,
            output_dim,
            output_data_type,
            mode: Mode::Quantized {
                method: quantization_method,
                mode: weight_quantization_mode,
                group_size: group_size as u32,
                scales,
                zero_points_or_biases,
                output_hadamard_factors,
            },
        })
    }
}

fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: usize,
    parameter_tree: Option<&ParameterTree<B>>,
) -> Result<Option<Allocation<B>>, LinearMatmulError<B>> {
    if parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    Ok(parameter_tree
        .map(|tree| tree.leaf("biases")?.validate(&[output_dim], weights_data_type)?.read_allocation())
        .transpose()?)
}

impl<B: Backend> LinearMatmul<B> {
    pub(crate) fn supports_int8_a(
        &self,
        context: &B::Context,
        group_size: u32,
    ) -> bool {
        let compatible_weights = matches!(
            &self.mode,
            Mode::Quantized {
                method: QuantizationMethod::ScaleSymmetric | QuantizationMethod::ScaleZeroPoint,
                mode: QuantizationMode::U8,
                group_size: weight_group_size,
                ..
            } if *weight_group_size == group_size
        );

        // The int8 GEMM is MXU-only and its K-loop is group-based, so the only
        // requirements are 8-bit sym/ZP weights, an RHT-block-aligned group that
        // divides K, and a hardware matrix unit.
        compatible_weights
            && context.supports_mxu()
            && group_size != 0
            && group_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE as u32)
            && (self.input_dim as u32).is_multiple_of(group_size)
    }

    pub(crate) fn encode_with_a(
        &self,
        a: MatmulA<'_, B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.output_data_type))?;

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
                    biases: zero_points_or_biases.as_ref().expect("ScaleBias quantization requires biases"),
                    mode: *mode,
                    group_size: *group_size,
                },
                QuantizationMethod::ScaleZeroPoint => MatmulB::ScaleZeroPointDequant {
                    b: &self.weights,
                    scales,
                    zero_points: zero_points_or_biases
                        .as_ref()
                        .expect("ScaleZeroPoint quantization requires zero_points"),
                    mode: *mode,
                    group_size: *group_size,
                },
                QuantizationMethod::ScaleSymmetric => MatmulB::ScaleSymmetricDequant {
                    b: &self.weights,
                    scales,
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
                a,
                b,
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

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            batch_dim,
            encoder,
        )
    }
}
