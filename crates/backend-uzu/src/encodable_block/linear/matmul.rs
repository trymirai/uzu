use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{QuantizationMethod, QuantizationMode},
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
    kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
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
            kernel: Mutex::new(kernel),
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

        let weights_leaf = weights_tree.leaf("weights")?;
        let weight_quantization_mode = match QuantizationMode::from_storage(bits, weights_leaf.data_type()) {
            Some(mode) => mode,
            None => {
                return Err(LinearMatmulError::UnsupportedConfiguration(format!(
                    "{quantization_method} bits={bits}, group_size={group_size}, storage={:?}",
                    weights_leaf.data_type()
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

        let weights =
            weights_leaf.validate(&[output_dim, input_dim / packing_divisor], storage_type)?.read_allocation()?;
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
                        .validate(&[output_dim, expected_zero_points_entries], DataType::U8)?
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
            kernel: Mutex::new(kernel),
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
    pub(super) fn encode_with_a(
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
            bias: self.biases.as_ref(),
            rht_factors,
            ..MatmulDOps::none()
        };

        self.kernel.lock().encode(
            MatmulArguments {
                a,
                b,
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform,
                gather_indices: None,
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use proc_macros::uzu_test;
    use tempfile::tempfile;

    use super::*;
    use crate::backends::{common::Context, cpu::Cpu};

    fn load_zero_point_w8(
        weights_dtype: &str,
        zero_points_dtype: &str,
    ) -> Result<LinearMatmul<Cpu>, LinearMatmulError<Cpu>> {
        let tensors = serde_json::json!({
            "weights": {"dtype": weights_dtype, "shape": [2, 64], "data_offsets": [0, 128]},
            "scales": {"dtype": "BF16", "shape": [2, 1], "data_offsets": [128, 132]},
            "zero_points": {"dtype": zero_points_dtype, "shape": [2, 1], "data_offsets": [132, 134]}
        });
        let header = serde_json::to_vec(&tensors).expect("serialize safetensor header");
        let mut file = tempfile().expect("temporary safetensor");
        file.write_all(&(header.len() as u64).to_le_bytes()).expect("write header length");
        file.write_all(&header).expect("write header");

        let context = <Cpu as Backend>::Context::new().expect("CPU context");
        let loader = crate::parameters::ParameterLoader::<Cpu>::new_random(&file, context.as_ref(), 0)
            .expect("parse safetensor header");
        let spec = serde_json::from_value(serde_json::json!({
            "type": "IntSpec",
            "bits": 8,
            "group_size": 64,
            "is_symmetric": false,
            "layout": "output_input"
        }))
        .expect("parse IntSpec");
        LinearMatmul::quantized(
            context.as_ref(),
            spec,
            64,
            2,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            &loader.tree(),
            None,
            None,
        )
    }

    #[uzu_test]
    fn w8_storage_mode_comes_from_safetensor_dtype() {
        for (dtype, expected) in [("U8", QuantizationMode::U8), ("I8", QuantizationMode::I8)] {
            let linear = load_zero_point_w8(dtype, "U8").expect("valid W8 tensor dtypes");
            let Mode::Quantized {
                mode,
                ..
            } = linear.mode
            else {
                panic!("expected quantized mode");
            };
            assert_eq!(mode, expected);
        }
    }

    #[uzu_test]
    fn signed_w8_still_requires_unsigned_zero_points() {
        let Err(error) = load_zero_point_w8("I8", "I8") else {
            panic!("I8 zero-points must be rejected");
        };
        assert!(matches!(
            error,
            LinearMatmulError::ParameterError(ParameterLoaderError::InvalidTensor {
                data_type: DataType::I8,
                expected_data_type: DataType::U8,
                ..
            })
        ));
    }
}
