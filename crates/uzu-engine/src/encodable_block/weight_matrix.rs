use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulB,
        microfloat::{MicrofloatEncoding, MicrofloatError, MicrofloatMetadata},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum WeightMatrixError<B: Backend> {
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Microfloat error: {0}")]
    MicrofloatError(#[from] MicrofloatError),
    #[error("Unsupported weight matrix configuration: {0}")]
    UnsupportedConfiguration(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationInfo {
    Integer {
        mode: QuantizationMode,
        method: QuantizationMethod,
        group_size: u32,
    },
    Microfloat(MicrofloatEncoding),
}

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
}

pub fn parse_spec<B: Backend>(spec: &AnyWeightMatrixSpec) -> Result<ParsedWeightSpec, WeightMatrixError<B>> {
    let (layout, quantization) = match spec {
        AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout.clone(), None),
        AnyWeightMatrixSpec::MLXSpec(spec) => {
            let quantization = integer_quantization::<B>(spec.bits, spec.group_size, QuantizationMethod::ScaleBias)?;
            (spec.layout.clone(), Some(quantization))
        },
        AnyWeightMatrixSpec::IntSpec(spec) => {
            let method = if spec.is_symmetric {
                QuantizationMethod::ScaleSymmetric
            } else {
                QuantizationMethod::ScaleZeroPoint
            };
            let quantization = integer_quantization::<B>(spec.bits, spec.group_size, method)?;
            (spec.layout.clone(), Some(quantization))
        },
        AnyWeightMatrixSpec::MicrofloatSpec(spec) => {
            if spec.layout != Layout::OutputInput {
                return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                    "microfloat matrices require output-input layout, got {:?}",
                    spec.layout,
                )));
            }
            let encoding = MicrofloatEncoding::new(spec.scale_mode, spec.bits, spec.group_size)?;
            (spec.layout.clone(), Some(QuantizationInfo::Microfloat(encoding)))
        },
        spec => return Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
    };
    Ok(ParsedWeightSpec {
        layout,
        quantization,
    })
}

fn integer_quantization<B: Backend>(
    bits: u32,
    group_size: u32,
    method: QuantizationMethod,
) -> Result<QuantizationInfo, WeightMatrixError<B>> {
    let mode = match bits {
        4 => QuantizationMode::U4,
        8 => QuantizationMode::U8,
        _ => {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "{method} bits={bits}, group_size={group_size}"
            )));
        },
    };
    if group_size == 0 {
        return Err(WeightMatrixError::UnsupportedConfiguration("group size must be non-zero".into()));
    }
    Ok(QuantizationInfo::Integer {
        mode,
        method,
        group_size,
    })
}

enum QuantizedCorrection<B: Backend> {
    Symmetric,
    Biases(Allocation<B>),
    ZeroPoints(Allocation<B>),
}

enum Quantized<B: Backend> {
    Integer {
        scales: Allocation<B>,
        correction: QuantizedCorrection<B>,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    Microfloat {
        scales: Allocation<B>,
        outer_scales: Allocation<B>,
        metadata: MicrofloatMetadata,
    },
}

pub struct WeightMatrix<B: Backend> {
    values: Allocation<B>,
    quantized: Option<Quantized<B>>,
}

impl<B: Backend> WeightMatrix<B> {
    pub fn load(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
    ) -> Result<Self, WeightMatrixError<B>> {
        let ParsedWeightSpec {
            layout,
            quantization,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        let Some(quantization) = quantization else {
            let values = tree.leaf("weights")?.validate(&[rows, columns], data_type)?.read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
            });
        };

        let (mode, method, group_size) = match quantization {
            QuantizationInfo::Microfloat(encoding) => {
                let metadata = MicrofloatMetadata::new(encoding, rows, columns)?;
                let values = tree.leaf("weights")?.validate(&[rows, columns / 2], DataType::U8)?.read_allocation()?;
                let scales = tree
                    .leaf("scales")?
                    .validate(&[rows, columns / encoding.group_size], DataType::U8)?
                    .read_allocation()?;
                let outer_scales = tree.leaf("global_scale")?.validate(&[1], data_type)?.read_allocation()?;
                return Ok(Self {
                    values,
                    quantized: Some(Quantized::Microfloat {
                        scales,
                        outer_scales,
                        metadata,
                    }),
                });
            },
            QuantizationInfo::Integer {
                mode,
                method,
                group_size,
            } => (mode, method, group_size),
        };

        let packing_divisor = mode.packing_divisor();
        let storage_data_type = mode.storage_type();
        if !columns.is_multiple_of(packing_divisor) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
            )));
        }
        let groups = columns.div_ceil(group_size);
        let values =
            tree.leaf("weights")?.validate(&[rows, columns / packing_divisor], storage_data_type)?.read_allocation()?;
        let scales = tree.leaf("scales")?.validate(&[rows, groups], data_type)?.read_allocation()?;
        let correction = match method {
            QuantizationMethod::ScaleBias => QuantizedCorrection::Biases(
                tree.leaf("biases")?.validate(&[rows, groups], data_type)?.read_allocation()?,
            ),
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(
                tree.leaf("zero_points")?
                    .validate(&[rows, groups.div_ceil(packing_divisor)], storage_data_type)?
                    .read_allocation()?,
            ),
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };
        Ok(Self {
            values,
            quantized: Some(Quantized::Integer {
                scales,
                correction,
                mode,
                group_size,
                signed_codes: false,
            }),
        })
    }

    /// Borrow weights; code mutation must also update the signed-code state.
    pub fn values(&self) -> &Allocation<B> {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        match &self.quantized {
            None => None,
            Some(Quantized::Microfloat {
                metadata,
                ..
            }) => Some(QuantizationInfo::Microfloat(metadata.encoding)),
            Some(Quantized::Integer {
                mode,
                correction,
                group_size,
                ..
            }) => {
                let method = match correction {
                    QuantizedCorrection::Symmetric => QuantizationMethod::ScaleSymmetric,
                    QuantizedCorrection::Biases(_) => QuantizationMethod::ScaleBias,
                    QuantizedCorrection::ZeroPoints(_) => QuantizationMethod::ScaleZeroPoint,
                };
                Some(QuantizationInfo::Integer {
                    mode: *mode,
                    method,
                    group_size: *group_size,
                })
            },
        }
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        match &self.quantized {
            None => None,
            Some(
                Quantized::Integer {
                    scales,
                    ..
                }
                | Quantized::Microfloat {
                    scales,
                    ..
                },
            ) => Some(scales),
        }
    }

    pub fn zero_points(&self) -> Option<&Allocation<B>> {
        match &self.quantized {
            Some(Quantized::Integer {
                correction: QuantizedCorrection::ZeroPoints(zero_points),
                ..
            }) => Some(zero_points),
            _ => None,
        }
    }

    pub fn biases(&self) -> Option<&Allocation<B>> {
        match &self.quantized {
            Some(Quantized::Integer {
                correction: QuantizedCorrection::Biases(biases),
                ..
            }) => Some(biases),
            _ => None,
        }
    }

    pub fn matmul_b(&self) -> MatmulB<'_, B> {
        let (scales, correction, mode, group_size, signed_codes) = match &self.quantized {
            None => {
                return MatmulB::FullPrecision {
                    b: &self.values,
                };
            },
            Some(Quantized::Microfloat {
                scales,
                outer_scales,
                metadata,
            }) => {
                return MatmulB::Microfloat {
                    codes: &self.values,
                    scales,
                    outer_scales,
                    metadata: *metadata,
                };
            },
            Some(Quantized::Integer {
                scales,
                correction,
                mode,
                group_size,
                signed_codes,
            }) => (scales, correction, *mode, *group_size, *signed_codes),
        };
        match correction {
            QuantizedCorrection::Biases(biases) => MatmulB::ScaleBiasDequant {
                b: &self.values,
                scales,
                biases,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::ZeroPoints(zero_points) => MatmulB::ScaleZeroPointDequant {
                b: &self.values,
                scales,
                zero_points,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::Symmetric => MatmulB::ScaleSymmetricDequant {
                b: &self.values,
                scales,
                mode,
                group_size,
                signed_codes,
            },
        }
    }

    pub fn make_codes_signed(&mut self) {
        let Some(Quantized::Integer {
            mode,
            signed_codes,
            ..
        }) = &mut self.quantized
        else {
            return;
        };
        if *signed_codes {
            return;
        }
        let Some(sign_flip_mask) = mode.weight_codes_sign_flip_mask() else {
            return;
        };
        let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
        let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(self.values.as_slice_mut());
        words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= sign_flip_mask);
        *signed_codes = true;
    }
}

fn physical_shape(
    layout: &Layout,
    output_dim: u32,
    input_dim: u32,
) -> (u32, u32) {
    match layout {
        Layout::OutputInput => (output_dim, input_dim),
        Layout::InputOutput => (input_dim, output_dim),
    }
}

#[cfg(test)]
#[path = "../../unit/encodable_block/weight_matrix/microfloat_test.rs"]
mod microfloat_test;
