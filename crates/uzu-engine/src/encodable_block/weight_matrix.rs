use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulB,
        microfloat::{MicrofloatEncoding, MicrofloatMetadata},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, microfloat_spec::MicrofloatSpec},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum WeightMatrixError<B: Backend> {
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
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
        AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
            bits,
            group_size,
            scale_mode,
            layout,
            ..
        }) => {
            if layout != &Layout::OutputInput {
                return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                    "microfloat matrices require output-input layout, got {layout:?}"
                )));
            }
            let group_size = u32::try_from(*group_size).map_err(|_| {
                WeightMatrixError::UnsupportedConfiguration(format!("microfloat group size {group_size} exceeds u32"))
            })?;
            let encoding = MicrofloatEncoding::new(*scale_mode, *bits, group_size)
                .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            (layout.clone(), Some(QuantizationInfo::Microfloat(encoding)))
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

struct Quantized<B: Backend> {
    scales: Allocation<B>,
    correction: QuantizedCorrection<B>,
    mode: QuantizationMode,
    method: QuantizationMethod,
    group_size: u32,
    signed_codes: bool,
}

struct Microfloat<B: Backend> {
    scales: Allocation<B>,
    outer_scales: Allocation<B>,
    metadata: MicrofloatMetadata,
}

enum WeightFormat<B: Backend> {
    FullPrecision,
    Integer(Quantized<B>),
    Microfloat(Microfloat<B>),
}

pub struct WeightMatrix<B: Backend> {
    values: Allocation<B>,
    format: WeightFormat<B>,
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
                format: WeightFormat::FullPrecision,
            });
        };

        let (mode, method, group_size) = match quantization {
            QuantizationInfo::Microfloat(encoding) => {
                let metadata = MicrofloatMetadata::new(encoding, rows, columns)
                    .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
                let values = tree.leaf("weights")?.validate(&[rows, columns / 2], DataType::U8)?.read_allocation()?;
                let scales = tree
                    .leaf("scales")?
                    .validate(&[rows, columns / encoding.group_size], DataType::U8)?
                    .read_allocation()?;
                // Preserving the artifact's established tensor name for the outer scale.
                let outer_scales = tree.leaf("global_scale")?.validate(&[1], data_type)?.read_allocation()?;
                return Ok(Self {
                    values,
                    format: WeightFormat::Microfloat(Microfloat {
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
            format: WeightFormat::Integer(Quantized {
                scales,
                correction,
                mode,
                method,
                group_size,
                signed_codes: false,
            }),
        })
    }

    pub fn values(&self) -> &Allocation<B> {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        match &self.format {
            WeightFormat::FullPrecision => None,
            WeightFormat::Integer(quantized) => Some(QuantizationInfo::Integer {
                mode: quantized.mode,
                method: quantized.method,
                group_size: quantized.group_size,
            }),
            WeightFormat::Microfloat(microfloat) => Some(QuantizationInfo::Microfloat(microfloat.metadata.encoding)),
        }
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        match &self.format {
            WeightFormat::Integer(quantized) => Some(&quantized.scales),
            WeightFormat::Microfloat(microfloat) => Some(&microfloat.scales),
            WeightFormat::FullPrecision => None,
        }
    }

    pub fn zero_points(&self) -> Option<&Allocation<B>> {
        match &self.format {
            WeightFormat::Integer(Quantized {
                correction: QuantizedCorrection::ZeroPoints(zero_points),
                ..
            }) => Some(zero_points),
            WeightFormat::FullPrecision | WeightFormat::Integer(_) | WeightFormat::Microfloat(_) => None,
        }
    }

    pub fn biases(&self) -> Option<&Allocation<B>> {
        match &self.format {
            WeightFormat::Integer(Quantized {
                correction: QuantizedCorrection::Biases(biases),
                ..
            }) => Some(biases),
            WeightFormat::FullPrecision | WeightFormat::Integer(_) | WeightFormat::Microfloat(_) => None,
        }
    }

    pub fn matmul_b(&self) -> MatmulB<'_, B> {
        let quantized = match &self.format {
            WeightFormat::FullPrecision => {
                return MatmulB::FullPrecision {
                    b: &self.values,
                };
            },
            WeightFormat::Microfloat(microfloat) => {
                return MatmulB::Microfloat {
                    codes: &self.values,
                    scales: &microfloat.scales,
                    outer_scales: &microfloat.outer_scales,
                    metadata: microfloat.metadata,
                };
            },
            WeightFormat::Integer(quantized) => quantized,
        };
        let mode = quantized.mode;
        let group_size = quantized.group_size;
        let signed_codes = quantized.signed_codes;
        match &quantized.correction {
            QuantizedCorrection::Biases(biases) => MatmulB::ScaleBiasDequant {
                b: &self.values,
                scales: &quantized.scales,
                biases,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::ZeroPoints(zero_points) => MatmulB::ScaleZeroPointDequant {
                b: &self.values,
                scales: &quantized.scales,
                zero_points,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::Symmetric => MatmulB::ScaleSymmetricDequant {
                b: &self.values,
                scales: &quantized.scales,
                mode,
                group_size,
                signed_codes,
            },
        }
    }

    pub fn make_codes_signed(&mut self) {
        let WeightFormat::Integer(quantized) = &mut self.format else {
            return;
        };
        if quantized.signed_codes {
            return;
        }
        let Some(sign_flip_mask) = quantized.mode.weight_codes_sign_flip_mask() else {
            return;
        };
        let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
        let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(self.values.as_slice_mut());
        words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= sign_flip_mask);
        quantized.signed_codes = true;
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
