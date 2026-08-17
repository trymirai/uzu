use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulB,
        microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        microfloat_spec::{MicrofloatScaleMode, MicrofloatSpec},
    },
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

#[derive(Clone, Copy)]
pub struct QuantizationInfo {
    pub mode: QuantizationMode,
    pub method: QuantizationMethod,
    pub group_size: u32,
}

#[derive(Clone, Copy)]
pub struct MicrofloatInfo {
    pub format: MicrofloatFormat,
    pub bits: u32,
    pub group_size: u32,
}

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
    pub microfloat: Option<MicrofloatInfo>,
}

pub fn parse_spec<B: Backend>(spec: &AnyWeightMatrixSpec) -> Result<ParsedWeightSpec, WeightMatrixError<B>> {
    let (layout, quantized, microfloat) = match spec {
        AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout.clone(), None, None),
        AnyWeightMatrixSpec::MLXSpec(spec) => {
            (spec.layout.clone(), Some((spec.bits, spec.group_size, QuantizationMethod::ScaleBias)), None)
        },
        AnyWeightMatrixSpec::IntSpec(spec) => (
            spec.layout.clone(),
            Some((
                spec.bits,
                spec.group_size,
                if spec.is_symmetric {
                    QuantizationMethod::ScaleSymmetric
                } else {
                    QuantizationMethod::ScaleZeroPoint
                },
            )),
            None,
        ),
        AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
            bits,
            group_size,
            scale_mode,
            layout,
            ..
        }) => {
            let format = match scale_mode {
                MicrofloatScaleMode::Mxfp4 => MicrofloatFormat::Mxfp4,
                MicrofloatScaleMode::Nvfp4 => {
                    return Err(WeightMatrixError::UnsupportedConfiguration(
                        "NVFP4 runtime storage is not supported".into(),
                    ));
                },
            };
            let group_size = u32::try_from(*group_size).map_err(|_| {
                WeightMatrixError::UnsupportedConfiguration(format!("microfloat group size {group_size} exceeds u32"))
            })?;
            let runtime_layout = match layout {
                Layout::OutputInput => MicrofloatLayout::OutputInput,
                Layout::InputOutput => MicrofloatLayout::InputOutput,
            };
            MicrofloatMetadata::new(format, *bits, group_size, runtime_layout, 1, 1, group_size)
                .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            (
                layout.clone(),
                None,
                Some(MicrofloatInfo {
                    format,
                    bits: *bits,
                    group_size,
                }),
            )
        },
        spec => return Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
    };
    let quantization = match quantized {
        None => None,
        Some((bits, group_size, method)) => {
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
            Some(QuantizationInfo {
                mode,
                method,
                group_size,
            })
        },
    };
    Ok(ParsedWeightSpec {
        layout,
        quantization,
        microfloat,
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
    info: QuantizationInfo,
    signed_codes: bool,
}

struct Microfloat<B: Backend> {
    scales: Allocation<B>,
    global_scales: Allocation<B>,
    metadata: MicrofloatMetadata,
}

pub struct WeightMatrix<B: Backend> {
    values: Allocation<B>,
    quantized: Option<Quantized<B>>,
    microfloat: Option<Microfloat<B>>,
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
        Self::load_impl(tree, spec, required_layout, 1, output_dim, input_dim, data_type, false)
    }

    pub fn load_bank(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        matrix_count: u32,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
    ) -> Result<Self, WeightMatrixError<B>> {
        Self::load_impl(tree, spec, required_layout, matrix_count, output_dim, input_dim, data_type, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_impl(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        matrix_count: u32,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
        banked: bool,
    ) -> Result<Self, WeightMatrixError<B>> {
        let ParsedWeightSpec {
            layout,
            quantization,
            microfloat,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        if let Some(info) = microfloat {
            let runtime_layout = match layout {
                Layout::OutputInput => MicrofloatLayout::OutputInput,
                Layout::InputOutput => MicrofloatLayout::InputOutput,
            };
            let metadata = MicrofloatMetadata::new(
                info.format,
                info.bits,
                info.group_size,
                runtime_layout,
                matrix_count,
                rows,
                columns,
            )
            .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            let values = tree
                .leaf("weights")?
                .validate(&bank_shape(matrix_count, &[rows, columns / 2], banked), DataType::U8)?
                .read_allocation()?;
            let scales = tree
                .leaf("scales")?
                .validate(&bank_shape(matrix_count, &[rows, columns / info.group_size], banked), DataType::U8)?
                .read_allocation()?;
            let global_scale_shape = if banked {
                vec![matrix_count]
            } else {
                vec![1]
            };
            let global_scales =
                tree.leaf("global_scale")?.validate(&global_scale_shape, data_type)?.read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
                microfloat: Some(Microfloat {
                    scales,
                    global_scales,
                    metadata,
                }),
            });
        }

        let Some(info) = quantization else {
            let values = tree
                .leaf("weights")?
                .validate(&bank_shape(matrix_count, &[rows, columns], banked), data_type)?
                .read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
                microfloat: None,
            });
        };

        let group_size = info.group_size;
        let packing_divisor = info.mode.packing_divisor();
        let storage_data_type = info.mode.storage_type();
        if !columns.is_multiple_of(packing_divisor) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
            )));
        }
        let groups = columns.div_ceil(group_size);
        let values = tree
            .leaf("weights")?
            .validate(&bank_shape(matrix_count, &[rows, columns / packing_divisor], banked), storage_data_type)?
            .read_allocation()?;
        let scales = tree
            .leaf("scales")?
            .validate(&bank_shape(matrix_count, &[rows, groups], banked), data_type)?
            .read_allocation()?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => QuantizedCorrection::Biases(
                tree.leaf("biases")?
                    .validate(&bank_shape(matrix_count, &[rows, groups], banked), data_type)?
                    .read_allocation()?,
            ),
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(
                tree.leaf("zero_points")?
                    .validate(
                        &bank_shape(matrix_count, &[rows, groups.div_ceil(packing_divisor)], banked),
                        storage_data_type,
                    )?
                    .read_allocation()?,
            ),
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };
        Ok(Self {
            values,
            quantized: Some(Quantized {
                scales,
                correction,
                info,
                signed_codes: false,
            }),
            microfloat: None,
        })
    }

    pub fn values(&self) -> &Allocation<B> {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        self.microfloat
            .as_ref()
            .map(|microfloat| &microfloat.scales)
            .or_else(|| self.quantized.as_ref().map(|quantized| &quantized.scales))
    }

    pub fn zero_points(&self) -> Option<&Allocation<B>> {
        match &self.quantized.as_ref()?.correction {
            QuantizedCorrection::ZeroPoints(zero_points) => Some(zero_points),
            QuantizedCorrection::Biases(_) | QuantizedCorrection::Symmetric => None,
        }
    }

    pub fn biases(&self) -> Option<&Allocation<B>> {
        match &self.quantized.as_ref()?.correction {
            QuantizedCorrection::Biases(biases) => Some(biases),
            QuantizedCorrection::ZeroPoints(_) | QuantizedCorrection::Symmetric => None,
        }
    }

    pub fn matmul_b(&self) -> MatmulB<'_, B> {
        if let Some(microfloat) = self.microfloat.as_ref() {
            return MatmulB::Microfloat {
                codes: &self.values,
                scales: &microfloat.scales,
                global_scales: &microfloat.global_scales,
                metadata: microfloat.metadata,
            };
        }
        let Some(quantized) = self.quantized.as_ref() else {
            return MatmulB::FullPrecision {
                b: &self.values,
            };
        };
        let mode = quantized.info.mode;
        let group_size = quantized.info.group_size;
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
        let Some(quantized) = self.quantized.as_mut() else {
            return;
        };
        if quantized.signed_codes {
            return;
        }
        let Some(sign_flip_mask) = quantized.info.mode.weight_codes_sign_flip_mask() else {
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

fn bank_shape(
    matrix_count: u32,
    matrix_shape: &[u32],
    banked: bool,
) -> Vec<u32> {
    if !banked {
        return matrix_shape.to_vec();
    }
    std::iter::once(matrix_count).chain(matrix_shape.iter().copied()).collect()
}
