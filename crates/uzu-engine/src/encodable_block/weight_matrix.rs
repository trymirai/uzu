use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::{MatmulB, QuantParams, QuantParamsLayout},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
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

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
}

pub fn parse_spec<B: Backend>(spec: &AnyWeightMatrixSpec) -> Result<ParsedWeightSpec, WeightMatrixError<B>> {
    let (layout, quantized) = match spec {
        AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout.clone(), None),
        AnyWeightMatrixSpec::MLXSpec(spec) => {
            (spec.layout.clone(), Some((spec.bits, spec.group_size, QuantizationMethod::ScaleBias)))
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
        ),
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
    })
}

fn params_layout_for(layout: Layout) -> QuantParamsLayout {
    match layout {
        Layout::OutputInput => QuantParamsLayout::GroupOutput,
        Layout::InputOutput => QuantParamsLayout::OutputGroup,
    }
}

enum QuantizedCorrection<B: Backend> {
    Symmetric,
    Biases(Allocation<B>),
    ZeroPoints(Allocation<B>),
}

struct Quantized<B: Backend> {
    scales: Allocation<B>,
    correction: QuantizedCorrection<B>,
    params: QuantParams,
    info: QuantizationInfo,
    signed_codes: bool,
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
            quantization: quantization_info,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} weight layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        let Some(info) = quantization_info else {
            let values = tree.leaf("weights")?.validate(&[rows, columns], data_type)?.read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
            });
        };
        let params_layout = params_layout_for(layout.clone());

        let group_size = info.group_size;
        let packing_divisor = info.mode.packing_divisor();
        let storage_data_type = info.mode.storage_type();
        if !columns.is_multiple_of(packing_divisor) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
            )));
        }
        let groups = columns.div_ceil(group_size);

        let values =
            tree.leaf("weights")?.validate(&[rows, columns / packing_divisor], storage_data_type)?.read_allocation()?;
        let params = QuantParams::new(params_layout, rows, groups);
        let load_plane = |name: &str, plane_type: DataType| -> Result<Allocation<B>, WeightMatrixError<B>> {
            Ok(tree
                .leaf(name)?
                .validate(&params.shape(plane_type), params.storage_type(plane_type))?
                .read_allocation()?)
        };
        let scales = load_plane("scales", data_type)?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => QuantizedCorrection::Biases(load_plane("biases", data_type)?),
            QuantizationMethod::ScaleZeroPoint => {
                QuantizedCorrection::ZeroPoints(load_plane("zero_points", DataType::from(info.mode))?)
            },
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };

        Ok(Self {
            values,
            quantized: Some(Quantized {
                scales,
                correction,
                params,
                info,
                signed_codes: false,
            }),
        })
    }

    pub fn values(&self) -> &Allocation<B> {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        self.quantized.as_ref().map(|quantized| &quantized.scales)
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
                params: quantized.params,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::ZeroPoints(zero_points) => MatmulB::ScaleZeroPointDequant {
                b: &self.values,
                scales: &quantized.scales,
                zero_points,
                params: quantized.params,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::Symmetric => MatmulB::ScaleSymmetricDequant {
                b: &self.values,
                scales: &quantized.scales,
                params: quantized.params,
                mode,
                group_size,
                signed_codes,
            },
        }
    }

    pub fn try_prepare_a8_storage(&mut self) -> bool {
        let Some(quantized) = self.quantized.as_mut() else {
            return false;
        };
        quantized.prepare_a8_storage(&mut self.values)
    }
}

impl<B: Backend> Quantized<B> {
    fn prepare_a8_storage(
        &mut self,
        values: &mut Allocation<B>,
    ) -> bool {
        if self.params.layout() != QuantParamsLayout::GroupOutput {
            return false;
        }
        if self.info.mode != QuantizationMode::U4 {
            if !self.signed_codes
                && let Some(sign_flip_mask) = self.info.mode.weight_codes_sign_flip_mask()
            {
                let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
                let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(values.as_slice_mut());
                words.iter_mut().for_each(|word| *word ^= broadcast_mask);
                prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= sign_flip_mask);
            }
            self.signed_codes = true;
        }
        true
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
