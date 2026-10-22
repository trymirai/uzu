use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, BufferMut,
        gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
        kernel::matmul::{
            MatmulB, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection, TRELLIS_CODES_TAIL_BYTES,
            TrellisFormat, WeightsLayout,
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, qtip_gaussian::QtipGaussianSpec},
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
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

struct Quantized<B: Backend> {
    scales: B::GlobalBuffer,
    correction: QuantizedCorrection<B::GlobalBuffer>,
    params: QuantParams,
    info: QuantizationInfo,
    signed_codes: bool,
    weights_layout: WeightsLayout<B>,
}

struct Trellis<B: Backend> {
    row_scales: B::GlobalBuffer,
    codebook: B::GlobalBuffer,
    format: TrellisFormat,
}

pub struct WeightMatrix<B: Backend> {
    values: B::GlobalBuffer,
    quantized: Option<Quantized<B>>,
    trellis: Option<Trellis<B>>,
}

impl<B: Backend> WeightMatrix<B> {
    pub fn load(
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        tree: &ParameterTree<B>,
    ) -> Result<Self, WeightMatrixError<B>> {
        if let AnyWeightMatrixSpec::QtipGaussianSpec(spec) = &spec {
            return Self::load_trellis(output_dim, input_dim, spec, required_layout, tree);
        }
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
            let values = tree.read_buffer("weights", &[rows, columns], data_type)?;
            return Ok(Self {
                values,
                quantized: None,
                trellis: None,
            });
        };
        // Parameters swap the weight axes once K is grouped: output-input stores [G, N], input-output stores [N, G].
        let params_layout = match layout {
            Layout::OutputInput => QuantParamsLayout::GroupOutput,
            Layout::InputOutput => QuantParamsLayout::OutputGroup,
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

        let values = tree.read_buffer("weights", &[rows, columns / packing_divisor], storage_data_type)?;
        let params = QuantParams::new(params_layout, rows, groups);
        let load_plane =
            |name: &str, shape: [u32; 2], storage_type: DataType| -> Result<B::GlobalBuffer, WeightMatrixError<B>> {
                Ok(tree.read_buffer(name, &shape, storage_type)?)
            };
        let scales = load_plane("scales", params.scale_shape(), data_type)?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => {
                QuantizedCorrection::Biases(load_plane("biases", params.scale_shape(), data_type)?)
            },
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(load_plane(
                "zero_points",
                params.zero_point_shape(info.mode),
                info.mode.storage_type(),
            )?),
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
                weights_layout: Default::default(),
            }),
            trellis: None,
        })
    }

    fn load_trellis(
        output_dim: u32,
        input_dim: u32,
        spec: &QtipGaussianSpec,
        required_layout: Layout,
        tree: &ParameterTree<B>,
    ) -> Result<Self, WeightMatrixError<B>> {
        if required_layout != Layout::OutputInput {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "QTIP Gaussian requires OutputInput layout".into(),
            ));
        }
        let format = TrellisFormat {
            vector_width: spec.vector_width,
            transition_bits: spec.transition_bits,
            restart_columns: (spec.restart_columns != 0).then_some(spec.restart_columns),
        };
        let block_columns = format.restart_columns.unwrap_or(input_dim);
        if !matches!(format.vector_width, 2 | 4)
            || !(4..=8).contains(&format.transition_bits)
            || !block_columns.is_multiple_of(2 * format.vector_width)
            || !input_dim.is_multiple_of(block_columns)
        {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "trellis format {format:?} with {input_dim} columns"
            )));
        }
        let code_shape = [output_dim, format.code_row_bytes(input_dim)];
        Ok(Self {
            values: tree.read_buffer_with_tail("codes", &code_shape, DataType::U8, TRELLIS_CODES_TAIL_BYTES)?,
            quantized: None,
            trellis: Some(Trellis {
                row_scales: tree.read_buffer("scales", &[output_dim], DataType::F32)?,
                codebook: tree.read_buffer("codebook", &[5], DataType::F32)?,
                format,
            }),
        })
    }

    pub fn values(&self) -> &B::GlobalBuffer {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref().map(|quantized| &quantized.scales)
    }

    pub fn zero_points(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref()?.correction.zero_points()
    }

    pub fn biases(&self) -> Option<&B::GlobalBuffer> {
        self.quantized.as_ref()?.correction.biases()
    }

    pub fn matmul_b(&self) -> MatmulB<&B::GlobalBuffer> {
        if let Some(trellis) = &self.trellis {
            return MatmulB::Trellis {
                codes: &self.values,
                row_scales: &trellis.row_scales,
                codebook: &trellis.codebook,
                format: trellis.format,
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
        MatmulB::Quantized(QuantizedB {
            codes: &self.values,
            scales: &quantized.scales,
            correction: quantized.correction.as_ref(),
            params: quantized.params,
            mode,
            group_size,
            signed_codes,
            weights_layout: quantized.weights_layout,
        })
    }

    pub fn optimize_weights_layout<E>(
        &mut self,
        optimize: impl FnOnce(&mut B::GlobalBuffer) -> Result<WeightsLayout<B>, E>,
    ) -> Result<(), E> {
        let Some(quantized) =
            self.quantized.as_mut().filter(|quantized| quantized.weights_layout == WeightsLayout::<B>::default())
        else {
            return Ok(());
        };
        quantized.weights_layout = optimize(&mut self.values)?;
        Ok(())
    }

    pub fn try_prepare_a8_storage(&mut self) -> bool {
        let Some(quantized) = self.quantized.as_mut() else {
            return false;
        };
        quantized.prepare_a8_storage(&mut self.values)
    }

    pub fn a8_signed_codes(&self) -> Option<bool> {
        self.quantized.as_ref().map(|quantized| quantized.info.mode != QuantizationMode::U4)
    }
}

impl<B: Backend> Quantized<B> {
    fn prepare_a8_storage(
        &mut self,
        values: impl BufferMut<Buffer: BufferCpuAccessible>,
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
