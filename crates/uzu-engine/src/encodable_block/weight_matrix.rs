use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, BufferMut,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::{MatmulB, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection, TrellisFormat},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, qtip_gaussian::QtipGaussianSpec},
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

struct Quantized<B: Backend> {
    scales: B::GlobalBuffer,
    correction: QuantizedCorrection<B::GlobalBuffer>,
    params: QuantParams,
    info: QuantizationInfo,
    signed_codes: bool,
}

pub struct WeightMatrix<B: Backend> {
    values: B::GlobalBuffer,
    encoding: WeightEncoding<B>,
}

enum WeightEncoding<B: Backend> {
    Dense,
    Quantized(Quantized<B>),
    Trellis {
        row_scales: B::GlobalBuffer,
        codebook: [f32; 5],
        format: TrellisFormat,
    },
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
        match spec {
            AnyWeightMatrixSpec::FullPrecisionSpec(spec) => {
                check_layout(&spec.layout, &required_layout)?;
                let (rows, columns) = physical_shape(&required_layout, output_dim, input_dim);
                let values = tree.leaf("weights")?.validate(&[rows, columns], data_type)?.read_buffer()?;
                Ok(Self {
                    values,
                    encoding: WeightEncoding::Dense,
                })
            },
            AnyWeightMatrixSpec::MLXSpec(spec) => {
                check_layout(&spec.layout, &required_layout)?;
                load_quantized(
                    tree,
                    required_layout,
                    output_dim,
                    input_dim,
                    data_type,
                    spec.bits,
                    spec.group_size,
                    QuantizationMethod::ScaleBias,
                )
            },
            AnyWeightMatrixSpec::IntSpec(spec) => {
                check_layout(&spec.layout, &required_layout)?;
                load_quantized(
                    tree,
                    required_layout,
                    output_dim,
                    input_dim,
                    data_type,
                    spec.bits,
                    spec.group_size,
                    if spec.is_symmetric {
                        QuantizationMethod::ScaleSymmetric
                    } else {
                        QuantizationMethod::ScaleZeroPoint
                    },
                )
            },
            AnyWeightMatrixSpec::QtipGaussianSpec(spec) => {
                load_trellis(tree, &spec, required_layout, output_dim, input_dim)
            },
            spec => Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub fn values(&self) -> &B::GlobalBuffer {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&B::GlobalBuffer> {
        self.quantized().map(|quantized| &quantized.scales)
    }

    pub fn zero_points(&self) -> Option<&B::GlobalBuffer> {
        self.quantized()?.correction.zero_points()
    }

    pub fn biases(&self) -> Option<&B::GlobalBuffer> {
        self.quantized()?.correction.biases()
    }

    pub fn matmul_b(&self) -> MatmulB<&B::GlobalBuffer> {
        match &self.encoding {
            WeightEncoding::Dense => MatmulB::FullPrecision {
                b: &self.values,
            },
            WeightEncoding::Quantized(quantized) => MatmulB::Quantized(QuantizedB {
                codes: &self.values,
                scales: &quantized.scales,
                correction: quantized.correction.as_ref(),
                params: quantized.params,
                mode: quantized.info.mode,
                group_size: quantized.info.group_size,
                signed_codes: quantized.signed_codes,
            }),
            WeightEncoding::Trellis {
                row_scales,
                codebook,
                format,
            } => MatmulB::Trellis {
                codes: &self.values,
                row_scales,
                codebook: *codebook,
                format: *format,
            },
        }
    }

    pub fn try_prepare_a8_storage(&mut self) -> bool {
        match &mut self.encoding {
            WeightEncoding::Quantized(quantized) => quantized.prepare_a8_storage(&mut self.values),
            _ => false,
        }
    }

    pub fn a8_signed_codes(&self) -> Option<bool> {
        self.quantized().map(|quantized| quantized.info.mode != QuantizationMode::U4)
    }

    fn quantized(&self) -> Option<&Quantized<B>> {
        match &self.encoding {
            WeightEncoding::Quantized(quantized) => Some(quantized),
            _ => None,
        }
    }
}

fn check_layout<B: Backend>(
    layout: &Layout,
    required_layout: &Layout,
) -> Result<(), WeightMatrixError<B>> {
    if layout != required_layout {
        return Err(WeightMatrixError::UnsupportedConfiguration(format!(
            "expected {required_layout:?} weight layout, got {layout:?}"
        )));
    }
    Ok(())
}

fn load_quantized<B: Backend>(
    tree: &ParameterTree<B>,
    layout: Layout,
    output_dim: u32,
    input_dim: u32,
    data_type: DataType,
    bits: u32,
    group_size: u32,
    method: QuantizationMethod,
) -> Result<WeightMatrix<B>, WeightMatrixError<B>> {
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
    let info = QuantizationInfo {
        mode,
        method,
        group_size,
    };
    let (rows, columns) = physical_shape(&layout, output_dim, input_dim);
    // Parameters swap the weight axes once K is grouped: output-input stores [G, N], input-output stores [N, G].
    let params_layout = match layout {
        Layout::OutputInput => QuantParamsLayout::GroupOutput,
        Layout::InputOutput => QuantParamsLayout::OutputGroup,
    };
    let packing_divisor = info.mode.packing_divisor();
    if !columns.is_multiple_of(packing_divisor) {
        return Err(WeightMatrixError::UnsupportedConfiguration(format!(
            "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
        )));
    }
    let values =
        tree.leaf("weights")?.validate(&[rows, columns / packing_divisor], info.mode.storage_type())?.read_buffer()?;
    let params = QuantParams::new(params_layout, rows, columns.div_ceil(group_size));
    let load_plane =
        |name: &str, shape: [u32; 2], storage_type: DataType| -> Result<B::GlobalBuffer, WeightMatrixError<B>> {
            Ok(tree.leaf(name)?.validate(&shape, storage_type)?.read_buffer()?)
        };
    let scales = load_plane("scales", params.scale_shape(), data_type)?;
    let correction = match method {
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
    Ok(WeightMatrix {
        values,
        encoding: WeightEncoding::Quantized(Quantized {
            scales,
            correction,
            params,
            info,
            signed_codes: false,
        }),
    })
}

fn load_trellis<B: Backend>(
    tree: &ParameterTree<B>,
    spec: &QtipGaussianSpec,
    required_layout: Layout,
    output_dim: u32,
    input_dim: u32,
) -> Result<WeightMatrix<B>, WeightMatrixError<B>> {
    if required_layout != Layout::OutputInput {
        return Err(WeightMatrixError::UnsupportedConfiguration("QTIP Gaussian requires OutputInput layout".into()));
    }
    let format = TrellisFormat {
        vector_width: spec.vector_width,
        transition_bits: spec.transition_bits,
        restart_columns: spec.restart_columns,
    };
    let block_columns = if spec.restart_columns == 0 {
        input_dim
    } else {
        spec.restart_columns
    };
    let code_row_bytes =
        input_dim / block_columns * (16 + (block_columns / spec.vector_width - 1) * spec.transition_bits).div_ceil(8);
    let values = tree.leaf("codes")?.validate(&[output_dim, code_row_bytes], DataType::U8)?.read_buffer()?;
    let row_scales = tree.leaf("scales")?.validate(&[output_dim], DataType::F32)?.read_buffer()?;
    let codebook: [f32; 5] = tree
        .leaf("codebook")?
        .validate(&[5], DataType::F32)?
        .read_slice::<f32>()?
        .as_ref()
        .try_into()
        .expect("validated codebook has five values");
    Ok(WeightMatrix {
        values,
        encoding: WeightEncoding::Trellis {
            row_scales,
            codebook,
            format,
        },
    })
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
