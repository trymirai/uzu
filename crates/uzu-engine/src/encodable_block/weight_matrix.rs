use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend,
        gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
        kernel::{
            matmul::{MatmulB, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection, TrellisFormat, WeightsLayout},
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        qtip_gaussian::{CODEBOOK_V2, CODEBOOK_V4},
    },
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

struct Quantized<B: Backend> {
    scales: B::GlobalGpuBuffer,
    correction: QuantizedCorrection<B::GlobalGpuBuffer>,
    params: QuantParams,
    mode: QuantizationMode,
    group_size: u32,
    signed_codes: bool,
    weights_layout: WeightsLayout<B>,
}

struct Trellis<B: Backend> {
    row_scales: B::GlobalGpuBuffer,
    codebook: B::GlobalGpuBuffer,
    format: TrellisFormat,
}

pub struct WeightMatrix<B: Backend> {
    values: B::GlobalGpuBuffer,
    encoding: Encoding<B>,
}

enum Encoding<B: Backend> {
    Dense,
    Quantized(Quantized<B>),
    Trellis(Trellis<B>),
}

impl<B: Backend> WeightMatrix<B> {
    pub async fn load(
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        tree: &ParameterTree<'_, B>,
    ) -> Result<Self, WeightMatrixError<B>> {
        let (layout, quantized) = match spec {
            AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout, None),
            AnyWeightMatrixSpec::MLXSpec(spec) => {
                (spec.layout, Some((spec.bits, spec.group_size, QuantizationMethod::ScaleBias)))
            },
            AnyWeightMatrixSpec::IntSpec(spec) => (
                spec.layout,
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
            AnyWeightMatrixSpec::QtipGaussianSpec(spec) => {
                let format = TrellisFormat {
                    vector_width: spec.vector_width,
                    transition_bits: spec.transition_bits,
                    restart_columns: (spec.restart_columns != 0).then_some(spec.restart_columns),
                };
                let code_shape = [output_dim, format.code_row_bytes(input_dim)];
                let values = tree.read_buffer("codes", &code_shape, DataType::U8).await?;
                let row_scales = tree.read_buffer("scales", &[output_dim], DataType::F32).await?;
                let codebook = tree.create_buffer_with_data(match spec.vector_width {
                    2 => &CODEBOOK_V2,
                    4 => &CODEBOOK_V4,
                    _ => unreachable!("unsupported Trellis vector width"),
                })?;
                return Ok(Self {
                    values,
                    encoding: Encoding::Trellis(Trellis {
                        row_scales,
                        codebook,
                        format,
                    }),
                });
            },
            spec => return Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
        };
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} weight layout, got {layout:?}"
            )));
        }
        let (rows, columns) = match layout {
            Layout::OutputInput => (output_dim, input_dim),
            Layout::InputOutput => (input_dim, output_dim),
        };

        let Some((bits, group_size, method)) = quantized else {
            let values = tree.read_buffer("weights", &[rows, columns], data_type).await?;
            return Ok(Self {
                values,
                encoding: Encoding::Dense,
            });
        };
        let mode = match bits {
            2 => QuantizationMode::U2,
            4 => QuantizationMode::U4,
            8 => QuantizationMode::U8,
            _ => {
                return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                    "{method} bits={bits}, group_size={group_size}"
                )));
            },
        };
        // Parameters swap the weight axes once K is grouped: output-input stores [G, N], input-output stores [N, G].
        let params_layout = match layout {
            Layout::OutputInput => QuantParamsLayout::GroupOutput,
            Layout::InputOutput => QuantParamsLayout::OutputGroup,
        };

        let packing_divisor = mode.packing_divisor();
        if !columns.is_multiple_of(packing_divisor) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
            )));
        }
        let values = tree.read_buffer("weights", &[rows, columns / packing_divisor], mode.storage_type()).await?;
        let params = QuantParams::new(params_layout, rows, columns.div_ceil(group_size));
        let scales = tree.read_buffer("scales", &params.scale_shape(), data_type).await?;
        let correction = match method {
            QuantizationMethod::ScaleBias => {
                QuantizedCorrection::Biases(tree.read_buffer("biases", &params.scale_shape(), data_type).await?)
            },
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(
                tree.read_buffer("zero_points", &params.zero_point_shape(mode), mode.storage_type()).await?,
            ),
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };

        Ok(Self {
            values,
            encoding: Encoding::Quantized(Quantized {
                scales,
                correction,
                params,
                mode,
                group_size,
                signed_codes: false,
                weights_layout: Default::default(),
            }),
        })
    }

    pub fn values(&self) -> &B::GlobalGpuBuffer {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized().map(|quantized| QuantizationInfo {
            mode: quantized.mode,
            method: match &quantized.correction {
                QuantizedCorrection::Symmetric => QuantizationMethod::ScaleSymmetric,
                QuantizedCorrection::Biases(_) => QuantizationMethod::ScaleBias,
                QuantizedCorrection::ZeroPoints(_) => QuantizationMethod::ScaleZeroPoint,
            },
            group_size: quantized.group_size,
        })
    }

    pub fn scales(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized().map(|quantized| &quantized.scales)
    }

    pub fn zero_points(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized()?.correction.zero_points()
    }

    pub fn biases(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized()?.correction.biases()
    }

    pub fn matmul_b(&self) -> MatmulB<&B::GlobalGpuBuffer> {
        match &self.encoding {
            Encoding::Dense => MatmulB::FullPrecision {
                b: &self.values,
            },
            Encoding::Quantized(quantized) => MatmulB::Quantized(QuantizedB {
                codes: &self.values,
                scales: &quantized.scales,
                correction: quantized.correction.as_ref(),
                params: quantized.params,
                mode: quantized.mode,
                group_size: quantized.group_size,
                signed_codes: quantized.signed_codes,
                weights_layout: quantized.weights_layout,
            }),
            Encoding::Trellis(trellis) => MatmulB::Trellis {
                codes: &self.values,
                row_scales: &trellis.row_scales,
                codebook: &trellis.codebook,
                format: trellis.format,
            },
        }
    }

    pub fn optimize_weights_layout<E>(
        &mut self,
        optimize: impl FnOnce(&mut B::GlobalGpuBuffer) -> Result<(WeightsLayout<B>, bool), E>,
    ) -> Result<(), E> {
        let Encoding::Quantized(quantized) = &mut self.encoding else {
            return Ok(());
        };
        if quantized.weights_layout != WeightsLayout::<B>::default() {
            return Ok(());
        }
        (quantized.weights_layout, quantized.signed_codes) = optimize(&mut self.values)?;
        Ok(())
    }

    fn quantized(&self) -> Option<&Quantized<B>> {
        match &self.encoding {
            Encoding::Quantized(quantized) => Some(quantized),
            Encoding::Dense
            | Encoding::Trellis {
                ..
            } => None,
        }
    }
}
