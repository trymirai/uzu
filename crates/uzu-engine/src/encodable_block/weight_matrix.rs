use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Context, CpuBuffer, GlobalBufferCpuGpu,
        gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
        kernel::{
            activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
            matmul::{
                MatmulB, QuantParams, QuantParamsLayout, QuantParamsStrides, QuantizedB, QuantizedCorrection,
                WeightsLayout,
            },
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
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
                2 => QuantizationMode::U2,
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

/// Row-major (`[N, G]`) copy of the scales and zero points/biases for gathered reads: a gathered row then touches one
/// contiguous run of metadata instead of one sector per group.
struct OutputGroupParams<B: Backend> {
    scales: B::GlobalGpuBuffer,
    correction: QuantizedCorrection<B::GlobalGpuBuffer>,
    params: QuantParams,
}

struct Quantized<B: Backend> {
    scales: B::GlobalGpuBuffer,
    correction: QuantizedCorrection<B::GlobalGpuBuffer>,
    params: QuantParams,
    info: QuantizationInfo,
    signed_codes: bool,
    weights_layout: WeightsLayout<B>,
    output_group: Option<OutputGroupParams<B>>,
}

pub struct WeightMatrix<B: Backend> {
    values: B::GlobalGpuBuffer,
    quantized: Option<Quantized<B>>,
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

        let Some(mut info) = quantization_info else {
            let values = tree.read_buffer("weights", &[rows, columns], data_type).await?;
            return Ok(Self {
                values,
                quantized: None,
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
        if info.mode == QuantizationMode::U2 {
            info.group_size = ACTIVATION_SCALE_GROUP_SIZE;
        }

        let values = tree.read_buffer("weights", &[rows, columns / packing_divisor], storage_data_type).await?;
        let params = QuantParams::new(params_layout, rows, groups);
        let load_plane = async |name: &str,
                                shape: [u32; 2],
                                storage_type: DataType|
               -> Result<B::GlobalGpuBuffer, WeightMatrixError<B>> {
            Ok(tree.read_buffer(name, &shape, storage_type).await?)
        };
        let scales = load_plane("scales", params.scale_shape(), data_type).await?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => {
                QuantizedCorrection::Biases(load_plane("biases", params.scale_shape(), data_type).await?)
            },
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(
                load_plane("zero_points", params.zero_point_shape(info.mode), info.mode.storage_type()).await?,
            ),
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
                output_group: None,
            }),
        })
    }

    pub fn values(&self) -> &B::GlobalGpuBuffer {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized.as_ref().map(|quantized| &quantized.scales)
    }

    pub fn zero_points(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized.as_ref()?.correction.zero_points()
    }

    pub fn biases(&self) -> Option<&B::GlobalGpuBuffer> {
        self.quantized.as_ref()?.correction.biases()
    }

    pub fn matmul_b(&self) -> MatmulB<&B::GlobalGpuBuffer> {
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

    /// `matmul_b` for gathered reads: the `[N, G]` metadata copy when `add_output_group_params` made one.
    pub fn gather_matmul_b(&self) -> MatmulB<&B::GlobalGpuBuffer> {
        let MatmulB::Quantized(mut quantized) = self.matmul_b() else {
            return self.matmul_b();
        };
        if let Some(copy) = self.quantized.as_ref().and_then(|quantized| quantized.output_group.as_ref()) {
            quantized.scales = &copy.scales;
            quantized.correction = copy.correction.as_ref();
            quantized.params = copy.params;
        }
        MatmulB::Quantized(quantized)
    }

    /// Load-time: transposes the `[G, N]` scales and zero points/biases into an `[N, G]` copy for gathered reads.
    pub async fn add_output_group_params(
        &mut self,
        context: &B::Context,
    ) -> Result<(), B::Error> {
        let Some(quantized) = self.quantized.as_mut().filter(|quantized| {
            quantized.params.layout() == QuantParamsLayout::GroupOutput && !quantized.params.is_per_row()
        }) else {
            return Ok(());
        };
        let mode = quantized.info.mode;
        let source = quantized.params;
        let (rows, groups) = (source.row_count() as usize, source.group_count() as usize);
        let destination = QuantParams::new(QuantParamsLayout::OutputGroup, source.row_count(), source.group_count());
        let transpose = async |buffer: &B::GlobalGpuBuffer,
                               bits: u32,
                               source_strides: QuantParamsStrides,
                               destination_strides: QuantParamsStrides,
                               destination_bytes: usize|
               -> Result<B::GlobalGpuBuffer, B::Error> {
            let values: Vec<u8> = context.copyout(buffer).await?;
            let mut output = context.create_buffer_cpu_gpu(destination_bytes)?;
            let bytes = output.as_slice_mut::<u8>();
            bytes.fill(0);
            let read = |index: usize| -> u8 {
                if bits == 4 {
                    (values[index / 2] >> (4 * (index % 2))) & 0xF
                } else {
                    values[index]
                }
            };
            for row in 0..rows {
                for group in 0..groups {
                    let from =
                        row * source_strides.output_stride as usize + group * source_strides.group_stride as usize;
                    let to = row * destination_strides.output_stride as usize
                        + group * destination_strides.group_stride as usize;
                    if bits == 4 {
                        bytes[to / 2] |= read(from) << (4 * (to % 2));
                    } else {
                        bytes[2 * to..2 * to + 2].copy_from_slice(&values[2 * from..2 * from + 2]);
                    }
                }
            }
            Ok(output.handoff())
        };
        let scale_bytes = destination.scale_shape().into_iter().product::<u32>() as usize * 2;
        let scales =
            transpose(&quantized.scales, 16, source.scale_strides(), destination.scale_strides(), scale_bytes).await?;
        let correction = match &quantized.correction {
            QuantizedCorrection::Symmetric => QuantizedCorrection::Symmetric,
            QuantizedCorrection::Biases(biases) => QuantizedCorrection::Biases(
                transpose(biases, 16, source.scale_strides(), destination.scale_strides(), scale_bytes).await?,
            ),
            QuantizedCorrection::ZeroPoints(zero_points) => QuantizedCorrection::ZeroPoints(
                transpose(
                    zero_points,
                    4,
                    source.zero_point_strides(mode),
                    destination.zero_point_strides(mode),
                    destination.zero_point_shape(mode).into_iter().product::<u32>() as usize,
                )
                .await?,
            ),
        };
        quantized.output_group = Some(OutputGroupParams {
            scales,
            correction,
            params: destination,
        });
        Ok(())
    }

    pub fn optimize_weights_layout<E>(
        &mut self,
        optimize: impl FnOnce(&mut B::GlobalGpuBuffer) -> Result<(WeightsLayout<B>, bool), E>,
    ) -> Result<(), E> {
        let Some(quantized) =
            self.quantized.as_mut().filter(|quantized| quantized.weights_layout == WeightsLayout::<B>::default())
        else {
            return Ok(());
        };
        (quantized.weights_layout, quantized.signed_codes) = optimize(&mut self.values)?;
        Ok(())
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
