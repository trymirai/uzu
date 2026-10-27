use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, BufferMut, BufferRef, Context,
        gpu_types::{quantization::QuantizationMode, quantization_method::QuantizationMethod},
        kernel::matmul::{MatmulB, QuantParams, QuantParamsLayout, QuantizedB, QuantizedCorrection, WeightsLayout},
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout, mlx_spec::MLXSpec},
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

pub struct WeightMatrix<B: Backend> {
    values: B::GlobalBuffer,
    quantized: Option<Quantized<B>>,
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
        if let AnyWeightMatrixSpec::MLXSpec(MLXSpec {
            bits: 2,
            group_size,
            layout: Layout::OutputInput,
            ..
        }) = spec
            && required_layout == Layout::OutputInput
        {
            return Self::load_mlx_2bit_as_int4(output_dim, input_dim, group_size, data_type, tree);
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
        })
    }

    /// Dash stopgap: 2-bit -> int4 repack at load; replace with a native 2-bit kernel.
    /// MLX 2-bit codes (4 per byte, LSB first) become int4 nibbles (2 per byte, low first) with the same values 0..3,
    /// and each source group's scale/bias is copied into every int4 group it covers, so the int4 kernels run as is.
    fn load_mlx_2bit_as_int4(
        rows: u32,
        columns: u32,
        group_size: u32,
        data_type: DataType,
        tree: &ParameterTree<B>,
    ) -> Result<Self, WeightMatrixError<B>> {
        let int4_group_size = if group_size.is_multiple_of(64) {
            64
        } else {
            32
        };
        if !group_size.is_multiple_of(int4_group_size) || !columns.is_multiple_of(group_size) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "MLX bits=2 needs group_size ({group_size}) to be a multiple of 32 dividing columns ({columns})"
            )));
        }
        let context = tree.context();

        let codes = tree.read_buffer("weights", &[rows, columns / 4], DataType::U8)?;
        let mut nibbles =
            context.create_buffer((rows * columns / 2) as usize).map_err(ParameterLoaderError::BackendError)?;
        for (pair, &byte) in nibbles.as_slice_mut::<u8>().chunks_exact_mut(2).zip(codes.as_slice::<u8>()) {
            let code = |index: u8| (byte >> (2 * index)) & 0b11;
            pair[0] = code(0) | (code(1) << 4);
            pair[1] = code(2) | (code(3) << 4);
        }

        let source_params = QuantParams::new(QuantParamsLayout::GroupOutput, rows, columns / group_size);
        let params = QuantParams::new(QuantParamsLayout::GroupOutput, rows, columns / int4_group_size);
        let replicate_groups = |name: &str| -> Result<B::GlobalBuffer, WeightMatrixError<B>> {
            let source = tree.read_buffer(name, &source_params.scale_shape(), data_type)?;
            let source = source.as_slice::<u8>();
            let group_bytes = source.len() / (columns / group_size) as usize;
            let mut target = context
                .create_buffer(group_bytes * (columns / int4_group_size) as usize)
                .map_err(ParameterLoaderError::BackendError)?;
            for (group, target_group) in target.as_slice_mut::<u8>().chunks_exact_mut(group_bytes).enumerate() {
                let source_group = group * int4_group_size as usize / group_size as usize;
                target_group.copy_from_slice(&source[source_group * group_bytes..][..group_bytes]);
            }
            Ok(target)
        };

        Ok(Self {
            values: nibbles,
            quantized: Some(Quantized {
                scales: replicate_groups("scales")?,
                correction: QuantizedCorrection::Biases(replicate_groups("biases")?),
                params,
                info: QuantizationInfo {
                    mode: QuantizationMode::U4,
                    method: QuantizationMethod::ScaleBias,
                    group_size: int4_group_size,
                },
                signed_codes: false,
                weights_layout: Default::default(),
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
