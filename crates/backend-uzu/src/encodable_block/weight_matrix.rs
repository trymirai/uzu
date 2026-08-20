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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use half::bf16;
    use proc_macros::uzu_test;
    use serde_json::{Map, Value, json};
    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        backends::{common::Context, cpu::Cpu},
        parameters::ParameterLoader,
    };

    fn add_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[u32],
        data_type: &str,
        data: &[u8],
    ) {
        let begin = payload.len();
        payload.extend_from_slice(data);
        header.insert(
            name.into(),
            json!({
                "dtype": data_type,
                "shape": shape,
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    /// Pristine dense MXFP4 artifact at the `WeightMatrix` loading boundary.
    fn dense_microfloat_parameter_file() -> NamedTempFile {
        const ROWS: u32 = 2;
        const COLUMNS: u32 = 32;
        const GROUP_SIZE: u32 = 16;

        let mut header = Map::new();
        header.insert(
            "__metadata__".into(),
            json!({
                "spec": json!({
                    "type": "MicrofloatSpec",
                    "bits": 4,
                    "group_size": GROUP_SIZE,
                    "scale_mode": "mxfp4",
                    "layout": "output_input"
                }).to_string()
            }),
        );
        let mut payload = Vec::new();
        let codes: Vec<u8> = (0..ROWS * COLUMNS / 2).map(|index| 0x10 | (index % 8) as u8).collect();
        let scales: Vec<u8> = (0..ROWS * COLUMNS / GROUP_SIZE).map(|index| 126 + (index % 3) as u8).collect();
        let global_scale = bf16::from_f32(0.75).to_le_bytes();
        add_tensor(&mut header, &mut payload, "weights", &[ROWS, COLUMNS / 2], "U8", &codes);
        add_tensor(&mut header, &mut payload, "scales", &[ROWS, COLUMNS / GROUP_SIZE], "U8", &scales);
        add_tensor(&mut header, &mut payload, "global_scale", &[1], "BF16", &global_scale);

        let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
        header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
        let mut file = NamedTempFile::new().expect("create dense MXFP4 fixture");
        file.write_all(&(header.len() as u64).to_le_bytes()).expect("write safetensors header length");
        file.write_all(&header).expect("write safetensors header");
        file.write_all(&payload).expect("write safetensors payload");
        file
    }

    #[uzu_test]
    fn loads_dense_microfloat_without_expert_routes() {
        let context = <Cpu as Backend>::Context::new().expect("create CPU context");
        let file = dense_microfloat_parameter_file();
        let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load dense MXFP4 fixture");
        let tree = loader.tree();
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec").expect("load dense MXFP4 spec");

        let matrix = WeightMatrix::load(&tree, spec, Layout::OutputInput, 2, 32, DataType::BF16)
            .expect("load dense MXFP4 matrix");
        let MatmulB::Microfloat {
            metadata,
            ..
        } = matrix.matmul_b()
        else {
            panic!("dense MXFP4 matrix did not preserve its operand format");
        };
        assert_eq!(metadata.matrix_count(), 1);
        assert_eq!(metadata.rows(), 2);
        assert_eq!(metadata.columns(), 32);
        tree.assert_all_tensors_validated().expect("validate dense MXFP4 tensors");
    }
}
