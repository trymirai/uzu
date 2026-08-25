use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulB,
        microfloat::{MicrofloatAxisOrder, MicrofloatEncoding, MicrofloatFormat, MicrofloatMetadata},
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

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
    pub microfloat: Option<MicrofloatEncoding>,
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
            };
            let group_size = u32::try_from(*group_size).map_err(|_| {
                WeightMatrixError::UnsupportedConfiguration(format!("microfloat group size {group_size} exceeds u32"))
            })?;
            let axis_order = match layout {
                Layout::OutputInput => MicrofloatAxisOrder::OutputInput,
                Layout::InputOutput => MicrofloatAxisOrder::InputOutput,
            };
            let encoding = MicrofloatEncoding::new(format, *bits, group_size, axis_order)
                .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            (layout.clone(), None, Some(encoding))
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
    outer_scales: Allocation<B>,
    metadata: MicrofloatMetadata,
}

enum WeightStorage<B: Backend> {
    FullPrecision {
        values: Allocation<B>,
    },
    Integer {
        values: Allocation<B>,
        quantized: Quantized<B>,
    },
    Microfloat {
        values: Allocation<B>,
        microfloat: Microfloat<B>,
    },
}

pub struct WeightMatrix<B: Backend> {
    storage: WeightStorage<B>,
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
            microfloat,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        if let Some(encoding) = microfloat {
            let metadata = MicrofloatMetadata::new(encoding, rows, columns)
                .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            let group_size = encoding.group_size();
            let values = tree.leaf("weights")?.validate(&[rows, columns / 2], DataType::U8)?.read_allocation()?;
            let scales =
                tree.leaf("scales")?.validate(&[rows, columns / group_size], DataType::U8)?.read_allocation()?;
            // The artifact retains its established tensor name; at runtime this is
            // the one outer scale applied after block-scale decoding.
            let outer_scales = tree.leaf("global_scale")?.validate(&[1], data_type)?.read_allocation()?;
            return Ok(Self {
                storage: WeightStorage::Microfloat {
                    values,
                    microfloat: Microfloat {
                        scales,
                        outer_scales,
                        metadata,
                    },
                },
            });
        }

        let Some(info) = quantization else {
            let values = tree.leaf("weights")?.validate(&[rows, columns], data_type)?.read_allocation()?;
            return Ok(Self {
                storage: WeightStorage::FullPrecision {
                    values,
                },
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
        let values =
            tree.leaf("weights")?.validate(&[rows, columns / packing_divisor], storage_data_type)?.read_allocation()?;
        let scales = tree.leaf("scales")?.validate(&[rows, groups], data_type)?.read_allocation()?;
        let correction = match info.method {
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
            storage: WeightStorage::Integer {
                values,
                quantized: Quantized {
                    scales,
                    correction,
                    info,
                    signed_codes: false,
                },
            },
        })
    }

    pub fn values(&self) -> &Allocation<B> {
        match &self.storage {
            WeightStorage::FullPrecision {
                values,
            }
            | WeightStorage::Integer {
                values,
                ..
            }
            | WeightStorage::Microfloat {
                values,
                ..
            } => values,
        }
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        match &self.storage {
            WeightStorage::Integer {
                quantized,
                ..
            } => Some(quantized.info),
            WeightStorage::FullPrecision {
                ..
            }
            | WeightStorage::Microfloat {
                ..
            } => None,
        }
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        match &self.storage {
            WeightStorage::Integer {
                quantized,
                ..
            } => Some(&quantized.scales),
            WeightStorage::Microfloat {
                microfloat,
                ..
            } => Some(&microfloat.scales),
            WeightStorage::FullPrecision {
                ..
            } => None,
        }
    }

    pub fn zero_points(&self) -> Option<&Allocation<B>> {
        match &self.storage {
            WeightStorage::Integer {
                quantized:
                    Quantized {
                        correction: QuantizedCorrection::ZeroPoints(zero_points),
                        ..
                    },
                ..
            } => Some(zero_points),
            WeightStorage::FullPrecision {
                ..
            }
            | WeightStorage::Integer {
                ..
            }
            | WeightStorage::Microfloat {
                ..
            } => None,
        }
    }

    pub fn biases(&self) -> Option<&Allocation<B>> {
        match &self.storage {
            WeightStorage::Integer {
                quantized:
                    Quantized {
                        correction: QuantizedCorrection::Biases(biases),
                        ..
                    },
                ..
            } => Some(biases),
            WeightStorage::FullPrecision {
                ..
            }
            | WeightStorage::Integer {
                ..
            }
            | WeightStorage::Microfloat {
                ..
            } => None,
        }
    }

    pub fn matmul_b(&self) -> MatmulB<'_, B> {
        let (values, quantized) = match &self.storage {
            WeightStorage::FullPrecision {
                values,
            } => {
                return MatmulB::FullPrecision {
                    b: values,
                };
            },
            WeightStorage::Microfloat {
                values,
                microfloat,
            } => {
                return MatmulB::Microfloat {
                    codes: values,
                    scales: &microfloat.scales,
                    outer_scales: &microfloat.outer_scales,
                    metadata: microfloat.metadata,
                };
            },
            WeightStorage::Integer {
                values,
                quantized,
            } => (values, quantized),
        };
        let mode = quantized.info.mode;
        let group_size = quantized.info.group_size;
        let signed_codes = quantized.signed_codes;
        match &quantized.correction {
            QuantizedCorrection::Biases(biases) => MatmulB::ScaleBiasDequant {
                b: values,
                scales: &quantized.scales,
                biases,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::ZeroPoints(zero_points) => MatmulB::ScaleZeroPointDequant {
                b: values,
                scales: &quantized.scales,
                zero_points,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::Symmetric => MatmulB::ScaleSymmetricDequant {
                b: values,
                scales: &quantized.scales,
                mode,
                group_size,
                signed_codes,
            },
        }
    }

    pub fn make_codes_signed(&mut self) {
        let WeightStorage::Integer {
            values,
            quantized,
        } = &mut self.storage
        else {
            return;
        };
        if quantized.signed_codes {
            return;
        }
        let Some(sign_flip_mask) = quantized.info.mode.weight_codes_sign_flip_mask() else {
            return;
        };
        let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
        let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(values.as_slice_mut());
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
mod tests {
    use std::io::Write;

    use half::bf16;
    use serde_json::{Map, Value, json};
    use tempfile::NamedTempFile;
    use uzu_engine_macros::uzu_test;

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
    fn loads_dense_mxfp4_storage() {
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
        assert_eq!(metadata.rows(), 2);
        assert_eq!(metadata.columns(), 32);
        tree.assert_all_tensors_validated().expect("validate dense MXFP4 tensors");
    }
}
