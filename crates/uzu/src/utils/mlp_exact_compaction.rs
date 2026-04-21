use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::BufReader,
    os::unix::fs::FileExt,
    path::{Component, Path, PathBuf},
};

use bytemuck::{Pod, try_cast_slice};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    MLPConfig, ModelMetadata,
    parameters::{
        SafetensorsDtype, SafetensorsMetadata, SafetensorsTensorInfo, SafetensorsWriteError, TensorData,
        read_safetensors_metadata, write_safetensors,
    },
};

const WEIGHTS_FILE_NAME: &str = "model.safetensors";
const CONFIG_FILE_NAME: &str = "config.json";
const MANIFEST_FILE_NAME: &str = "mlp_exact_compaction.json";

#[derive(Debug, Error)]
pub enum ExactMlpCompactionError {
    #[error("Failed to read model metadata")]
    ModelMetadata(#[source] serde_json::Error),
    #[error("Failed to read safetensors metadata")]
    SafetensorsMetadata(#[source] crate::parameters::HeaderLoadingError),
    #[error("Failed to write compacted safetensors")]
    SafetensorsWrite(#[from] SafetensorsWriteError),
    #[error("Unsupported model type: only language models are supported")]
    UnsupportedModelType,
    #[error("Output directory `{0}` already exists")]
    OutputAlreadyExists(PathBuf),
    #[error("Output directory `{output}` must not be inside input model directory `{input}`")]
    OutputInsideInput {
        input: PathBuf,
        output: PathBuf,
    },
    #[error("Missing tensor `{0}`")]
    MissingTensor(String),
    #[error(
        "Tensor `{name}` has unsupported dtype {dtype:?}; exact compaction only supports bf16, f16, and f32 source tensors"
    )]
    UnsupportedTensorDtype {
        name: String,
        dtype: SafetensorsDtype,
    },
    #[error("Tensor `{name}` is malformed: {reason}")]
    InvalidTensor {
        name: String,
        reason: String,
    },
    #[error("Failed to copy model files")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExactMlpCompactionManifest {
    pub source_model_path: String,
    pub scanned_dense_layers: usize,
    pub compacted_layers: Vec<ExactMlpCompactionLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExactMlpCompactionLayer {
    pub layer_index: usize,
    pub tensor_prefix: String,
    pub original_hidden_dimension: usize,
    pub kept_hidden_dimension: usize,
    pub removed_hidden_indices: Vec<usize>,
    pub zero_up_channels: usize,
    pub zero_gate_channels: usize,
    pub zero_down_channels: usize,
}

pub fn compact_model_directory(
    input_model_path: &Path,
    output_model_path: &Path,
) -> Result<ExactMlpCompactionManifest, ExactMlpCompactionError> {
    let input_model_path = input_model_path.canonicalize()?;
    let output_model_path = absolute_path(output_model_path)?;
    if output_model_path.starts_with(&input_model_path) {
        return Err(ExactMlpCompactionError::OutputInsideInput {
            input: input_model_path,
            output: output_model_path,
        });
    }
    if output_model_path.exists() {
        return Err(ExactMlpCompactionError::OutputAlreadyExists(output_model_path.to_path_buf()));
    }

    let config_file = File::open(input_model_path.join(CONFIG_FILE_NAME))?;
    let model_metadata: ModelMetadata =
        serde_json::from_reader(BufReader::new(config_file)).map_err(ExactMlpCompactionError::ModelMetadata)?;
    let language_model =
        model_metadata.model_config.as_language_model().ok_or(ExactMlpCompactionError::UnsupportedModelType)?;
    let layer_configs = &language_model.model_config.transformer_config.layer_configs;

    let weights_path = input_model_path.join(WEIGHTS_FILE_NAME);
    let weights_file = File::open(&weights_path)?;
    let (data_offset, safetensors_metadata) =
        read_safetensors_metadata(&weights_file).map_err(ExactMlpCompactionError::SafetensorsMetadata)?;

    let mut rewrites = HashMap::new();
    let mut compacted_layers = Vec::new();
    let mut scanned_dense_layers = 0;
    for (layer_index, layer_config) in layer_configs.iter().enumerate() {
        if !matches!(layer_config.mlp_config, MLPConfig::Dense(_)) {
            continue;
        }
        scanned_dense_layers += 1;
        if let Some((manifest, layer_rewrites)) =
            compact_dense_layer(&weights_file, data_offset, &safetensors_metadata, layer_index)?
        {
            compacted_layers.push(manifest);
            rewrites.extend(layer_rewrites);
        }
    }

    copy_model_directory(input_model_path, output_model_path)?;

    let mut tensors = BTreeMap::new();
    let mut tensor_names = safetensors_metadata.tensors.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    for tensor_name in tensor_names {
        let tensor = if let Some(tensor) = rewrites.remove(&tensor_name) {
            tensor
        } else {
            let info = tensor_info(&safetensors_metadata, &tensor_name)?;
            TensorData {
                dtype: info.dtype,
                shape: info.shape.clone(),
                bytes: read_tensor_bytes(&weights_file, data_offset, info)?,
            }
        };
        tensors.insert(tensor_name, tensor);
    }

    write_safetensors(&output_model_path.join(WEIGHTS_FILE_NAME), safetensors_metadata.metadata.clone(), &tensors)?;

    let manifest = ExactMlpCompactionManifest {
        source_model_path: input_model_path.display().to_string(),
        scanned_dense_layers,
        compacted_layers,
    };
    fs::write(
        output_model_path.join(MANIFEST_FILE_NAME),
        serde_json::to_vec_pretty(&manifest).map_err(ExactMlpCompactionError::ModelMetadata)?,
    )?;
    Ok(manifest)
}

fn absolute_path(path: &Path) -> Result<PathBuf, std::io::Error> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut absolute = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => absolute.push(prefix.as_os_str()),
            Component::RootDir => absolute.push(component.as_os_str()),
            Component::CurDir => {},
            Component::ParentDir => {
                absolute.pop();
            },
            Component::Normal(component) => absolute.push(component),
        }
    }
    Ok(absolute)
}

fn compact_dense_layer(
    weights_file: &File,
    data_offset: usize,
    metadata: &SafetensorsMetadata,
    layer_index: usize,
) -> Result<Option<(ExactMlpCompactionLayer, HashMap<String, TensorData>)>, ExactMlpCompactionError> {
    let prefix = format!("transformer.layers.{layer_index}.mlp");
    let up_weights_key = format!("{prefix}.up_projection.weights");
    let up_bias_key = format!("{prefix}.up_projection.biases");
    let down_weights_key = format!("{prefix}.down_projection.weights");

    let up_weights_info = tensor_info(metadata, &up_weights_key)?;
    let down_weights_info = tensor_info(metadata, &down_weights_key)?;
    let up_weights = read_tensor_bytes(weights_file, data_offset, up_weights_info)?;
    let down_weights = read_tensor_bytes(weights_file, data_offset, down_weights_info)?;
    let up_bias = metadata
        .tensors
        .get(&up_bias_key)
        .map(|info| read_tensor_bytes(weights_file, data_offset, info))
        .transpose()?;

    let detection = detect_dead_channels(
        &up_weights_key,
        up_weights_info,
        &up_weights,
        up_bias.as_deref(),
        &down_weights_key,
        down_weights_info,
        &down_weights,
    )?;
    if detection.removed_hidden_indices.is_empty() {
        return Ok(None);
    }

    let kept_rows = detection
        .kept_hidden_indices
        .iter()
        .copied()
        .chain(detection.kept_hidden_indices.iter().map(|&index| detection.original_hidden_dimension + index))
        .collect::<Vec<_>>();
    let mut rewrites = HashMap::new();
    rewrites.insert(
        up_weights_key,
        TensorData {
            dtype: up_weights_info.dtype,
            shape: vec![2 * detection.kept_hidden_indices.len(), up_weights_info.shape[1]],
            bytes: slice_rows(
                &up_weights_key,
                up_weights_info.dtype,
                &up_weights,
                up_weights_info.shape[1],
                &kept_rows,
            )?,
        },
    );
    if let Some(up_bias) = up_bias {
        rewrites.insert(
            up_bias_key,
            TensorData {
                dtype: up_weights_info.dtype,
                shape: vec![2 * detection.kept_hidden_indices.len()],
                bytes: slice_rows(
                    &format!("{prefix}.up_projection.biases"),
                    up_weights_info.dtype,
                    &up_bias,
                    1,
                    &kept_rows,
                )?,
            },
        );
    }
    rewrites.insert(
        down_weights_key,
        TensorData {
            dtype: down_weights_info.dtype,
            shape: vec![down_weights_info.shape[0], detection.kept_hidden_indices.len()],
            bytes: slice_columns(
                &format!("{prefix}.down_projection.weights"),
                down_weights_info.dtype,
                &down_weights,
                down_weights_info.shape[0],
                down_weights_info.shape[1],
                &detection.kept_hidden_indices,
            )?,
        },
    );

    Ok(Some((
        ExactMlpCompactionLayer {
            layer_index,
            tensor_prefix: prefix,
            original_hidden_dimension: detection.original_hidden_dimension,
            kept_hidden_dimension: detection.kept_hidden_indices.len(),
            removed_hidden_indices: detection.removed_hidden_indices,
            zero_up_channels: detection.zero_up_channels,
            zero_gate_channels: detection.zero_gate_channels,
            zero_down_channels: detection.zero_down_channels,
        },
        rewrites,
    )))
}

fn tensor_info<'a>(
    metadata: &'a SafetensorsMetadata,
    name: &str,
) -> Result<&'a SafetensorsTensorInfo, ExactMlpCompactionError> {
    metadata.tensors.get(name).ok_or_else(|| ExactMlpCompactionError::MissingTensor(name.to_string()))
}

fn read_tensor_bytes(
    file: &File,
    data_offset: usize,
    info: &SafetensorsTensorInfo,
) -> Result<Vec<u8>, ExactMlpCompactionError> {
    let mut bytes = vec![0; info.data_offsets.1 - info.data_offsets.0];
    file.read_exact_at(&mut bytes, (data_offset + info.data_offsets.0) as u64)?;
    Ok(bytes)
}

#[derive(Debug, PartialEq, Eq)]
struct DetectionResult {
    original_hidden_dimension: usize,
    kept_hidden_indices: Vec<usize>,
    removed_hidden_indices: Vec<usize>,
    zero_up_channels: usize,
    zero_gate_channels: usize,
    zero_down_channels: usize,
}

fn detect_dead_channels(
    up_weights_name: &str,
    up_weights_info: &SafetensorsTensorInfo,
    up_weights_bytes: &[u8],
    up_bias_bytes: Option<&[u8]>,
    down_weights_name: &str,
    down_weights_info: &SafetensorsTensorInfo,
    down_weights_bytes: &[u8],
) -> Result<DetectionResult, ExactMlpCompactionError> {
    if up_weights_info.shape.len() != 2 {
        return Err(ExactMlpCompactionError::InvalidTensor {
            name: up_weights_name.to_string(),
            reason: format!("expected a 2D tensor, got {:?}", up_weights_info.shape),
        });
    }
    if up_weights_info.shape[0] % 2 != 0 {
        return Err(ExactMlpCompactionError::InvalidTensor {
            name: up_weights_name.to_string(),
            reason: format!("expected an even first dimension, got {:?}", up_weights_info.shape),
        });
    }

    let hidden_dimension = up_weights_info.shape[0] / 2;
    if down_weights_info.shape != [up_weights_info.shape[1], hidden_dimension] {
        return Err(ExactMlpCompactionError::InvalidTensor {
            name: down_weights_name.to_string(),
            reason: format!(
                "expected shape [{}, {}], got {:?}",
                up_weights_info.shape[1], hidden_dimension, down_weights_info.shape,
            ),
        });
    }

    let up_weights = FloatTensor::from_bytes(up_weights_name, up_weights_info.dtype, up_weights_bytes)?;
    let up_bias = match up_bias_bytes {
        Some(bytes) => {
            Some(FloatTensor::from_bytes(&format!("{up_weights_name}.biases"), up_weights_info.dtype, bytes)?)
        },
        None => None,
    };
    let down_weights = FloatTensor::from_bytes(down_weights_name, down_weights_info.dtype, down_weights_bytes)?;

    let mut kept_hidden_indices = Vec::with_capacity(hidden_dimension);
    let mut removed_hidden_indices = Vec::new();
    let mut zero_up_channels = 0;
    let mut zero_gate_channels = 0;
    let mut zero_down_channels = 0;
    for hidden_index in 0..hidden_dimension {
        let up_zero = up_weights.row_is_zero(hidden_index, up_weights_info.shape[1])
            && up_bias.as_ref().is_none_or(|bias| bias.value_is_zero(hidden_index));
        let gate_zero = up_weights.row_is_zero(hidden_dimension + hidden_index, up_weights_info.shape[1])
            && up_bias.as_ref().is_none_or(|bias| bias.value_is_zero(hidden_dimension + hidden_index));
        let down_zero =
            down_weights.column_is_zero(down_weights_info.shape[0], down_weights_info.shape[1], hidden_index);
        zero_up_channels += usize::from(up_zero);
        zero_gate_channels += usize::from(gate_zero);
        zero_down_channels += usize::from(down_zero);
        if up_zero || gate_zero || down_zero {
            removed_hidden_indices.push(hidden_index);
        } else {
            kept_hidden_indices.push(hidden_index);
        }
    }

    Ok(DetectionResult {
        original_hidden_dimension: hidden_dimension,
        kept_hidden_indices,
        removed_hidden_indices,
        zero_up_channels,
        zero_gate_channels,
        zero_down_channels,
    })
}

trait ExactFloat: Pod + Copy {
    fn is_zero(self) -> bool;
}

impl ExactFloat for bf16 {
    fn is_zero(self) -> bool {
        self.to_bits() & 0x7fff == 0
    }
}

impl ExactFloat for f16 {
    fn is_zero(self) -> bool {
        self.to_bits() & 0x7fff == 0
    }
}

impl ExactFloat for f32 {
    fn is_zero(self) -> bool {
        self.to_bits() & 0x7fff_ffff == 0
    }
}

enum FloatTensor<'a> {
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
}

impl<'a> FloatTensor<'a> {
    fn from_bytes(
        name: &str,
        dtype: SafetensorsDtype,
        bytes: &'a [u8],
    ) -> Result<Self, ExactMlpCompactionError> {
        match dtype {
            SafetensorsDtype::BF16 => {
                Ok(Self::BF16(try_cast_slice(bytes).map_err(|_| ExactMlpCompactionError::InvalidTensor {
                    name: name.to_string(),
                    reason: "failed to decode bf16 tensor bytes".to_string(),
                })?))
            },
            SafetensorsDtype::F16 => {
                Ok(Self::F16(try_cast_slice(bytes).map_err(|_| ExactMlpCompactionError::InvalidTensor {
                    name: name.to_string(),
                    reason: "failed to decode f16 tensor bytes".to_string(),
                })?))
            },
            SafetensorsDtype::F32 => {
                Ok(Self::F32(try_cast_slice(bytes).map_err(|_| ExactMlpCompactionError::InvalidTensor {
                    name: name.to_string(),
                    reason: "failed to decode f32 tensor bytes".to_string(),
                })?))
            },
            dtype => Err(ExactMlpCompactionError::UnsupportedTensorDtype {
                name: name.to_string(),
                dtype,
            }),
        }
    }

    fn row_is_zero(
        &self,
        row_index: usize,
        width: usize,
    ) -> bool {
        match self {
            Self::BF16(values) => row_is_zero(values, row_index, width),
            Self::F16(values) => row_is_zero(values, row_index, width),
            Self::F32(values) => row_is_zero(values, row_index, width),
        }
    }

    fn column_is_zero(
        &self,
        rows: usize,
        cols: usize,
        column_index: usize,
    ) -> bool {
        match self {
            Self::BF16(values) => column_is_zero(values, rows, cols, column_index),
            Self::F16(values) => column_is_zero(values, rows, cols, column_index),
            Self::F32(values) => column_is_zero(values, rows, cols, column_index),
        }
    }

    fn value_is_zero(
        &self,
        index: usize,
    ) -> bool {
        match self {
            Self::BF16(values) => values[index].is_zero(),
            Self::F16(values) => values[index].is_zero(),
            Self::F32(values) => values[index].is_zero(),
        }
    }
}

fn row_is_zero<T: ExactFloat>(
    values: &[T],
    row_index: usize,
    width: usize,
) -> bool {
    values[row_index * width..(row_index + 1) * width].iter().copied().all(ExactFloat::is_zero)
}

fn column_is_zero<T: ExactFloat>(
    values: &[T],
    rows: usize,
    cols: usize,
    column_index: usize,
) -> bool {
    (0..rows).all(|row_index| values[row_index * cols + column_index].is_zero())
}

fn slice_rows(
    name: &str,
    dtype: SafetensorsDtype,
    bytes: &[u8],
    width: usize,
    selected_rows: &[usize],
) -> Result<Vec<u8>, ExactMlpCompactionError> {
    match FloatTensor::from_bytes(name, dtype, bytes)? {
        FloatTensor::BF16(values) => Ok(slice_rows_t(values, width, selected_rows)),
        FloatTensor::F16(values) => Ok(slice_rows_t(values, width, selected_rows)),
        FloatTensor::F32(values) => Ok(slice_rows_t(values, width, selected_rows)),
    }
}

fn slice_columns(
    name: &str,
    dtype: SafetensorsDtype,
    bytes: &[u8],
    rows: usize,
    cols: usize,
    selected_columns: &[usize],
) -> Result<Vec<u8>, ExactMlpCompactionError> {
    match FloatTensor::from_bytes(name, dtype, bytes)? {
        FloatTensor::BF16(values) => Ok(slice_columns_t(values, rows, cols, selected_columns)),
        FloatTensor::F16(values) => Ok(slice_columns_t(values, rows, cols, selected_columns)),
        FloatTensor::F32(values) => Ok(slice_columns_t(values, rows, cols, selected_columns)),
    }
}

fn slice_rows_t<T: Pod + Copy>(
    values: &[T],
    width: usize,
    selected_rows: &[usize],
) -> Vec<u8> {
    let mut compact = Vec::with_capacity(selected_rows.len() * width);
    for &row_index in selected_rows {
        compact.extend_from_slice(&values[row_index * width..(row_index + 1) * width]);
    }
    bytemuck::cast_slice(&compact).to_vec()
}

fn slice_columns_t<T: Pod + Copy>(
    values: &[T],
    rows: usize,
    cols: usize,
    selected_columns: &[usize],
) -> Vec<u8> {
    let mut compact = Vec::with_capacity(rows * selected_columns.len());
    for row_index in 0..rows {
        for &column_index in selected_columns {
            compact.push(values[row_index * cols + column_index]);
        }
    }
    bytemuck::cast_slice(&compact).to_vec()
}

fn copy_model_directory(
    input: &Path,
    output: &Path,
) -> Result<(), ExactMlpCompactionError> {
    fs::create_dir_all(output)?;
    copy_directory_contents(input, output)
}

fn copy_directory_contents(
    input: &Path,
    output: &Path,
) -> Result<(), ExactMlpCompactionError> {
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let input_path = entry.path();
        let output_path = output.join(entry.file_name());
        if input_path.is_dir() {
            fs::create_dir_all(&output_path)?;
            copy_directory_contents(&input_path, &output_path)?;
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == WEIGHTS_FILE_NAME || name == MANIFEST_FILE_NAME {
            continue;
        }
        fs::copy(input_path, output_path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs, fs::File};

    use serde_json::json;
    use tempfile::tempdir;

    use super::{
        ExactMlpCompactionLayer, ExactMlpCompactionManifest, MANIFEST_FILE_NAME, WEIGHTS_FILE_NAME,
        compact_model_directory, detect_dead_channels,
    };
    use crate::parameters::{SafetensorsDtype, TensorData, read_safetensors_metadata, write_safetensors};

    #[test]
    fn detect_dead_channels_uses_up_gate_and_down_certificates() {
        let up_weights = [
            0.0f32, 0.0, // up channel 0
            1.0, 2.0, // up channel 1
            3.0, 4.0, // up channel 2
            5.0, 6.0, // gate channel 0
            0.0, 0.0, // gate channel 1
            7.0, 8.0, // gate channel 2
        ];
        let up_biases = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let down_weights = [9.0f32, 10.0, 0.0, 11.0, 12.0, 0.0];
        let detection = detect_dead_channels(
            "up",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![6, 2],
                data_offsets: (0, up_weights.len() * 4),
            },
            bytemuck::cast_slice(&up_weights),
            Some(bytemuck::cast_slice(&up_biases)),
            "down",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![2, 3],
                data_offsets: (0, down_weights.len() * 4),
            },
            bytemuck::cast_slice(&down_weights),
        )
        .unwrap();
        assert_eq!(detection.kept_hidden_indices, vec![2]);
        assert_eq!(detection.removed_hidden_indices, vec![0, 1]);
        assert_eq!(detection.zero_up_channels, 1);
        assert_eq!(detection.zero_gate_channels, 1);
        assert_eq!(detection.zero_down_channels, 1);
    }

    #[test]
    fn detect_dead_channels_requires_zero_bias_for_zero_row_certificate() {
        let up_weights = [0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up_biases = [1.0f32, 0.0, 0.0, 0.0];
        let down_weights = [7.0f32, 8.0, 9.0, 10.0];
        let detection = detect_dead_channels(
            "up",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![4, 2],
                data_offsets: (0, up_weights.len() * 4),
            },
            bytemuck::cast_slice(&up_weights),
            Some(bytemuck::cast_slice(&up_biases)),
            "down",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![2, 2],
                data_offsets: (0, down_weights.len() * 4),
            },
            bytemuck::cast_slice(&down_weights),
        )
        .unwrap();
        assert_eq!(detection.kept_hidden_indices, vec![0, 1]);
        assert!(detection.removed_hidden_indices.is_empty());
    }

    #[test]
    fn compact_model_directory_rewrites_dense_mlp_tensors_and_keeps_other_files() {
        let input = tempdir().unwrap();
        let output = tempdir().unwrap();
        let output_model = output.path().join("compacted");

        fs::write(
            input.path().join("config.json"),
            serde_json::to_vec_pretty(&json!({
                "toolchain_version": "test",
                "vendor": "mirai",
                "family": "test",
                "name": "tiny",
                "size": "tiny",
                "quantization": null,
                "repo": "test",
                "use_cases": ["test"],
                "model_type": "language_model",
                "model_config": {
                    "model_config": {
                        "embedding_config": {
                            "type": "TiedEmbeddingConfig",
                            "input_scale": null,
                            "logit_soft_cap": null,
                            "precision": "float32"
                        },
                        "transformer_config": {
                            "global_rope_config": null,
                            "local_rope_config": null,
                            "layer_configs": [{
                                "pre_attention_norm_config": null,
                                "mixer_config": {
                                    "type": "AttentionConfig",
                                    "qkv_projection_config": {"type": "FullPrecisionLinearConfig", "precision": "float32"},
                                    "out_projection_config": {"type": "FullPrecisionLinearConfig", "precision": "float32"},
                                    "query_norm_config": null,
                                    "key_norm_config": null,
                                    "num_heads": 1,
                                    "num_groups": 1,
                                    "head_dim": 4,
                                    "is_causal": true,
                                    "scale": null,
                                    "sliding_window_size": null,
                                    "logit_soft_cap": null,
                                    "has_sinks": false,
                                    "has_qkv_biases": false,
                                    "has_out_biases": false
                                },
                                "post_attention_norm_config": null,
                                "pre_mlp_norm_config": {
                                    "scale_precision": "float32",
                                    "accumulation_precision": "float32",
                                    "epsilon": 1e-5,
                                    "scale_offset": null,
                                    "upcast_mode": "only_normalization",
                                    "subtract_mean": false
                                },
                                "mlp_config": {
                                    "type": "DenseMLPConfig",
                                    "linear_config": {"type": "FullPrecisionLinearConfig", "precision": "float32"},
                                    "activation": {"type": "SiLU"},
                                    "has_up_biases": true,
                                    "has_down_biases": false,
                                    "gate_clipping": null,
                                    "up_clipping": null,
                                    "activation_to_gate": true
                                },
                                "post_mlp_norm_config": null
                            }],
                            "output_norm_config": {
                                "scale_precision": "float32",
                                "accumulation_precision": "float32",
                                "epsilon": 1e-5,
                                "scale_offset": null,
                                "upcast_mode": "only_normalization",
                                "subtract_mean": false
                            },
                            "model_dim": 2,
                            "hidden_dim": 3,
                            "num_heads": 1,
                            "num_groups": 1,
                            "head_dim": 4,
                            "attention_scale": null,
                            "num_layers": 1,
                            "context_length": 32
                        },
                        "vocab_size": 16
                    },
                    "message_processor_config": {
                        "prompt_template": "{{ messages }}",
                        "output_parser_regex": null,
                        "system_role_name": "system",
                        "user_role_name": "user",
                        "assistant_role_name": "assistant",
                        "bos_token": null
                    },
                    "generation_config": {
                        "stop_token_ids": [],
                        "temperature": null,
                        "top_k": null,
                        "top_p": null,
                        "min_p": null,
                        "banned_tokens": null
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(input.path().join("tokenizer.json"), b"{}" as &[u8]).unwrap();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "transformer.layers.0.mlp.up_projection.weights".to_string(),
            TensorData {
                dtype: SafetensorsDtype::F32,
                shape: vec![6, 2],
                bytes: bytemuck::cast_slice(&[0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0]).to_vec(),
            },
        );
        tensors.insert(
            "transformer.layers.0.mlp.up_projection.biases".to_string(),
            TensorData {
                dtype: SafetensorsDtype::F32,
                shape: vec![6],
                bytes: bytemuck::cast_slice(&[0.0f32, 1.0, 2.0, 0.0, 0.0, 3.0]).to_vec(),
            },
        );
        tensors.insert(
            "transformer.layers.0.mlp.down_projection.weights".to_string(),
            TensorData {
                dtype: SafetensorsDtype::F32,
                shape: vec![2, 3],
                bytes: bytemuck::cast_slice(&[9.0f32, 10.0, 0.0, 11.0, 12.0, 0.0]).to_vec(),
            },
        );
        tensors.insert(
            "embedding.weights".to_string(),
            TensorData {
                dtype: SafetensorsDtype::F32,
                shape: vec![2, 2],
                bytes: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec(),
            },
        );
        write_safetensors(&input.path().join(WEIGHTS_FILE_NAME), None, &tensors).unwrap();

        let manifest = compact_model_directory(input.path(), &output_model).unwrap();
        assert_eq!(
            manifest,
            ExactMlpCompactionManifest {
                source_model_path: input.path().display().to_string(),
                scanned_dense_layers: 1,
                compacted_layers: vec![ExactMlpCompactionLayer {
                    layer_index: 0,
                    tensor_prefix: "transformer.layers.0.mlp".to_string(),
                    original_hidden_dimension: 3,
                    kept_hidden_dimension: 1,
                    removed_hidden_indices: vec![0, 2],
                    zero_up_channels: 1,
                    zero_gate_channels: 1,
                    zero_down_channels: 1,
                }],
            }
        );
        assert!(output_model.join("tokenizer.json").exists());
        assert!(output_model.join(MANIFEST_FILE_NAME).exists());

        let weights_file = File::open(output_model.join(WEIGHTS_FILE_NAME)).unwrap();
        let (_offset, metadata) = read_safetensors_metadata(&weights_file).unwrap();
        assert_eq!(metadata.tensors["transformer.layers.0.mlp.up_projection.weights"].shape, vec![2, 2]);
        assert_eq!(metadata.tensors["transformer.layers.0.mlp.up_projection.biases"].shape, vec![2]);
        assert_eq!(metadata.tensors["transformer.layers.0.mlp.down_projection.weights"].shape, vec![2, 1]);
        assert_eq!(metadata.tensors["embedding.weights"].shape, vec![2, 2]);
    }

    #[test]
    fn compact_model_directory_rejects_nested_output_directory() {
        let input = tempdir().unwrap();
        fs::write(input.path().join("config.json"), b"{}" as &[u8]).unwrap();
        let error = compact_model_directory(input.path(), &input.path().join("nested").join("..").join("compacted"))
            .unwrap_err();
        assert!(matches!(error, super::ExactMlpCompactionError::OutputInsideInput { .. }));
    }

    #[test]
    fn compacted_dense_mlp_matches_original_outputs_exactly() {
        let up_weights = [
            0.0f32, 0.0, // up channel 0
            1.0, 2.0, // up channel 1
            3.0, 4.0, // up channel 2
            5.0, 6.0, // gate channel 0
            0.0, 0.0, // gate channel 1
            7.0, 8.0, // gate channel 2
        ];
        let up_biases = [0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let down_weights = [9.0f32, 10.0, 0.0, 11.0, 12.0, 0.0];

        let detection = detect_dead_channels(
            "up",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![6, 2],
                data_offsets: (0, up_weights.len() * 4),
            },
            bytemuck::cast_slice(&up_weights),
            Some(bytemuck::cast_slice(&up_biases)),
            "down",
            &crate::parameters::SafetensorsTensorInfo {
                dtype: SafetensorsDtype::F32,
                shape: vec![2, 3],
                data_offsets: (0, down_weights.len() * 4),
            },
            bytemuck::cast_slice(&down_weights),
        )
        .unwrap();

        let kept_rows = detection
            .kept_hidden_indices
            .iter()
            .copied()
            .chain(detection.kept_hidden_indices.iter().map(|&index| detection.original_hidden_dimension + index))
            .collect::<Vec<_>>();
        let compact_up_weights = bytemuck::cast_slice::<u8, f32>(
            &slice_rows("up", SafetensorsDtype::F32, bytemuck::cast_slice(&up_weights), 2, &kept_rows).unwrap(),
        )
        .to_vec();
        let compact_up_biases = bytemuck::cast_slice::<u8, f32>(
            &slice_rows("up_biases", SafetensorsDtype::F32, bytemuck::cast_slice(&up_biases), 1, &kept_rows).unwrap(),
        )
        .to_vec();
        let compact_down_weights = bytemuck::cast_slice::<u8, f32>(
            &slice_columns(
                "down",
                SafetensorsDtype::F32,
                bytemuck::cast_slice(&down_weights),
                2,
                3,
                &detection.kept_hidden_indices,
            )
            .unwrap(),
        )
        .to_vec();

        for input in [[1.0f32, -1.0], [0.5, 2.0], [-3.0, 4.0]] {
            assert_eq!(
                dense_mlp_output(&up_weights, &up_biases, &down_weights, 3, &input),
                dense_mlp_output(
                    &compact_up_weights,
                    &compact_up_biases,
                    &compact_down_weights,
                    detection.kept_hidden_indices.len(),
                    &input,
                ),
            );
        }
    }

    fn dense_mlp_output(
        up_weights: &[f32],
        up_biases: &[f32],
        down_weights: &[f32],
        hidden_dimension: usize,
        input: &[f32; 2],
    ) -> Vec<f32> {
        let hidden = (0..hidden_dimension)
            .map(|hidden_index| {
                let up = dot(&up_weights[hidden_index * 2..(hidden_index + 1) * 2], input) + up_biases[hidden_index];
                let gate = dot(
                    &up_weights[(hidden_dimension + hidden_index) * 2..(hidden_dimension + hidden_index + 1) * 2],
                    input,
                ) + up_biases[hidden_dimension + hidden_index];
                silu(gate) * up
            })
            .collect::<Vec<_>>();
        (0..2)
            .map(|row_index| {
                dot(&down_weights[row_index * hidden_dimension..(row_index + 1) * hidden_dimension], &hidden)
            })
            .collect()
    }

    fn dot(
        lhs: &[f32],
        rhs: &[f32],
    ) -> f32 {
        lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}
