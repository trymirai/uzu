use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    DataType,
    parameters::{Dtype, HashMetadata, TensorInfo},
};

use super::torch_checkpoint::{TorchCheckpoint, TorchCheckpointError, TorchModule, TorchTensorSpec};

#[derive(Debug, Error)]
pub enum NemoNanoCodecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Torch checkpoint error: {0}")]
    Torch(#[from] TorchCheckpointError),
    #[error("Missing {0} in .nemo archive")]
    MissingArchiveMember(&'static str),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NemoVectorQuantizerConfig {
    pub num_groups: usize,
    pub num_levels_per_group: Vec<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NemoAudioEncoderConfig {
    pub down_sample_rates: Vec<usize>,
    pub encoded_dim: usize,
    pub base_channels: usize,
    pub activation: String,
    pub pad_mode: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NemoAudioDecoderConfig {
    pub up_sample_rates: Vec<usize>,
    pub input_dim: usize,
    pub base_channels: usize,
    pub activation: String,
    pub output_activation: String,
    pub pad_mode: String,
    pub n_groups_equal_to_out_channels: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NemoNanoCodecConfig {
    pub sample_rate: usize,
    pub samples_per_frame: usize,
    pub vector_quantizer: NemoVectorQuantizerConfig,
    pub audio_encoder: NemoAudioEncoderConfig,
    pub audio_decoder: NemoAudioDecoderConfig,
}

#[derive(Debug, Clone)]
pub struct NemoNanoCodecExport {
    pub export_dir: PathBuf,
    pub config_json: PathBuf,
    pub audio_encoder_safetensors: PathBuf,
    pub audio_decoder_safetensors: PathBuf,
    pub vector_quantizer_safetensors: PathBuf,
}

pub fn export_nanocodec_from_nemo(
    nemo_path: &Path,
    export_dir: &Path,
) -> Result<(NemoNanoCodecConfig, NemoNanoCodecExport), NemoNanoCodecError> {
    std::fs::create_dir_all(export_dir)?;

    let config_json = export_dir.join("nanocodec_config.json");
    let audio_encoder_safetensors = export_dir.join("audio_encoder.safetensors");
    let audio_decoder_safetensors = export_dir.join("audio_decoder.safetensors");
    let vector_quantizer_safetensors = export_dir.join("vector_quantizer.safetensors");

    let nemo_file = File::open(nemo_path)?;
    let mut archive = tar::Archive::new(nemo_file);

    let mut model_config_yaml: Option<Vec<u8>> = None;
    let mut ckpt_tmp = tempfile::NamedTempFile::new()?;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path = path.to_string_lossy();
        if path.ends_with("model_config.yaml") {
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes)?;
            model_config_yaml = Some(bytes);
        } else if path.ends_with("model_weights.ckpt") {
            std::io::copy(&mut entry, &mut ckpt_tmp)?;
        }
    }

    let yaml_bytes = model_config_yaml.ok_or(NemoNanoCodecError::MissingArchiveMember("model_config.yaml"))?;
    let config: NemoNanoCodecConfig = serde_yaml::from_slice(&yaml_bytes)?;
    std::fs::write(&config_json, serde_json::to_vec_pretty(&config)?)?;

    ckpt_tmp.flush()?;

    let mut ckpt = TorchCheckpoint::open_from_path(ckpt_tmp.path())?;

    // Clone the spec maps so we can mutably read tensors from the zip archive.
    let audio_encoder = ckpt.state_dict.audio_encoder.clone();
    let audio_decoder = ckpt.state_dict.audio_decoder.clone();
    let vector_quantizer = ckpt.state_dict.vector_quantizer.clone();

    write_safetensors_from_torch_map(&audio_encoder_safetensors, &mut ckpt, TorchModule::AudioEncoder, &audio_encoder)?;
    write_safetensors_from_torch_map(&audio_decoder_safetensors, &mut ckpt, TorchModule::AudioDecoder, &audio_decoder)?;
    write_safetensors_from_torch_map(
        &vector_quantizer_safetensors,
        &mut ckpt,
        TorchModule::VectorQuantizer,
        &vector_quantizer,
    )?;

    Ok((
        config,
        NemoNanoCodecExport {
            export_dir: export_dir.to_path_buf(),
            config_json,
            audio_encoder_safetensors,
            audio_decoder_safetensors,
            vector_quantizer_safetensors,
        },
    ))
}

fn write_safetensors_from_torch_map(
    path: &Path,
    ckpt: &mut TorchCheckpoint<File>,
    _module: TorchModule,
    tensors: &HashMap<String, TorchTensorSpec>,
) -> Result<(), NemoNanoCodecError> {
    let mut entries: Vec<(&String, &TorchTensorSpec)> = tensors.iter().collect();
    entries.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));

    let mut offset: usize = 0;
    let mut header = HashMetadata {
        metadata: None,
        tensors: HashMap::new(),
    };

    for (name, spec) in entries.iter() {
        let dtype: DataType = spec.dtype.to_data_type();
        let elem_bytes = dtype.size_in_bytes();
        let numel: usize = spec.shape.iter().product();
        let byte_len = numel.saturating_mul(elem_bytes);

        let begin = offset;
        let end = offset + byte_len;
        offset = end;

        header.tensors.insert(
            (*name).clone(),
            TensorInfo {
                dtype: Dtype::from(dtype),
                shape: spec.shape.to_vec(),
                data_offsets: (begin, end),
            },
        );
    }

    let mut header_bytes = serde_json::to_vec(&header)?;
    let padding = (8 - (header_bytes.len() % 8)) % 8;
    if padding != 0 {
        header_bytes.extend(std::iter::repeat(b' ').take(padding));
    }

    let header_len: u64 = header_bytes.len().try_into().expect("header too large for u64");

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;

    for (name, spec) in entries {
        let tensor = ckpt.load_tensor_from_spec(name, spec)?;
        w.write_all(&tensor.data)?;
    }
    w.flush()?;
    Ok(())
}
