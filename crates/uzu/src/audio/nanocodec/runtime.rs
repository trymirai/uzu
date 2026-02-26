use std::{collections::HashMap, fs::File, os::unix::fs::FileExt, path::Path, rc::Rc};

use half::{bf16, f16};
use serde::Deserialize;

use super::{
    decoder::{
        CausalConv1dJson, CausalConvTranspose1dJson, NanoCodecDecoderGraph, NanoCodecDecoderJson,
        NanoCodecHiFiGanResBlockJson, NanoCodecHiFiGanResLayerJson, NanoCodecResidualBlockJson,
        NanoCodecUpsampleStageJson, Tensor3Json,
    },
    fsq::compute_dim_base_index,
};
use crate::{
    DataType,
    array::ArrayContextExt,
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking},
    backends::{
        common::{
            Backend, CommandBuffer, Context, Kernels,
            kernel::{AudioFsqDecodeKernel, AudioFsqEncodeKernel},
        },
        metal::Metal,
    },
    parameters::read_safetensors_metadata,
};

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

fn usize_to_i32(
    value: usize,
    name: &str,
) -> AudioResult<i32> {
    i32::try_from(value).map_err(|_| AudioError::Runtime(format!("{name} exceeds i32 range")))
}

fn convert_lengths_to_i32(
    lengths: &[usize],
    frames: usize,
) -> AudioResult<Vec<i32>> {
    let mut out = Vec::with_capacity(lengths.len());
    for &length in lengths {
        if length > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length,
                frames,
            });
        }
        out.push(usize_to_i32(length, "length")?);
    }
    Ok(out)
}

fn checked_mul_i32(
    value: i32,
    mul: usize,
) -> AudioResult<i32> {
    i32::try_from(mul)
        .map_err(|_| AudioError::Runtime("length scaling factor exceeds i32 range".to_string()))?
        .checked_mul(value)
        .ok_or(AudioError::Runtime("scaled length overflow".to_string()))
}

fn pack_pcm_to_padded(
    pcm: &AudioPcmBatch,
    expected_channels: usize,
) -> AudioResult<(Vec<f32>, Vec<usize>, Vec<i32>, usize)> {
    if pcm.channels() != expected_channels {
        return Err(AudioError::Runtime(format!(
            "pcm channel mismatch: expected {expected_channels}, got {}",
            pcm.channels()
        )));
    }

    let lengths = pcm.lengths().to_vec();
    let frames = lengths.iter().copied().max().unwrap_or(0);
    let lengths_i32 = convert_lengths_to_i32(&lengths, frames)?;
    let padded_len = checked_product(&[pcm.batch_size(), expected_channels, frames])?;
    let mut padded = vec![0.0_f32; padded_len];

    let mut src_offset = 0usize;
    let samples = pcm.samples();
    for (batch, &frame_count) in lengths.iter().enumerate() {
        let sample_count = frame_count
            .checked_mul(expected_channels)
            .ok_or(AudioError::Runtime("pcm frame-count overflow".to_string()))?;
        let src_end =
            src_offset.checked_add(sample_count).ok_or(AudioError::Runtime("pcm indexing overflow".to_string()))?;
        if src_end > samples.len() {
            return Err(AudioError::Runtime("pcm indexing out of bounds".to_string()));
        }

        for frame in 0..frame_count {
            let src_frame_base = src_offset + frame * expected_channels;
            for channel in 0..expected_channels {
                let dst_index = (batch * expected_channels + channel) * frames + frame;
                padded[dst_index] = samples[src_frame_base + channel];
            }
        }

        src_offset = src_end;
    }

    Ok((padded, lengths, lengths_i32, frames))
}

fn unpack_padded_to_pcm(
    padded: &[f32],
    batch_size: usize,
    channels: usize,
    frames: usize,
    lengths: &[usize],
) -> AudioResult<Vec<f32>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }

    let expected_padded = checked_product(&[batch_size, channels, frames])?;
    if padded.len() != expected_padded {
        return Err(AudioError::InvalidPcmShape {
            expected_samples: expected_padded,
            actual_samples: padded.len(),
        });
    }

    let total_frames = lengths.iter().sum::<usize>();
    let packed_len = checked_product(&[total_frames, channels])?;
    let mut packed = vec![0.0_f32; packed_len];
    let mut dst_offset = 0usize;

    for (batch, &frame_count) in lengths.iter().enumerate() {
        if frame_count > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length: frame_count,
                frames,
            });
        }

        for frame in 0..frame_count {
            for channel in 0..channels {
                let src_index = (batch * channels + channel) * frames + frame;
                let dst_index = dst_offset + frame * channels + channel;
                packed[dst_index] = padded[src_index];
            }
        }
        dst_offset = dst_offset
            .checked_add(
                frame_count
                    .checked_mul(channels)
                    .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?,
            )
            .ok_or(AudioError::Runtime("pcm destination indexing overflow".to_string()))?;
    }

    Ok(packed)
}

fn default_eps() -> f32 {
    1e-3
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimePacking {
    #[default]
    FrameMajor,
    CodebookMajor,
}

impl From<RuntimePacking> for AudioTokenPacking {
    fn from(value: RuntimePacking) -> Self {
        match value {
            RuntimePacking::FrameMajor => AudioTokenPacking::FrameMajor,
            RuntimePacking::CodebookMajor => AudioTokenPacking::CodebookMajor,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeConfigJson {
    #[serde(default)]
    r#type: Option<String>,
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
    #[serde(default)]
    output_packing: RuntimePacking,
    #[serde(default)]
    decoder: Option<NanoCodecDecoderJson>,
}

fn parse_runtime_config_json(tts_config: &serde_json::Value) -> AudioResult<RuntimeConfigJson> {
    let candidate = tts_config.get("audio_codec").unwrap_or(tts_config);
    let parsed: RuntimeConfigJson = serde_json::from_value(candidate.clone())
        .map_err(|err| AudioError::Runtime(format!("invalid tts audio codec config: {err}")))?;

    if let Some(ref codec_type) = parsed.r#type {
        if codec_type != "nanocodec_fsq" {
            return Err(AudioError::Runtime(format!(
                "unsupported audio codec type '{codec_type}', expected 'nanocodec_fsq'"
            )));
        }
    }

    Ok(parsed)
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoTtsConfigJson {
    audio_decoder_config: LalamoAudioDecoderConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoAudioDecoderConfigJson {
    samplerate: u32,
    quantizer_config: LalamoGroupedQuantizerConfigJson,
    decoder_config: LalamoDecoderConfigJson,
    base_channels: usize,
    up_sample_rates: Vec<usize>,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilations: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoGroupedQuantizerConfigJson {
    num_groups: usize,
    quantizer_config: LalamoQuantizerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoQuantizerConfigJson {
    num_levels: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoDecoderConfigJson {
    activation_config: LalamoActivationConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct LalamoActivationConfigJson {
    #[serde(default = "default_negative_slope")]
    leaky_relu_negative_slope: f32,
}

fn default_negative_slope() -> f32 {
    0.01
}

fn default_decoder_eps() -> f32 {
    1e-9
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioTtsConfigJson {
    audio_decoder_config: FishAudioAudioDecoderConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioAudioDecoderConfigJson {
    samplerate: u32,
    n_codebooks: usize,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    downsample_factor: Vec<usize>,
    decoder_rates: Vec<usize>,
    quantizer_config: FishAudioQuantizerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioQuantizerConfigJson {
    post_module_config: crate::config::TransformerConfig,
    upsampler_config: FishAudioUpsamplerConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioUpsamplerConfigJson {
    block_configs: Vec<FishAudioUpsamplingBlockConfigJson>,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioUpsamplingBlockConfigJson {
    convnext_config: FishAudioConvNeXtConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioConvNeXtConfigJson {
    norm_config: FishAudioConvNeXtNormConfigJson,
}

#[derive(Debug, Clone, Deserialize)]
struct FishAudioConvNeXtNormConfigJson {
    epsilon: f32,
    #[serde(default)]
    subtract_mean: bool,
    #[serde(default)]
    use_bias: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct MatrixF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl MatrixF32 {
    fn row(
        &self,
        index: usize,
    ) -> Option<&[f32]> {
        if index >= self.rows {
            return None;
        }
        let start = index.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.values.get(start..end)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioVectorQuantizer {
    codebook: MatrixF32,
    out_proj: MatrixF32,
    out_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConv1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConvTranspose1dLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioNormLayer {
    scales: Vec<f32>,
    biases: Option<Vec<f32>>,
    epsilon: f32,
    subtract_mean: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioConvNeXtLayer {
    depthwise_conv: FishAudioConv1dLayer,
    norm: FishAudioNormLayer,
    pwconv1: MatrixF32,
    pwconv1_bias: Vec<f32>,
    pwconv2: MatrixF32,
    pwconv2_bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioResidualUnitLayer {
    snake1_alpha: Vec<f32>,
    conv1: FishAudioConv1dLayer,
    snake2_alpha: Vec<f32>,
    conv2: FishAudioConv1dLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioDecoderBlockLayer {
    snake_alpha: Vec<f32>,
    trans_conv: FishAudioConvTranspose1dLayer,
    res_unit1: FishAudioResidualUnitLayer,
    res_unit2: FishAudioResidualUnitLayer,
    res_unit3: FishAudioResidualUnitLayer,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioDecoderGraph {
    first_conv: FishAudioConv1dLayer,
    upsample_blocks: Vec<(FishAudioConvTranspose1dLayer, FishAudioConvNeXtLayer)>,
    decoder_blocks: Vec<FishAudioDecoderBlockLayer>,
    final_snake_alpha: Vec<f32>,
    final_conv: FishAudioConv1dLayer,
    upsample_factor: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioPostModuleLayer {
    pre_mixer_norm: FishAudioNormLayer,
    qkv_projection: MatrixF32,
    out_projection: MatrixF32,
    pre_mlp_norm: FishAudioNormLayer,
    up_projection: MatrixF32,
    down_projection: MatrixF32,
    num_heads: usize,
    num_groups: usize,
    head_dim: usize,
    attention_scale: f32,
    sliding_window_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioPostModule {
    rope_cosines: MatrixF32,
    rope_sines: MatrixF32,
    layers: Vec<FishAudioPostModuleLayer>,
    output_norm: FishAudioNormLayer,
    model_dim: usize,
    hidden_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FishAudioCodecGraph {
    semantic_quantizer: FishAudioVectorQuantizer,
    residual_quantizers: Vec<FishAudioVectorQuantizer>,
    post_module: FishAudioPostModule,
    decoder: FishAudioDecoderGraph,
    codebook_size: usize,
    semantic_codebook_size: usize,
    input_dim: usize,
    total_codebooks: usize,
    upsample_factor: usize,
}

#[derive(Debug, Clone)]
struct SafeTensorEntry {
    data_type: DataType,
    shape: Vec<usize>,
    offset: usize,
    size: usize,
}

#[derive(Debug)]
struct SafeTensorReader {
    file: File,
    entries: HashMap<String, SafeTensorEntry>,
}

impl SafeTensorReader {
    fn open(path: &Path) -> AudioResult<Self> {
        let file = File::open(path).map_err(|err| {
            AudioError::Runtime(format!("failed to open safetensors file '{}': {err}", path.display()))
        })?;
        let (global_offset, metadata) = read_safetensors_metadata(&file).map_err(|err| {
            AudioError::Runtime(format!("failed to parse safetensors metadata from '{}': {err}", path.display()))
        })?;

        let mut entries = HashMap::new();
        for (name, tensor) in metadata.tensors {
            let (local_begin, local_end) = tensor.data_offsets;
            let size = local_end
                .checked_sub(local_begin)
                .ok_or(AudioError::Runtime("invalid tensor data offsets in safetensors metadata".to_string()))?;
            let offset = global_offset
                .checked_add(local_begin)
                .ok_or(AudioError::Runtime("safetensors tensor offset overflow".to_string()))?;

            entries.insert(
                name,
                SafeTensorEntry {
                    data_type: tensor.dtype.into(),
                    shape: tensor.shape,
                    offset,
                    size,
                },
            );
        }

        Ok(Self {
            file,
            entries,
        })
    }

    fn read_tensor_f32(
        &self,
        key: &str,
    ) -> AudioResult<(Vec<usize>, Vec<f32>)> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| AudioError::Runtime(format!("missing tensor '{key}' in model.safetensors")))?;

        let num_elements = checked_product(&entry.shape)?;
        let expected_size = num_elements
            .checked_mul(entry.data_type.size_in_bytes())
            .ok_or(AudioError::Runtime(format!("tensor '{key}' byte-size overflow")))?;
        if entry.size != expected_size {
            return Err(AudioError::Runtime(format!(
                "tensor '{key}' size mismatch: expected {expected_size} bytes from shape {:?} and dtype {:?}, got {}",
                entry.shape, entry.data_type, entry.size
            )));
        }

        let mut bytes = vec![0_u8; entry.size];
        self.file.read_exact_at(&mut bytes, entry.offset as u64).map_err(|err| {
            AudioError::Runtime(format!("failed reading tensor '{key}' from model.safetensors: {err}"))
        })?;

        let values = match entry.data_type {
            DataType::F32 => decode_f32_bytes(&bytes),
            DataType::F16 => decode_f16_bytes(&bytes).into_iter().map(f16::to_f32).collect::<Vec<f32>>(),
            DataType::BF16 => decode_bf16_bytes(&bytes).into_iter().map(bf16::to_f32).collect::<Vec<f32>>(),
            other => {
                return Err(AudioError::Runtime(format!(
                    "unsupported tensor dtype for '{key}': {other:?} (expected F32/F16/BF16)"
                )));
            },
        };

        if values.len() != num_elements {
            return Err(AudioError::Runtime(format!(
                "decoded tensor '{key}' element count mismatch: expected {num_elements}, got {}",
                values.len()
            )));
        }

        Ok((entry.shape.clone(), values))
    }
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn decode_f16_bytes(bytes: &[u8]) -> Vec<f16> {
    bytes.chunks_exact(2).map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn decode_bf16_bytes(bytes: &[u8]) -> Vec<bf16> {
    bytes.chunks_exact(2).map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))).collect()
}

fn read_tensor_1d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Vec<f32>> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 1 {
        return Err(AudioError::Runtime(format!("expected rank-1 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(values)
}

fn read_tensor_3d(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<Tensor3Json> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(Tensor3Json {
        shape: [shape[0], shape[1], shape[2]],
        values,
    })
}

fn read_causal_conv_1d(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<CausalConv1dJson> {
    Ok(CausalConv1dJson {
        weight: read_tensor_3d(reader, &format!("{prefix}.weights"))?,
        bias: read_tensor_1d(reader, &format!("{prefix}.biases"))?,
        dilation,
    })
}

fn convert_lalamo_transpose_weight_oih_to_iog(
    weight: &Tensor3Json,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
) -> AudioResult<Tensor3Json> {
    let [out_channels_in_weight, in_channels_per_group, kernel_size] = weight.shape;
    if out_channels_in_weight != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv out-channel mismatch: expected {out_channels}, got {out_channels_in_weight}"
        )));
    }
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid transpose conv grouping: in_channels={in_channels}, out_channels={out_channels}, groups={groups}"
        )));
    }

    let expected_in_per_group = in_channels / groups;
    if in_channels_per_group != expected_in_per_group {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight shape mismatch: expected in_per_group={expected_in_per_group}, got {in_channels_per_group}"
        )));
    }

    let out_channels_per_group = out_channels / groups;
    let expected_weight_len = checked_product(&[out_channels, in_channels_per_group, kernel_size])?;
    if weight.values.len() != expected_weight_len {
        return Err(AudioError::Runtime(format!(
            "transpose conv weight value length mismatch: expected {expected_weight_len}, got {}",
            weight.values.len()
        )));
    }

    let converted_len = checked_product(&[in_channels, out_channels_per_group, kernel_size])?;
    let mut converted = vec![0.0_f32; converted_len];

    for group in 0..groups {
        let in_base = group * in_channels_per_group;
        let out_base = group * out_channels_per_group;

        for out_idx in 0..out_channels_per_group {
            for in_idx in 0..in_channels_per_group {
                for k in 0..kernel_size {
                    let src_index = ((out_base + out_idx) * in_channels_per_group + in_idx) * kernel_size + k;
                    let dst_index = ((in_base + in_idx) * out_channels_per_group + out_idx) * kernel_size + k;
                    converted[dst_index] = weight.values[src_index];
                }
            }
        }
    }

    Ok(Tensor3Json {
        shape: [in_channels, out_channels_per_group, kernel_size],
        values: converted,
    })
}

fn parse_lalamo_tts_config_json(tts_config: &serde_json::Value) -> AudioResult<LalamoTtsConfigJson> {
    serde_json::from_value(tts_config.clone())
        .map_err(|err| AudioError::Runtime(format!("invalid Lalamo tts_config for NanoCodec runtime import: {err}")))
}

fn build_decoder_json_from_lalamo_export(
    tts_config: &LalamoTtsConfigJson,
    model_weights_path: &Path,
) -> AudioResult<NanoCodecDecoderJson> {
    let cfg = &tts_config.audio_decoder_config;
    let reader = SafeTensorReader::open(model_weights_path)?;

    let pre_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.pre_conv", 1)?;
    let mut stages = Vec::with_capacity(cfg.up_sample_rates.len());

    let mut current_channels = cfg.base_channels;
    for (stage_index, &stride) in cfg.up_sample_rates.iter().enumerate() {
        if stride == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid upsample stride at stage {stage_index}: stride must be > 0"
            )));
        }
        let out_channels = current_channels
            .checked_div(2)
            .ok_or(AudioError::Runtime("decoder channel progression overflow".to_string()))?;
        if out_channels == 0 {
            return Err(AudioError::Runtime(format!(
                "invalid decoder stage {stage_index}: current_channels={current_channels}, expected > 1"
            )));
        }
        let groups = out_channels;

        let activation_alpha =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.activations.{stage_index}.snake.alpha"))?;

        let upsample_weight_oih =
            read_tensor_3d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.weights"))?;
        let upsample_bias =
            read_tensor_1d(&reader, &format!("audio_decoder.decoder.upsample_convs.{stage_index}.biases"))?;
        let upsample_weight =
            convert_lalamo_transpose_weight_oih_to_iog(&upsample_weight_oih, current_channels, out_channels, groups)?;
        let upsample_conv = CausalConvTranspose1dJson {
            weight: upsample_weight,
            bias: upsample_bias,
            stride,
            groups,
        };

        let mut res_blocks = Vec::with_capacity(cfg.resblock_kernel_sizes.len());
        for (res_block_index, _) in cfg.resblock_kernel_sizes.iter().enumerate() {
            let mut residuals = Vec::with_capacity(cfg.resblock_dilations.len());
            for (residual_index, &dilation) in cfg.resblock_dilations.iter().enumerate() {
                let prefix = format!(
                    "audio_decoder.decoder.res_layers.{stage_index}.res_blocks.{res_block_index}.res_blocks.{residual_index}"
                );
                residuals.push(NanoCodecResidualBlockJson {
                    input_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.input_activation.snake.alpha"))?,
                    input_conv: read_causal_conv_1d(&reader, &format!("{prefix}.input_conv"), dilation)?,
                    skip_activation_alpha: read_tensor_1d(&reader, &format!("{prefix}.skip_activation.snake.alpha"))?,
                    skip_conv: read_causal_conv_1d(&reader, &format!("{prefix}.skip_conv"), 1)?,
                });
            }
            res_blocks.push(NanoCodecHiFiGanResBlockJson {
                res_blocks: residuals,
            });
        }

        stages.push(NanoCodecUpsampleStageJson {
            activation_alpha,
            upsample_conv,
            res_layer: Some(NanoCodecHiFiGanResLayerJson {
                res_blocks,
            }),
        });

        current_channels = out_channels;
    }

    let post_activation_alpha = read_tensor_1d(&reader, "audio_decoder.decoder.post_activation.snake.alpha")?;
    let post_conv = read_causal_conv_1d(&reader, "audio_decoder.decoder.post_conv", 1)?;

    Ok(NanoCodecDecoderJson {
        pre_conv,
        stages,
        post_activation_alpha,
        post_conv,
        negative_slope: cfg.decoder_config.activation_config.leaky_relu_negative_slope,
        eps: default_decoder_eps(),
    })
}

fn parse_fishaudio_tts_config_json(tts_config: &serde_json::Value) -> AudioResult<FishAudioTtsConfigJson> {
    serde_json::from_value(tts_config.clone()).map_err(|err| {
        AudioError::Runtime(format!("invalid Lalamo tts_config for FishAudio DAC runtime import: {err}"))
    })
}

fn read_matrix_f32(
    reader: &SafeTensorReader,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(AudioError::Runtime(format!(
            "tensor '{key}' shape mismatch: expected [{expected_rows}, {expected_cols}], got {shape:?}"
        )));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_matrix_f32_any(
    reader: &SafeTensorReader,
    key: &str,
) -> AudioResult<MatrixF32> {
    let (shape, values) = reader.read_tensor_f32(key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{key}', got shape {shape:?}")));
    }
    Ok(MatrixF32 {
        rows: shape[0],
        cols: shape[1],
        values,
    })
}

fn read_fishaudio_vector_quantizer(
    reader: &SafeTensorReader,
    prefix: &str,
    codebook_size: usize,
    output_dim: usize,
) -> AudioResult<FishAudioVectorQuantizer> {
    let codebook_key = format!("{prefix}.codebook.weights");
    let (shape, values) = reader.read_tensor_f32(&codebook_key)?;
    if shape.len() != 2 {
        return Err(AudioError::Runtime(format!("expected rank-2 tensor for '{codebook_key}', got shape {shape:?}")));
    }
    if shape[0] != codebook_size {
        return Err(AudioError::Runtime(format!(
            "codebook rows mismatch for '{codebook_key}': expected {codebook_size}, got {}",
            shape[0]
        )));
    }
    let code_dim = shape[1];
    if code_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let out_proj = read_matrix_f32(reader, &format!("{prefix}.out_proj.weights"), output_dim, code_dim)?;
    let out_bias = read_tensor_1d(reader, &format!("{prefix}.out_proj.biases"))?;
    if out_bias.len() != output_dim {
        return Err(AudioError::Runtime(format!(
            "out_proj bias shape mismatch for '{prefix}': expected {output_dim}, got {}",
            out_bias.len()
        )));
    }

    Ok(FishAudioVectorQuantizer {
        codebook: MatrixF32 {
            rows: shape[0],
            cols: shape[1],
            values,
        },
        out_proj,
        out_bias,
    })
}

fn read_fishaudio_conv1d_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
    groups: usize,
) -> AudioResult<FishAudioConv1dLayer> {
    let weight_key = format!("{prefix}.weights");
    let bias_key = format!("{prefix}.biases");
    let (shape, values) = reader.read_tensor_f32(&weight_key)?;
    if shape.len() != 3 {
        return Err(AudioError::Runtime(format!("expected rank-3 tensor for '{weight_key}', got shape {shape:?}")));
    }
    let bias = read_tensor_1d(reader, &bias_key)?;
    if bias.len() != shape[0] {
        return Err(AudioError::Runtime(format!(
            "bias shape mismatch for '{bias_key}': expected {}, got {}",
            shape[0],
            bias.len()
        )));
    }
    if groups == 0 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if shape[0] % groups != 0 {
        return Err(AudioError::Runtime(format!(
            "invalid grouped conv weights for '{weight_key}': out_channels {} not divisible by groups {groups}",
            shape[0]
        )));
    }

    Ok(FishAudioConv1dLayer {
        weight: values,
        bias,
        cin: shape[1].checked_mul(groups).ok_or(AudioError::Runtime("conv input channel overflow".to_string()))?,
        cout: shape[0],
        kernel_size: shape[2],
        dilation,
        groups,
    })
}

fn read_fishaudio_transpose_conv_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    stride: usize,
    groups: usize,
) -> AudioResult<FishAudioConvTranspose1dLayer> {
    if stride == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let weight_oih = read_tensor_3d(reader, &format!("{prefix}.weights"))?;
    let bias = read_tensor_1d(reader, &format!("{prefix}.biases"))?;
    let out_channels = weight_oih.shape[0];
    if bias.len() != out_channels {
        return Err(AudioError::Runtime(format!(
            "transpose conv bias mismatch for '{prefix}.biases': expected {out_channels}, got {}",
            bias.len()
        )));
    }
    let in_channels = weight_oih.shape[1]
        .checked_mul(groups)
        .ok_or(AudioError::Runtime("transpose conv input channel overflow".to_string()))?;
    let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, in_channels, out_channels, groups)?;
    Ok(FishAudioConvTranspose1dLayer {
        weight: converted.values,
        bias,
        cin: in_channels,
        cout: out_channels,
        stride,
        groups,
    })
}

fn read_fishaudio_residual_unit_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    dilation: usize,
) -> AudioResult<FishAudioResidualUnitLayer> {
    Ok(FishAudioResidualUnitLayer {
        snake1_alpha: read_tensor_1d(reader, &format!("{prefix}.snake1.alpha"))?,
        conv1: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv1"), dilation, 1)?,
        snake2_alpha: read_tensor_1d(reader, &format!("{prefix}.snake2.alpha"))?,
        conv2: read_fishaudio_conv1d_layer(reader, &format!("{prefix}.conv2"), 1, 1)?,
    })
}

fn read_fishaudio_norm_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    epsilon: f32,
    subtract_mean: bool,
    use_bias: bool,
) -> AudioResult<FishAudioNormLayer> {
    let scales = read_tensor_1d(reader, &format!("{prefix}.scales"))?;
    let biases = if use_bias {
        Some(read_tensor_1d(reader, &format!("{prefix}.biases"))?)
    } else {
        None
    };
    if let Some(biases) = &biases {
        if biases.len() != scales.len() {
            return Err(AudioError::Runtime(format!(
                "norm scale/bias length mismatch at '{prefix}': {} vs {}",
                scales.len(),
                biases.len()
            )));
        }
    }
    Ok(FishAudioNormLayer {
        scales,
        biases,
        epsilon,
        subtract_mean,
    })
}

fn read_fishaudio_convnext_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    norm_config: &FishAudioConvNeXtNormConfigJson,
) -> AudioResult<FishAudioConvNeXtLayer> {
    let depthwise_weight = read_tensor_3d(reader, &format!("{prefix}.dwconv.weights"))?;
    let depthwise_bias = read_tensor_1d(reader, &format!("{prefix}.dwconv.biases"))?;
    if depthwise_weight.shape[1] != 1 {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise weight in_channels_per_group must be 1 at {prefix}, got {}",
            depthwise_weight.shape[1]
        )));
    }
    let depthwise_conv = FishAudioConv1dLayer {
        weight: depthwise_weight.values,
        bias: depthwise_bias,
        cin: depthwise_weight.shape[0],
        cout: depthwise_weight.shape[0],
        kernel_size: depthwise_weight.shape[2],
        dilation: 1,
        groups: depthwise_weight.shape[0],
    };
    if depthwise_conv.bias.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise bias mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            depthwise_conv.bias.len()
        )));
    }
    if depthwise_conv.cout == 0 || depthwise_conv.kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if depthwise_conv.cin != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt depthwise conv expects cin==cout, got {} vs {} at {prefix}",
            depthwise_conv.cin, depthwise_conv.cout
        )));
    }

    let norm = read_fishaudio_norm_layer(
        reader,
        &format!("{prefix}.norm"),
        norm_config.epsilon,
        norm_config.subtract_mean,
        norm_config.use_bias,
    )?;
    if norm.scales.len() != depthwise_conv.cout {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt norm channels mismatch at {prefix}: expected {}, got {}",
            depthwise_conv.cout,
            norm.scales.len()
        )));
    }

    let pwconv1 = read_matrix_f32_any(reader, &format!("{prefix}.pwconv1.weights"))?;
    let pwconv1_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv1.biases"))?;
    if pwconv1.cols != depthwise_conv.cout || pwconv1.rows != pwconv1_bias.len() {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv1 shape mismatch at {prefix}: weight [{}, {}], bias {}",
            pwconv1.rows,
            pwconv1.cols,
            pwconv1_bias.len()
        )));
    }

    let pwconv2 = read_matrix_f32(reader, &format!("{prefix}.pwconv2.weights"), depthwise_conv.cout, pwconv1.rows)?;
    let pwconv2_bias = read_tensor_1d(reader, &format!("{prefix}.pwconv2.biases"))?;
    if pwconv2_bias.len() != pwconv2.rows {
        return Err(AudioError::Runtime(format!(
            "ConvNeXt pwconv2 bias mismatch at {prefix}: expected {}, got {}",
            pwconv2.rows,
            pwconv2_bias.len()
        )));
    }

    Ok(FishAudioConvNeXtLayer {
        depthwise_conv,
        norm,
        pwconv1,
        pwconv1_bias,
        pwconv2,
        pwconv2_bias,
    })
}

fn read_fishaudio_post_module_layer(
    reader: &SafeTensorReader,
    prefix: &str,
    layer_config: &crate::config::TransformerLayerConfig,
    model_dim: usize,
    hidden_dim: usize,
) -> AudioResult<FishAudioPostModuleLayer> {
    let Some(pre_mixer_norm_config) = layer_config.pre_attention_norm_config.as_ref() else {
        return Err(AudioError::Runtime("FishAudio post_module requires pre_attention_norm_config".to_string()));
    };
    let pre_mixer_norm = read_fishaudio_norm_layer(
        reader,
        &format!("{prefix}.pre_mixer_norm"),
        pre_mixer_norm_config.epsilon,
        pre_mixer_norm_config.subtract_mean,
        false,
    )?;
    let pre_mlp_norm = read_fishaudio_norm_layer(
        reader,
        &format!("{prefix}.pre_mlp_norm"),
        layer_config.pre_mlp_norm_config.epsilon,
        layer_config.pre_mlp_norm_config.subtract_mean,
        false,
    )?;

    let crate::config::MixerConfig::Attention(attention_config) = &layer_config.mixer_config else {
        return Err(AudioError::Runtime(
            "FishAudio post_module layer mixer must be AttentionConfig".to_string(),
        ));
    };
    let num_heads = attention_config
        .num_heads
        .ok_or(AudioError::Runtime("FishAudio attention num_heads missing".to_string()))?;
    let num_groups = attention_config
        .num_groups
        .ok_or(AudioError::Runtime("FishAudio attention num_groups missing".to_string()))?;
    let head_dim = attention_config
        .head_dim
        .ok_or(AudioError::Runtime("FishAudio attention head_dim missing".to_string()))?;
    if num_heads == 0 || num_groups == 0 || head_dim == 0 || num_heads % num_groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }
    let attention_dim = num_heads
        .checked_mul(head_dim)
        .ok_or(AudioError::Runtime("FishAudio attention dimension overflow".to_string()))?;
    let qkv_projection =
        read_matrix_f32(reader, &format!("{prefix}.mixer.qkv_projection.weights"), attention_dim * 3, model_dim)?;
    let out_projection =
        read_matrix_f32(reader, &format!("{prefix}.mixer.out_projection.weights"), model_dim, attention_dim)?;

    let crate::config::MLPConfig::Dense(mlp_config) = &layer_config.mlp_config else {
        return Err(AudioError::Runtime("FishAudio post_module MLP must be dense".to_string()));
    };
    let up_projection = read_matrix_f32(
        reader,
        &format!("{prefix}.mlp.up_projection.weights"),
        hidden_dim
            .checked_mul(2)
            .ok_or(AudioError::Runtime("FishAudio hidden dimension overflow".to_string()))?,
        model_dim,
    )?;
    let down_projection =
        read_matrix_f32(reader, &format!("{prefix}.mlp.down_projection.weights"), model_dim, hidden_dim)?;

    let _ = mlp_config; // validated by shape and current supported path.

    let attention_scale = attention_config.scale.unwrap_or(1.0 / (head_dim as f32).sqrt());
    Ok(FishAudioPostModuleLayer {
        pre_mixer_norm,
        qkv_projection,
        out_projection,
        pre_mlp_norm,
        up_projection,
        down_projection,
        num_heads,
        num_groups,
        head_dim,
        attention_scale,
        sliding_window_size: attention_config.sliding_window_size,
    })
}

fn build_fishaudio_post_module(
    reader: &SafeTensorReader,
    cfg: &FishAudioAudioDecoderConfigJson,
) -> AudioResult<FishAudioPostModule> {
    let transformer = &cfg.quantizer_config.post_module_config;
    if transformer.model_dim != cfg.input_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio post_module model_dim mismatch: expected {}, got {}",
            cfg.input_dim, transformer.model_dim
        )));
    }
    let first_layer = transformer
        .layer_configs
        .first()
        .ok_or(AudioError::Runtime("FishAudio post_module has no layers".to_string()))?;
    let crate::config::MixerConfig::Attention(first_attention) = &first_layer.mixer_config else {
        return Err(AudioError::Runtime(
            "FishAudio post_module first layer mixer must be AttentionConfig".to_string(),
        ));
    };
    let rope_head_dim = first_attention
        .head_dim
        .ok_or(AudioError::Runtime("FishAudio post_module head_dim missing".to_string()))?;
    let rope_cosines = read_matrix_f32(
        reader,
        "audio_decoder.quantizer.post_module.global_rope.cosines",
        transformer.context_length,
        rope_head_dim,
    )?;
    let rope_sines = read_matrix_f32(
        reader,
        "audio_decoder.quantizer.post_module.global_rope.sines",
        rope_cosines.rows,
        rope_cosines.cols,
    )?;
    let output_norm = read_fishaudio_norm_layer(
        reader,
        "audio_decoder.quantizer.post_module.output_norm",
        transformer.output_norm_config.epsilon,
        transformer.output_norm_config.subtract_mean,
        false,
    )?;
    if output_norm.scales.len() != transformer.model_dim {
        return Err(AudioError::Runtime(format!(
            "FishAudio output_norm scale mismatch: expected {}, got {}",
            transformer.model_dim,
            output_norm.scales.len()
        )));
    }

    let mut layers = Vec::with_capacity(transformer.layer_configs.len());
    for (index, layer_config) in transformer.layer_configs.iter().enumerate() {
        layers.push(read_fishaudio_post_module_layer(
            reader,
            &format!("audio_decoder.quantizer.post_module.layers.{index}"),
            layer_config,
            transformer.model_dim,
            transformer.hidden_dim,
        )?);
    }

    Ok(FishAudioPostModule {
        rope_cosines,
        rope_sines,
        layers,
        output_norm,
        model_dim: transformer.model_dim,
        hidden_dim: transformer.hidden_dim,
    })
}

fn build_fishaudio_decoder_graph(
    reader: &SafeTensorReader,
    cfg: &FishAudioAudioDecoderConfigJson,
) -> AudioResult<FishAudioDecoderGraph> {
    if cfg.decoder_rates.is_empty() {
        return Err(AudioError::InvalidTokenCardinality);
    }
    if cfg.quantizer_config.upsampler_config.block_configs.len() != cfg.downsample_factor.len() {
        return Err(AudioError::Runtime(format!(
            "FishAudio upsampler config mismatch: {} block configs for {} downsample factors",
            cfg.quantizer_config.upsampler_config.block_configs.len(),
            cfg.downsample_factor.len()
        )));
    }

    let mut upsample_blocks = Vec::with_capacity(cfg.downsample_factor.len());
    for (index, &stride) in cfg.downsample_factor.iter().rev().enumerate() {
        let trans_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.trans_conv");
        let convnext_prefix = format!("audio_decoder.quantizer.upsampler.blocks.{index}.convnext");
        let trans_conv = read_fishaudio_transpose_conv_layer(reader, &trans_prefix, stride, 1)?;
        let convnext = read_fishaudio_convnext_layer(
            reader,
            &convnext_prefix,
            &cfg.quantizer_config.upsampler_config.block_configs[index].convnext_config.norm_config,
        )?;
        if convnext.depthwise_conv.cin != trans_conv.cout {
            return Err(AudioError::Runtime(format!(
                "FishAudio upsampler convnext channel mismatch at block {index}: trans_conv out {} vs convnext in {}",
                trans_conv.cout, convnext.depthwise_conv.cin
            )));
        }
        upsample_blocks.push((trans_conv, convnext));
    }

    let first_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.first_conv", 1, 1)?;
    let mut decoder_blocks = Vec::with_capacity(cfg.decoder_rates.len());
    for (index, &stride) in cfg.decoder_rates.iter().enumerate() {
        let base = format!("audio_decoder.decoder.decoder_blocks.{index}");
        decoder_blocks.push(FishAudioDecoderBlockLayer {
            snake_alpha: read_tensor_1d(reader, &format!("{base}.snake.alpha"))?,
            trans_conv: read_fishaudio_transpose_conv_layer(reader, &format!("{base}.trans_conv"), stride, 1)?,
            res_unit1: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit1"), 1)?,
            res_unit2: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit2"), 3)?,
            res_unit3: read_fishaudio_residual_unit_layer(reader, &format!("{base}.res_unit3"), 9)?,
        });
    }

    let final_snake_alpha = read_tensor_1d(reader, "audio_decoder.decoder.final_snake.alpha")?;
    let final_conv = read_fishaudio_conv1d_layer(reader, "audio_decoder.decoder.final_conv", 1, 1)?;

    let upsample_factor = cfg
        .downsample_factor
        .iter()
        .chain(cfg.decoder_rates.iter())
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("FishAudio decoder upsample factor overflow".to_string()))?;

    Ok(FishAudioDecoderGraph {
        first_conv,
        upsample_blocks,
        decoder_blocks,
        final_snake_alpha,
        final_conv,
        upsample_factor,
    })
}

fn build_fishaudio_codec_graph(
    cfg: &FishAudioAudioDecoderConfigJson,
    model_weights_path: &Path,
) -> AudioResult<FishAudioCodecGraph> {
    let reader = SafeTensorReader::open(model_weights_path)?;
    if cfg.n_codebooks == 0 || cfg.codebook_size <= 1 || cfg.semantic_codebook_size <= 1 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let semantic_quantizer = read_fishaudio_vector_quantizer(
        &reader,
        "audio_decoder.quantizer.semantic_quantizer.quantizers.0",
        cfg.semantic_codebook_size,
        cfg.input_dim,
    )?;
    let mut residual_quantizers = Vec::with_capacity(cfg.n_codebooks);
    for index in 0..cfg.n_codebooks {
        let prefix = format!("audio_decoder.quantizer.quantizer.quantizers.{index}");
        residual_quantizers.push(read_fishaudio_vector_quantizer(&reader, &prefix, cfg.codebook_size, cfg.input_dim)?);
    }
    let post_module = build_fishaudio_post_module(&reader, cfg)?;
    let decoder = build_fishaudio_decoder_graph(&reader, cfg)?;
    let total_codebooks =
        cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime("FishAudio codebook count overflow".to_string()))?;

    Ok(FishAudioCodecGraph {
        semantic_quantizer,
        residual_quantizers,
        post_module,
        decoder,
        codebook_size: cfg.codebook_size,
        semantic_codebook_size: cfg.semantic_codebook_size,
        input_dim: cfg.input_dim,
        total_codebooks,
        upsample_factor: cfg
            .downsample_factor
            .iter()
            .chain(cfg.decoder_rates.iter())
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or(AudioError::Runtime("FishAudio upsample factor overflow".to_string()))?,
    })
}

fn snake1d_reference(
    input: &[f32],
    alpha: &[f32],
    batch_size: usize,
    channels: usize,
    seq_len: usize,
) -> AudioResult<Vec<f32>> {
    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }
    if alpha.len() != channels {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: channels,
            actual_tokens: alpha.len(),
        });
    }

    let mut output = vec![0.0_f32; input.len()];
    for batch in 0..batch_size {
        for channel in 0..channels {
            let a = alpha[channel];
            for time in 0..seq_len {
                let index = (batch * channels + channel) * seq_len + time;
                let x = input[index];
                let sine = (a * x).sin();
                output[index] = x + (sine * sine) / (a + 1e-9);
            }
        }
    }
    Ok(output)
}

fn causal_conv1d_grouped_reference(
    input: &[f32],
    layer: &FishAudioConv1dLayer,
    lengths: &[i32],
    batch_size: usize,
    seq_len: usize,
) -> AudioResult<Vec<f32>> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }
    if layer.groups == 1 {
        return super::ops::causal_conv1d_reference(super::ops::CausalConv1dSpec {
            input,
            weight: &layer.weight,
            bias: &layer.bias,
            lengths,
            batch_size,
            cin: layer.cin,
            cout: layer.cout,
            seq_len,
            kernel_size: layer.kernel_size,
            dilation: layer.dilation,
        });
    }
    if layer.groups == 0 || layer.cin % layer.groups != 0 || layer.cout % layer.groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let expected_input = checked_product(&[batch_size, layer.cin, seq_len])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }
    let cin_per_group = layer.cin / layer.groups;
    let cout_per_group = layer.cout / layer.groups;
    let expected_weight = checked_product(&[layer.cout, cin_per_group, layer.kernel_size])?;
    if layer.weight.len() != expected_weight {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_weight,
            actual_tokens: layer.weight.len(),
        });
    }
    if layer.bias.len() != layer.cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: layer.cout,
            actual_tokens: layer.bias.len(),
        });
    }

    let output_len = checked_product(&[batch_size, layer.cout, seq_len])?;
    let mut output = vec![0.0_f32; output_len];
    let pad = (layer.kernel_size - 1) * layer.dilation;

    for batch in 0..batch_size {
        let length = lengths[batch];
        if length < 0 || length as usize > seq_len {
            return Err(AudioError::InvalidTokenLengthValue {
                length: length.max(0) as usize,
                frames: seq_len,
            });
        }
        let length = length as usize;

        for out_channel in 0..layer.cout {
            let group = out_channel / cout_per_group;
            let in_begin = group * cin_per_group;
            let in_end = in_begin + cin_per_group;
            let out_channel_in_group = out_channel % cout_per_group;

            for time in 0..seq_len {
                let out_index = (batch * layer.cout + out_channel) * seq_len + time;
                if time >= length {
                    output[out_index] = 0.0;
                    continue;
                }
                let mut acc = layer.bias[out_channel];
                for in_channel in in_begin..in_end {
                    let input_base = (batch * layer.cin + in_channel) * seq_len;
                    let in_channel_in_group = in_channel - in_begin;
                    let weight_base = ((group * cout_per_group + out_channel_in_group) * cin_per_group
                        + in_channel_in_group)
                        * layer.kernel_size;
                    for kernel_offset in 0..layer.kernel_size {
                        let x_time = time as isize + (kernel_offset * layer.dilation) as isize - pad as isize;
                        if x_time < 0 || x_time >= seq_len as isize {
                            continue;
                        }
                        let x_index = input_base + x_time as usize;
                        let w_index = weight_base + kernel_offset;
                        acc += layer.weight[w_index] * input[x_index];
                    }
                }
                output[out_index] = acc;
            }
        }
    }

    Ok(output)
}

impl FishAudioCodecGraph {
    fn add_quantized_code_to_latent(
        target: &mut [f32],
        quantizer: &FishAudioVectorQuantizer,
        code_index: usize,
    ) -> AudioResult<()> {
        let code_row = quantizer
            .codebook
            .row(code_index)
            .ok_or(AudioError::Runtime("quantizer code index out of range".to_string()))?;
        if quantizer.out_proj.cols != code_row.len() || quantizer.out_proj.rows != target.len() {
            return Err(AudioError::Runtime("quantizer projection shape mismatch".to_string()));
        }
        for (row_index, row) in quantizer.out_proj.values.chunks_exact(quantizer.out_proj.cols).enumerate() {
            let mut acc = quantizer.out_bias[row_index];
            for (&w, &x) in row.iter().zip(code_row.iter()) {
                acc += w * x;
            }
            target[row_index] += acc;
        }
        Ok(())
    }

    fn decode_quantizer_to_nsc(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        if codebooks != self.total_codebooks {
            return Err(AudioError::Runtime(format!(
                "FishAudio codebook mismatch: expected {}, got {codebooks}",
                self.total_codebooks
            )));
        }
        let expected_tokens = checked_product(&[batch_size, codebooks, frames])?;
        if tokens.len() != expected_tokens {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens,
                actual_tokens: tokens.len(),
            });
        }
        if lengths.len() != batch_size {
            return Err(AudioError::InvalidTokenLengths {
                expected_lengths: batch_size,
                actual_lengths: lengths.len(),
            });
        }

        let mut latent_nsc = vec![0.0_f32; checked_product(&[batch_size, frames, self.input_dim])?];
        for batch in 0..batch_size {
            let active_frames = lengths[batch];
            if active_frames > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_frames,
                    frames,
                });
            }
            for frame in 0..active_frames {
                let row_start = (batch * frames + frame) * self.input_dim;
                let row_end = row_start + self.input_dim;
                let target = &mut latent_nsc[row_start..row_end];

                let semantic_token_index = ((batch * codebooks) * frames) + frame;
                let semantic_token = tokens[semantic_token_index] as usize;
                let semantic_clamped = semantic_token.min(self.semantic_codebook_size.saturating_sub(1));
                Self::add_quantized_code_to_latent(target, &self.semantic_quantizer, semantic_clamped)?;

                for (residual_index, quantizer) in self.residual_quantizers.iter().enumerate() {
                    let token_index = ((batch * codebooks + (residual_index + 1)) * frames) + frame;
                    let token = tokens[token_index] as usize;
                    let clamped = token.min(self.codebook_size.saturating_sub(1));
                    Self::add_quantized_code_to_latent(target, quantizer, clamped)?;
                }
            }
        }

        Ok(latent_nsc)
    }

    fn nsc_to_ncs(
        values: &[f32],
        batch_size: usize,
        frames: usize,
        channels: usize,
    ) -> AudioResult<Vec<f32>> {
        let expected = checked_product(&[batch_size, frames, channels])?;
        if values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: values.len(),
            });
        }

        let mut output = vec![0.0_f32; checked_product(&[batch_size, channels, frames])?];
        for batch in 0..batch_size {
            for frame in 0..frames {
                let src_base = (batch * frames + frame) * channels;
                for channel in 0..channels {
                    let dst_index = (batch * channels + channel) * frames + frame;
                    output[dst_index] = values[src_base + channel];
                }
            }
        }
        Ok(output)
    }

    fn ncs_to_nsc(
        values: &[f32],
        batch_size: usize,
        channels: usize,
        frames: usize,
    ) -> AudioResult<Vec<f32>> {
        let expected = checked_product(&[batch_size, channels, frames])?;
        if values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: values.len(),
            });
        }

        let mut output = vec![0.0_f32; checked_product(&[batch_size, frames, channels])?];
        for batch in 0..batch_size {
            for frame in 0..frames {
                let dst_base = (batch * frames + frame) * channels;
                for channel in 0..channels {
                    let src_index = (batch * channels + channel) * frames + frame;
                    output[dst_base + channel] = values[src_index];
                }
            }
        }
        Ok(output)
    }

    fn apply_norm_sequence(
        values: &mut [f32],
        seq_len: usize,
        channels: usize,
        norm: &FishAudioNormLayer,
    ) -> AudioResult<()> {
        if norm.scales.len() != channels {
            return Err(AudioError::Runtime(format!(
                "norm scale length mismatch: expected {channels}, got {}",
                norm.scales.len()
            )));
        }
        if let Some(biases) = &norm.biases {
            if biases.len() != channels {
                return Err(AudioError::Runtime(format!(
                    "norm bias length mismatch: expected {channels}, got {}",
                    biases.len()
                )));
            }
        }
        let expected = checked_product(&[seq_len, channels])?;
        if values.len() != expected {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected,
                actual_tokens: values.len(),
            });
        }

        for token in 0..seq_len {
            let row_start = token * channels;
            let row = &mut values[row_start..row_start + channels];
            let mean = if norm.subtract_mean {
                row.iter().sum::<f32>() / channels as f32
            } else {
                0.0
            };
            let variance_sum = if norm.subtract_mean {
                row.iter()
                    .map(|&value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
            } else {
                row.iter().map(|&value| value * value).sum::<f32>()
            };
            let variance = variance_sum / channels as f32;

            let inv_std = 1.0 / (variance + norm.epsilon).sqrt();
            for channel in 0..channels {
                let mut normalized = if norm.subtract_mean {
                    row[channel] - mean
                } else {
                    row[channel]
                };
                normalized *= inv_std * norm.scales[channel];
                if let Some(biases) = &norm.biases {
                    normalized += biases[channel];
                }
                row[channel] = normalized;
            }
        }
        Ok(())
    }

    fn linear_sequence(
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        weight: &MatrixF32,
        bias: Option<&[f32]>,
    ) -> AudioResult<Vec<f32>> {
        let expected_input = checked_product(&[seq_len, in_dim])?;
        if input.len() != expected_input {
            return Err(AudioError::InvalidTokenShape {
                expected_tokens: expected_input,
                actual_tokens: input.len(),
            });
        }
        if weight.cols != in_dim {
            return Err(AudioError::Runtime(format!(
                "linear input dim mismatch: expected {}, got {}",
                weight.cols, in_dim
            )));
        }
        if let Some(bias_values) = bias {
            if bias_values.len() != weight.rows {
                return Err(AudioError::Runtime(format!(
                    "linear bias shape mismatch: expected {}, got {}",
                    weight.rows,
                    bias_values.len()
                )));
            }
        }

        let mut output = vec![0.0_f32; checked_product(&[seq_len, weight.rows])?];
        for token in 0..seq_len {
            let input_row = &input[token * in_dim..(token + 1) * in_dim];
            let output_row = &mut output[token * weight.rows..(token + 1) * weight.rows];
            for row_index in 0..weight.rows {
                let mut acc = bias.map_or(0.0, |bias_values| bias_values[row_index]);
                let row = &weight.values[row_index * weight.cols..(row_index + 1) * weight.cols];
                for (&weight_value, &input_value) in row.iter().zip(input_row.iter()) {
                    acc += weight_value * input_value;
                }
                output_row[row_index] = acc;
            }
        }
        Ok(output)
    }

    fn apply_convnext_ncs(
        &self,
        input: &[f32],
        layer: &FishAudioConvNeXtLayer,
        lengths: &[i32],
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Vec<f32>> {
        let residual = input.to_vec();
        let x = causal_conv1d_grouped_reference(input, &layer.depthwise_conv, lengths, batch_size, seq_len)?;
        let mut x_nsc = Self::ncs_to_nsc(&x, batch_size, channels, seq_len)?;
        for batch in 0..batch_size {
            let active_len = usize::try_from(lengths[batch]).map_err(|_| {
                AudioError::Runtime("ConvNeXt received negative sequence length".to_string())
            })?;
            let batch_start = batch * seq_len * channels;
            let active_slice = &mut x_nsc[batch_start..batch_start + active_len * channels];
            Self::apply_norm_sequence(active_slice, active_len, channels, &layer.norm)?;

            let y = Self::linear_sequence(active_slice, active_len, channels, &layer.pwconv1, Some(&layer.pwconv1_bias))?;
            let mut y = y
                .into_iter()
                .map(|value| {
                    let x3 = value * value * value;
                    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * x3)).tanh())
                })
                .collect::<Vec<_>>();
            y = Self::linear_sequence(&y, active_len, layer.pwconv1.rows, &layer.pwconv2, Some(&layer.pwconv2_bias))?;
            active_slice.copy_from_slice(&y);
        }
        let mut x_ncs = Self::nsc_to_ncs(&x_nsc, batch_size, seq_len, channels)?;
        if x_ncs.len() != residual.len() {
            return Err(AudioError::Runtime("ConvNeXt residual shape mismatch".to_string()));
        }
        for (dst, res) in x_ncs.iter_mut().zip(residual.iter()) {
            *dst += *res;
        }
        Ok(x_ncs)
    }

    fn apply_post_module(
        &self,
        latent_nsc: &mut [f32],
        lengths: &[usize],
        batch_size: usize,
        frames: usize,
    ) -> AudioResult<()> {
        if self.post_module.model_dim != self.input_dim {
            return Err(AudioError::Runtime("post_module model_dim mismatch".to_string()));
        }
        for batch in 0..batch_size {
            let active_len = lengths[batch];
            if active_len == 0 {
                continue;
            }
            if active_len > frames {
                return Err(AudioError::InvalidTokenLengthValue {
                    length: active_len,
                    frames,
                });
            }

            let batch_base = batch * frames * self.input_dim;
            let sequence = &mut latent_nsc[batch_base..batch_base + active_len * self.input_dim];
            let mut x = sequence.to_vec();

            for layer in &self.post_module.layers {
                let mut normed = x.clone();
                Self::apply_norm_sequence(&mut normed, active_len, self.input_dim, &layer.pre_mixer_norm)?;
                let qkv = Self::linear_sequence(&normed, active_len, self.input_dim, &layer.qkv_projection, None)?;
                let attention_dim = layer
                    .num_heads
                    .checked_mul(layer.head_dim)
                    .ok_or(AudioError::Runtime("attention dimension overflow".to_string()))?;
                let group_dim = layer
                    .num_groups
                    .checked_mul(layer.head_dim)
                    .ok_or(AudioError::Runtime("group dimension overflow".to_string()))?;
                if attention_dim != group_dim {
                    return Err(AudioError::Runtime(
                        "post_module currently requires num_heads == num_groups".to_string(),
                    ));
                }

                let mut q = vec![0.0_f32; active_len * attention_dim];
                let mut k = vec![0.0_f32; active_len * attention_dim];
                let mut v = vec![0.0_f32; active_len * attention_dim];
                for token in 0..active_len {
                    let row = &qkv[token * (attention_dim * 3)..(token + 1) * (attention_dim * 3)];
                    q[token * attention_dim..(token + 1) * attention_dim].copy_from_slice(&row[..attention_dim]);
                    k[token * attention_dim..(token + 1) * attention_dim]
                        .copy_from_slice(&row[attention_dim..attention_dim * 2]);
                    v[token * attention_dim..(token + 1) * attention_dim]
                        .copy_from_slice(&row[attention_dim * 2..attention_dim * 3]);
                }

                let half = layer.head_dim / 2;
                for token in 0..active_len {
                    for head in 0..layer.num_heads {
                        let rope_row = token.min(self.post_module.rope_cosines.rows.saturating_sub(1));
                        for dim in 0..layer.head_dim {
                            let cos = self.post_module.rope_cosines.values[rope_row * layer.head_dim + dim];
                            let sin = self.post_module.rope_sines.values[rope_row * layer.head_dim + dim];
                            let base = token * attention_dim + head * layer.head_dim;
                            let qv = q[base + dim];
                            let kv = k[base + dim];
                            let q_pair = if dim < half {
                                -q[base + dim + half]
                            } else {
                                q[base + dim - half]
                            };
                            let k_pair = if dim < half {
                                -k[base + dim + half]
                            } else {
                                k[base + dim - half]
                            };
                            q[base + dim] = qv * cos + q_pair * sin;
                            k[base + dim] = kv * cos + k_pair * sin;
                        }
                    }
                }

                let mut attention_output = vec![0.0_f32; active_len * attention_dim];
                for token in 0..active_len {
                    let window_start = layer
                        .sliding_window_size
                        .map(|window| token.saturating_sub(window.saturating_sub(1)))
                        .unwrap_or(0);

                    for head in 0..layer.num_heads {
                        let query_offset = token * attention_dim + head * layer.head_dim;
                        let mut logits = Vec::with_capacity(token + 1 - window_start);
                        let mut max_logit = f32::NEG_INFINITY;
                        for key_token in window_start..=token {
                            let key_offset = key_token * attention_dim + head * layer.head_dim;
                            let mut score = 0.0_f32;
                            for dim in 0..layer.head_dim {
                                score += q[query_offset + dim] * k[key_offset + dim];
                            }
                            score *= layer.attention_scale;
                            max_logit = max_logit.max(score);
                            logits.push((key_token, score));
                        }
                        let mut denom = 0.0_f32;
                        for (_, score) in logits.iter_mut() {
                            *score = (*score - max_logit).exp();
                            denom += *score;
                        }
                        if denom <= 0.0 {
                            continue;
                        }
                        for (key_token, score) in logits {
                            let weight = score / denom;
                            let value_offset = key_token * attention_dim + head * layer.head_dim;
                            for dim in 0..layer.head_dim {
                                attention_output[query_offset + dim] += weight * v[value_offset + dim];
                            }
                        }
                    }
                }

                let attention_projected =
                    Self::linear_sequence(&attention_output, active_len, attention_dim, &layer.out_projection, None)?;
                for (dst, value) in x.iter_mut().zip(attention_projected.iter()) {
                    *dst += *value;
                }

                let mut mlp_in = x.clone();
                Self::apply_norm_sequence(&mut mlp_in, active_len, self.input_dim, &layer.pre_mlp_norm)?;
                let up = Self::linear_sequence(&mlp_in, active_len, self.input_dim, &layer.up_projection, None)?;
                let mut hidden = vec![0.0_f32; active_len * self.post_module.hidden_dim];
                for token in 0..active_len {
                    let up_row = &up[token * self.post_module.hidden_dim * 2
                        ..(token + 1) * self.post_module.hidden_dim * 2];
                    let hidden_row =
                        &mut hidden[token * self.post_module.hidden_dim..(token + 1) * self.post_module.hidden_dim];
                    for dim in 0..self.post_module.hidden_dim {
                        let up_val = up_row[dim];
                        let gate_val = up_row[self.post_module.hidden_dim + dim];
                        let silu = gate_val / (1.0 + (-gate_val).exp());
                        hidden_row[dim] = up_val * silu;
                    }
                }
                let mlp_out =
                    Self::linear_sequence(&hidden, active_len, self.post_module.hidden_dim, &layer.down_projection, None)?;
                for (dst, value) in x.iter_mut().zip(mlp_out.iter()) {
                    *dst += *value;
                }
            }

            Self::apply_norm_sequence(&mut x, active_len, self.input_dim, &self.post_module.output_norm)?;
            sequence.copy_from_slice(&x);
        }

        Ok(())
    }

    fn run_residual_unit(
        &self,
        input: &[f32],
        unit: &FishAudioResidualUnitLayer,
        lengths: &[i32],
        batch_size: usize,
        channels: usize,
        seq_len: usize,
    ) -> AudioResult<Vec<f32>> {
        let x = snake1d_reference(input, &unit.snake1_alpha, batch_size, channels, seq_len)?;
        let x = causal_conv1d_grouped_reference(&x, &unit.conv1, lengths, batch_size, seq_len)?;
        let x = snake1d_reference(&x, &unit.snake2_alpha, batch_size, channels, seq_len)?;
        let x = causal_conv1d_grouped_reference(&x, &unit.conv2, lengths, batch_size, seq_len)?;
        if x.len() != input.len() {
            return Err(AudioError::Runtime("FishAudio residual output shape mismatch".to_string()));
        }
        let mut out = input.to_vec();
        for (dst, value) in out.iter_mut().zip(x.iter()) {
            *dst += *value;
        }
        Ok(out)
    }

    fn decode_padded(
        &self,
        tokens: &[u32],
        lengths: &[usize],
        batch_size: usize,
        codebooks: usize,
        frames: usize,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if batch_size == 0 || frames == 0 {
            let out_lengths = lengths
                .iter()
                .map(|&length| {
                    length
                        .checked_mul(self.upsample_factor)
                        .ok_or(AudioError::Runtime("FishAudio length scaling overflow".to_string()))
                })
                .collect::<AudioResult<Vec<_>>>()?;
            return Ok(super::decoder::DecodedPaddedAudio {
                samples: Vec::new(),
                channels: 1,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
            });
        }

        let mut lengths_i32 = lengths
            .iter()
            .map(|&length| {
                i32::try_from(length).map_err(|_| AudioError::Runtime("FishAudio length exceeds i32 range".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;

        let mut latent_nsc = self.decode_quantizer_to_nsc(tokens, lengths, batch_size, codebooks, frames)?;
        self.apply_post_module(&mut latent_nsc, lengths, batch_size, frames)?;
        let mut x = Self::nsc_to_ncs(&latent_nsc, batch_size, frames, self.input_dim)?;
        let mut current_channels = self.input_dim;
        let mut current_frames = frames;

        for (trans_conv, convnext) in &self.decoder.upsample_blocks {
            if trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio upsampler input channel mismatch: expected {}, got {}",
                    trans_conv.cin, current_channels
                )));
            }
            let next_frames = current_frames
                .checked_mul(trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio upsampler frame overflow".to_string()))?;
            let next_lengths = lengths_i32
                .iter()
                .copied()
                .map(|length| checked_mul_i32(length, trans_conv.stride))
                .collect::<AudioResult<Vec<_>>>()?;

            x = super::ops::causal_conv_transpose1d_lalamo_reference(super::ops::CausalConvTranspose1dSpec {
                input: &x,
                weight: &trans_conv.weight,
                bias: &trans_conv.bias,
                lengths: &next_lengths,
                batch_size,
                cin: trans_conv.cin,
                cout: trans_conv.cout,
                seq_len_in: current_frames,
                seq_len_out: next_frames,
                stride: trans_conv.stride,
                groups: trans_conv.groups,
            })?;
            x = self.apply_convnext_ncs(&x, convnext, &next_lengths, batch_size, trans_conv.cout, next_frames)?;

            current_frames = next_frames;
            current_channels = trans_conv.cout;
            lengths_i32 = next_lengths;
        }

        if self.decoder.first_conv.cin != current_channels {
            return Err(AudioError::Runtime(format!(
                "FishAudio decoder input channels mismatch: expected {}, got {}",
                self.decoder.first_conv.cin, current_channels
            )));
        }
        x = causal_conv1d_grouped_reference(&x, &self.decoder.first_conv, &lengths_i32, batch_size, current_frames)?;
        current_channels = self.decoder.first_conv.cout;

        for block in &self.decoder.decoder_blocks {
            if block.trans_conv.cin != current_channels {
                return Err(AudioError::Runtime(format!(
                    "FishAudio decoder block input mismatch: expected {}, got {}",
                    block.trans_conv.cin, current_channels
                )));
            }
            x = snake1d_reference(&x, &block.snake_alpha, batch_size, current_channels, current_frames)?;

            let next_frames = current_frames
                .checked_mul(block.trans_conv.stride)
                .ok_or(AudioError::Runtime("FishAudio decoder frame overflow".to_string()))?;
            let next_lengths = lengths_i32
                .iter()
                .copied()
                .map(|length| checked_mul_i32(length, block.trans_conv.stride))
                .collect::<AudioResult<Vec<_>>>()?;

            x = super::ops::causal_conv_transpose1d_lalamo_reference(super::ops::CausalConvTranspose1dSpec {
                input: &x,
                weight: &block.trans_conv.weight,
                bias: &block.trans_conv.bias,
                lengths: &next_lengths,
                batch_size,
                cin: block.trans_conv.cin,
                cout: block.trans_conv.cout,
                seq_len_in: current_frames,
                seq_len_out: next_frames,
                stride: block.trans_conv.stride,
                groups: block.trans_conv.groups,
            })?;

            current_frames = next_frames;
            current_channels = block.trans_conv.cout;
            lengths_i32 = next_lengths;

            x = self.run_residual_unit(
                &x,
                &block.res_unit1,
                &lengths_i32,
                batch_size,
                current_channels,
                current_frames,
            )?;
            x = self.run_residual_unit(
                &x,
                &block.res_unit2,
                &lengths_i32,
                batch_size,
                current_channels,
                current_frames,
            )?;
            x = self.run_residual_unit(
                &x,
                &block.res_unit3,
                &lengths_i32,
                batch_size,
                current_channels,
                current_frames,
            )?;
        }

        x = snake1d_reference(&x, &self.decoder.final_snake_alpha, batch_size, current_channels, current_frames)?;
        x = causal_conv1d_grouped_reference(&x, &self.decoder.final_conv, &lengths_i32, batch_size, current_frames)?;
        for sample in &mut x {
            *sample = sample.tanh();
        }

        let out_lengths = lengths_i32
            .into_iter()
            .map(|length| {
                usize::try_from(length)
                    .map_err(|_| AudioError::Runtime("FishAudio decoder produced invalid negative length".to_string()))
            })
            .collect::<AudioResult<Vec<_>>>()?;

        Ok(super::decoder::DecodedPaddedAudio {
            samples: x,
            channels: self.decoder.final_conv.cout,
            frames: current_frames,
            lengths: out_lengths,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NanoCodecFsqRuntimeConfig {
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Box<[i32]>,
    dim_base_index: Box<[i32]>,
    codebook_dim_per_group: usize,
    channels: usize,
    codec_cardinality: u32,
    eps: f32,
    output_packing: AudioTokenPacking,
    decoder: Option<NanoCodecDecoderGraph>,
    fishaudio_decoder: Option<FishAudioCodecGraph>,
}

impl NanoCodecFsqRuntimeConfig {
    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        let parsed = parse_runtime_config_json(tts_config)?;
        Self::from_runtime_config_json(parsed)
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        if tts_config.get("audio_codec").is_some() {
            return Self::from_tts_config_value(tts_config);
        }

        if tts_config.get("audio_decoder_config").is_some() {
            if let Some(decoder_type) = tts_config
                .get("audio_decoder_config")
                .and_then(|decoder| decoder.get("type"))
                .and_then(serde_json::Value::as_str)
            {
                if decoder_type == "DescriptAudioCodecConfig" {
                    let fishaudio_weights = model_path.join("model.safetensors");
                    if !fishaudio_weights.is_file() {
                        return Err(AudioError::Runtime(format!(
                            "missing exported FishAudio decoder weights '{}'",
                            fishaudio_weights.display()
                        )));
                    }
                    let parsed_fishaudio = parse_fishaudio_tts_config_json(tts_config)?;
                    let cfg = &parsed_fishaudio.audio_decoder_config;
                    let total_codebooks = cfg.n_codebooks.checked_add(1).ok_or(AudioError::Runtime(
                        "FishAudio codebook count overflow while building runtime config".to_string(),
                    ))?;
                    let codebook_size_i32 = i32::try_from(cfg.codebook_size).map_err(|_| {
                        AudioError::Runtime("FishAudio codebook_size exceeds i32 kernel range".to_string())
                    })?;
                    if codebook_size_i32 <= 1 {
                        return Err(AudioError::InvalidTokenCardinality);
                    }

                    let parsed = RuntimeConfigJson {
                        r#type: Some("nanocodec_fsq".to_string()),
                        sample_rate: cfg.samplerate,
                        num_groups: total_codebooks,
                        num_levels_per_group: vec![codebook_size_i32],
                        eps: default_eps(),
                        output_packing: RuntimePacking::CodebookMajor,
                        decoder: None,
                    };
                    let mut runtime = Self::from_runtime_config_json(parsed)?;
                    runtime.fishaudio_decoder = Some(build_fishaudio_codec_graph(cfg, &fishaudio_weights)?);
                    return Ok(runtime);
                }
            }
            let parsed_tts = parse_lalamo_tts_config_json(tts_config)?;
            let cfg = &parsed_tts.audio_decoder_config;
            let decoder_json =
                build_decoder_json_from_lalamo_export(&parsed_tts, &model_path.join("model.safetensors"))?;

            let parsed = RuntimeConfigJson {
                r#type: Some("nanocodec_fsq".to_string()),
                sample_rate: cfg.samplerate,
                num_groups: cfg.quantizer_config.num_groups,
                num_levels_per_group: cfg.quantizer_config.quantizer_config.num_levels.clone(),
                eps: cfg.quantizer_config.quantizer_config.eps,
                output_packing: RuntimePacking::CodebookMajor,
                decoder: Some(decoder_json),
            };

            return Self::from_runtime_config_json(parsed);
        }

        Self::from_tts_config_value(tts_config)
    }

    fn from_runtime_config_json(parsed: RuntimeConfigJson) -> AudioResult<Self> {
        let mut config = Self::new(
            parsed.sample_rate,
            parsed.num_groups,
            parsed.num_levels_per_group.into_boxed_slice(),
            parsed.eps,
            parsed.output_packing.into(),
        )?;
        if let Some(decoder_json) = parsed.decoder {
            config.decoder = Some(NanoCodecDecoderGraph::try_from(decoder_json)?);
        }
        Ok(config)
    }

    pub fn new(
        sample_rate: u32,
        num_groups: usize,
        num_levels_per_group: Box<[i32]>,
        eps: f32,
        output_packing: AudioTokenPacking,
    ) -> AudioResult<Self> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidSampleRate);
        }
        if num_groups == 0 || num_levels_per_group.is_empty() {
            return Err(AudioError::InvalidTokenCardinality);
        }
        if !eps.is_finite() || !(0.0..1.0).contains(&eps) {
            return Err(AudioError::Runtime("eps must be finite and satisfy 0.0 <= eps < 1.0".to_string()));
        }
        for &level in num_levels_per_group.iter() {
            if level <= 1 {
                return Err(AudioError::InvalidTokenCardinality);
            }
        }

        let codebook_dim_per_group = num_levels_per_group.len();
        let channels = num_groups
            .checked_mul(codebook_dim_per_group)
            .ok_or(AudioError::Runtime("num_groups * codebook_dim_per_group overflow".to_string()))?;
        let dim_base_index = compute_dim_base_index(&num_levels_per_group)?;

        let mut codec_cardinality_u64 = 1_u64;
        for &level in num_levels_per_group.iter() {
            codec_cardinality_u64 = codec_cardinality_u64
                .checked_mul(level as u64)
                .ok_or(AudioError::Runtime("codec cardinality overflow".to_string()))?;
        }

        if codec_cardinality_u64 > u32::MAX as u64 {
            return Err(AudioError::Runtime("codec cardinality exceeds u32 range".to_string()));
        }

        let codec_cardinality = codec_cardinality_u64 as u32;
        if codec_cardinality_u64 > (i32::MAX as u64 + 1) {
            return Err(AudioError::Runtime("codec cardinality exceeds i32 token kernel range".to_string()));
        }

        Ok(Self {
            sample_rate,
            num_groups,
            num_levels_per_group,
            dim_base_index: dim_base_index.into_boxed_slice(),
            codebook_dim_per_group,
            channels,
            codec_cardinality,
            eps,
            output_packing,
            decoder: None,
            fishaudio_decoder: None,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    pub fn num_levels_per_group(&self) -> &[i32] {
        &self.num_levels_per_group
    }

    pub fn dim_base_index(&self) -> &[i32] {
        &self.dim_base_index
    }

    pub fn codebook_dim_per_group(&self) -> usize {
        self.codebook_dim_per_group
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn codec_cardinality(&self) -> u32 {
        self.codec_cardinality
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn output_packing(&self) -> AudioTokenPacking {
        self.output_packing
    }

    pub fn decoder(&self) -> Option<&NanoCodecDecoderGraph> {
        self.decoder.as_ref()
    }

    fn fishaudio_decoder(&self) -> Option<&FishAudioCodecGraph> {
        self.fishaudio_decoder.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct NanoCodecFsqRuntime {
    config: NanoCodecFsqRuntimeConfig,
}

impl NanoCodecFsqRuntime {
    pub fn new(config: NanoCodecFsqRuntimeConfig) -> Self {
        Self {
            config,
        }
    }

    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value(tts_config)?))
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(tts_config, model_path)?))
    }

    pub fn config(&self) -> &NanoCodecFsqRuntimeConfig {
        &self.config
    }

    fn create_context() -> AudioResult<Rc<<Metal as Backend>::Context>> {
        <Metal as Backend>::Context::new()
            .map_err(|err| AudioError::Runtime(format!("failed to create metal audio context: {err}")))
    }
}

impl AudioCodecRuntime for NanoCodecFsqRuntime {
    fn encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> AudioResult<AudioTokenGrid> {
        if self.config.decoder().is_some() || self.config.fishaudio_decoder().is_some() {
            return Err(AudioError::Runtime("encode is not supported when a decoder graph is configured".to_string()));
        }

        if pcm.sample_rate() != self.config.sample_rate() {
            return Err(AudioError::Runtime(format!(
                "pcm sample-rate mismatch: expected {}, got {}",
                self.config.sample_rate(),
                pcm.sample_rate()
            )));
        }

        let (padded_input, lengths_usize, lengths_i32, frames) = pack_pcm_to_padded(pcm, self.config.channels())?;
        let batch_size = pcm.batch_size();

        if batch_size == 0 || frames == 0 {
            let empty_tokens = Vec::<u32>::new().into_boxed_slice();
            let grid = AudioTokenGrid::new(
                empty_tokens,
                batch_size,
                self.config.num_groups(),
                frames,
                lengths_usize.into_boxed_slice(),
                self.config.output_packing(),
            )?;
            return Ok(grid);
        }

        let context = Self::create_context()?;
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqEncodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq encode kernel: {err}")))?;

        let mut input = context.create_array(
            &[batch_size, self.config.channels(), frames],
            DataType::F32,
            "nanocodec_fsq_encode_input",
        );
        input.as_slice_mut::<f32>().copy_from_slice(&padded_input);

        let mut lengths = context.create_array(&[batch_size], DataType::I32, "nanocodec_fsq_encode_lengths");
        lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let tokens = context.create_array(
            &[batch_size, self.config.num_groups(), frames],
            DataType::I32,
            "nanocodec_fsq_encode_tokens",
        );

        let command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;

        command_buffer.with_compute_encoder(|compute_encoder| {
            kernel.encode(
                input.buffer(),
                tokens.buffer(),
                lengths.buffer(),
                num_groups_i32,
                frames_i32,
                codebook_dim_i32,
                self.config.num_levels_per_group(),
                self.config.dim_base_index(),
                self.config.eps(),
                batch_size_i32,
                compute_encoder,
            );
        });

        command_buffer.submit();
        command_buffer.wait_until_completed();

        let mut tokens_u32 = vec![0_u32; tokens.num_elements()];
        for (index, &token) in tokens.as_slice::<i32>().iter().enumerate() {
            if token < 0 {
                return Err(AudioError::Runtime(format!(
                    "fsq encode returned negative token at index {index}: {token}"
                )));
            }

            let token_u32 = token as u32;
            if token_u32 >= self.config.codec_cardinality() {
                return Err(AudioError::InvalidCodecToken {
                    token: token_u32,
                    cardinality: self.config.codec_cardinality(),
                });
            }
            tokens_u32[index] = token_u32;
        }

        let codebook_major = AudioTokenGrid::new(
            tokens_u32.into_boxed_slice(),
            batch_size,
            self.config.num_groups(),
            frames,
            lengths_usize.into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )?;

        Ok(codebook_major.to_packing(self.config.output_packing()))
    }

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<AudioPcmBatch> {
        if tokens.codebooks() != self.config.num_groups() {
            return Err(AudioError::Runtime(format!(
                "token codebook mismatch: expected {}, got {}",
                self.config.num_groups(),
                tokens.codebooks()
            )));
        }

        let batch_size = tokens.batch_size();
        let frames = tokens.frames();
        let lengths_usize = tokens.lengths().to_vec();
        let lengths_i32 = convert_lengths_to_i32(&lengths_usize, frames)?;

        if batch_size == 0 || frames == 0 {
            let channels = if let Some(decoder) = self.config.decoder() {
                decoder.output_channels()
            } else if self.config.fishaudio_decoder().is_some() {
                1
            } else {
                self.config.channels()
            };
            let out_lengths = if let Some(decoder) = self.config.decoder() {
                lengths_usize
                    .iter()
                    .map(|&length| {
                        length
                            .checked_mul(decoder.upsample_factor())
                            .ok_or(AudioError::Runtime("decoder length scaling overflow".to_string()))
                    })
                    .collect::<AudioResult<Vec<_>>>()?
            } else if let Some(decoder) = self.config.fishaudio_decoder() {
                lengths_usize
                    .iter()
                    .map(|&length| {
                        length
                            .checked_mul(decoder.upsample_factor)
                            .ok_or(AudioError::Runtime("decoder length scaling overflow".to_string()))
                    })
                    .collect::<AudioResult<Vec<_>>>()?
            } else {
                lengths_usize.clone()
            };
            return AudioPcmBatch::new(
                Vec::<f32>::new().into_boxed_slice(),
                self.config.sample_rate(),
                channels,
                out_lengths.into_boxed_slice(),
            );
        }

        let codebook_major = tokens.to_packing(AudioTokenPacking::CodebookMajor);
        if let Some(fishaudio) = self.config.fishaudio_decoder() {
            for &token in codebook_major.tokens() {
                if token >= self.config.codec_cardinality() {
                    return Err(AudioError::InvalidCodecToken {
                        token,
                        cardinality: self.config.codec_cardinality(),
                    });
                }
            }
            let decoded = fishaudio.decode_padded(
                codebook_major.tokens(),
                &lengths_usize,
                batch_size,
                codebook_major.codebooks(),
                frames,
            )?;
            let samples =
                unpack_padded_to_pcm(&decoded.samples, batch_size, decoded.channels, decoded.frames, &decoded.lengths)?;
            return AudioPcmBatch::new(
                samples.into_boxed_slice(),
                self.config.sample_rate(),
                decoded.channels,
                decoded.lengths.into_boxed_slice(),
            );
        }

        let mut tokens_i32 = vec![0_i32; codebook_major.tokens().len()];
        for (index, &token) in codebook_major.tokens().iter().enumerate() {
            if token >= self.config.codec_cardinality() {
                return Err(AudioError::InvalidCodecToken {
                    token,
                    cardinality: self.config.codec_cardinality(),
                });
            }
            if token > i32::MAX as u32 {
                return Err(AudioError::Runtime(format!("token at index {index} exceeds i32 kernel range: {token}")));
            }
            tokens_i32[index] = token as i32;
        }

        let context = Self::create_context()?;
        let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqDecodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq decode kernel: {err}")))?;

        let mut tokens_array = context.create_array(
            &[batch_size, self.config.num_groups(), frames],
            DataType::I32,
            "nanocodec_fsq_decode_tokens",
        );
        tokens_array.as_slice_mut::<i32>().copy_from_slice(&tokens_i32);

        let mut lengths_array = context.create_array(&[batch_size], DataType::I32, "nanocodec_fsq_decode_lengths");
        lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths_i32);

        let output = context.create_array(
            &[batch_size, self.config.channels(), frames],
            DataType::F32,
            "nanocodec_fsq_decode_output",
        );

        let command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;

        command_buffer.with_compute_encoder(|compute_encoder| {
            kernel.encode(
                tokens_array.buffer(),
                output.buffer(),
                lengths_array.buffer(),
                num_groups_i32,
                frames_i32,
                codebook_dim_i32,
                self.config.num_levels_per_group(),
                batch_size_i32,
                compute_encoder,
            );
        });

        command_buffer.submit();
        command_buffer.wait_until_completed();

        let (padded_output, out_channels, out_frames, out_lengths) = if let Some(decoder) = self.config.decoder() {
            let decoded = decoder.decode_padded(
                output.as_slice::<f32>(),
                &lengths_usize,
                batch_size,
                self.config.channels(),
                frames,
            )?;
            (decoded.samples, decoded.channels, decoded.frames, decoded.lengths)
        } else {
            (output.as_slice::<f32>().to_vec(), self.config.channels(), frames, lengths_usize.clone())
        };

        let samples = unpack_padded_to_pcm(&padded_output, batch_size, out_channels, out_frames, &out_lengths)?;

        AudioPcmBatch::new(
            samples.into_boxed_slice(),
            self.config.sample_rate(),
            out_channels,
            out_lengths.into_boxed_slice(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        FishAudioCodecGraph, FishAudioConv1dLayer, FishAudioDecoderGraph, FishAudioNormLayer, FishAudioPostModule,
        FishAudioVectorQuantizer, MatrixF32, NanoCodecFsqRuntimeConfig, RuntimePacking, Tensor3Json,
        convert_lalamo_transpose_weight_oih_to_iog, pack_pcm_to_padded, parse_lalamo_tts_config_json,
        parse_runtime_config_json, unpack_padded_to_pcm,
    };
    use crate::audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioTokenPacking};

    #[test]
    fn config_rejects_invalid_eps() {
        let config = NanoCodecFsqRuntimeConfig::new(
            24_000,
            2,
            vec![8, 6].into_boxed_slice(),
            1.1,
            AudioTokenPacking::FrameMajor,
        );

        assert!(matches!(config, Err(AudioError::Runtime(_))));
    }

    #[test]
    fn pack_and_unpack_round_trip_variable_lengths() {
        let channels = 4usize;
        let lengths = vec![3usize, 1usize];
        let samples: Vec<f32> =
            (0..(lengths.iter().sum::<usize>() * channels)).map(|index| (index as f32 * 0.17 - 1.2).sin()).collect();

        let batch = AudioPcmBatch::new(samples.clone().into_boxed_slice(), 24_000, channels, lengths.clone().into())
            .expect("pcm");

        let (padded, got_lengths, _got_lengths_i32, frames) = pack_pcm_to_padded(&batch, channels).expect("pack");
        assert_eq!(frames, 3);
        assert_eq!(got_lengths, lengths);

        let unpacked = unpack_padded_to_pcm(&padded, 2, channels, frames, &lengths).expect("unpack");
        assert_eq!(samples, unpacked);
    }

    #[test]
    fn parses_runtime_config_from_nested_audio_codec() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "nanocodec_fsq",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6],
                "output_packing": "codebook_major"
            }
        });

        let parsed = parse_runtime_config_json(&tts_config).expect("parse runtime json");
        assert_eq!(parsed.sample_rate, 24_000);
        assert_eq!(parsed.num_groups, 2);
        assert_eq!(parsed.num_levels_per_group, vec![8, 6]);
        assert_eq!(parsed.eps, 1e-3);
        assert_eq!(parsed.output_packing, RuntimePacking::CodebookMajor);
    }

    #[test]
    fn runtime_config_builder_rejects_unsupported_type() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "other_codec",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6]
            }
        });

        let error = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect_err("must fail");
        match error {
            AudioError::Runtime(message) => assert!(message.contains("unsupported audio codec type")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn runtime_config_can_include_decoder_graph() {
        let tts_config = serde_json::json!({
            "audio_codec": {
                "type": "nanocodec_fsq",
                "sample_rate": 24000,
                "num_groups": 2,
                "num_levels_per_group": [8, 6],
                "decoder": {
                    "pre_conv": {
                        "weight": {
                            "shape": [2, 2, 1],
                            "values": [1.0, 0.0, 0.0, 1.0]
                        },
                        "bias": [0.0, 0.0]
                    },
                    "stages": [{
                        "activation_alpha": [1.0],
                        "upsample_conv": {
                            "weight": {
                                "shape": [2, 1, 4],
                                "values": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                            },
                            "bias": [0.0, 0.0],
                            "stride": 2,
                            "groups": 2
                        }
                    }],
                    "post_activation_alpha": [1.0],
                    "post_conv": {
                        "weight": {
                            "shape": [1, 2, 1],
                            "values": [1.0, 0.0]
                        },
                        "bias": [0.0]
                    }
                }
            }
        });

        let config = NanoCodecFsqRuntimeConfig::from_tts_config_value(&tts_config).expect("runtime config");
        assert!(config.decoder().is_some());
        let decoder = config.decoder().expect("decoder");
        assert_eq!(decoder.upsample_factor(), 2);
    }

    #[test]
    fn parses_lalamo_tts_config_shape() {
        let tts_config = serde_json::json!({
            "text_decoder_config": {"type": "StubTextDecoderConfig"},
            "audio_decoder_config": {
                "samplerate": 22050,
                "quantizer_config": {
                    "num_groups": 13,
                    "quantizer_config": {
                        "num_levels": [8, 7, 6, 6],
                        "eps": 1e-3
                    }
                },
                "decoder_config": {
                    "activation_config": {
                        "leaky_relu_negative_slope": 0.01
                    }
                },
                "base_channels": 864,
                "up_sample_rates": [7, 7, 6, 3, 2],
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilations": [1, 3, 5]
            },
            "vocoder_config": {},
            "activation_precision": "float32"
        });

        let parsed = parse_lalamo_tts_config_json(&tts_config).expect("parse");
        assert_eq!(parsed.audio_decoder_config.samplerate, 22050);
        assert_eq!(parsed.audio_decoder_config.quantizer_config.num_groups, 13);
        assert_eq!(parsed.audio_decoder_config.quantizer_config.quantizer_config.num_levels, vec![8, 7, 6, 6]);
        assert_eq!(parsed.audio_decoder_config.up_sample_rates, vec![7, 7, 6, 3, 2]);
    }

    #[test]
    fn fishaudio_dac_config_requires_model_weights() {
        let tts_config = serde_json::json!({
            "audio_decoder_config": {
                "type": "DescriptAudioCodecConfig",
                "samplerate": 44100,
                "n_codebooks": 9,
                "codebook_size": 1024,
                "semantic_codebook_size": 4096,
                "input_dim": 1024,
                "downsample_factor": [2, 2],
                "decoder_rates": [8, 8, 4, 2]
            }
        });

        let error = NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(&tts_config, Path::new("."))
            .expect_err("fishaudio runtime must require exported model.safetensors");
        match error {
            AudioError::Runtime(message) => {
                assert!(message.contains("model.safetensors"), "unexpected error message: {message}");
            },
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn fishaudio_runtime_decode_path_emits_pcm() {
        let mut config = NanoCodecFsqRuntimeConfig::new(
            44_100,
            2,
            vec![4].into_boxed_slice(),
            1e-3,
            AudioTokenPacking::CodebookMajor,
        )
        .expect("config");

        let one_by_one = FishAudioConv1dLayer {
            weight: vec![1.0],
            bias: vec![0.0],
            cin: 1,
            cout: 1,
            kernel_size: 1,
            dilation: 1,
            groups: 1,
        };

        config.fishaudio_decoder = Some(FishAudioCodecGraph {
            semantic_quantizer: FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 2,
                    cols: 1,
                    values: vec![0.0, 1.0],
                },
                out_proj: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![1.0],
                },
                out_bias: vec![0.0],
            },
            residual_quantizers: vec![FishAudioVectorQuantizer {
                codebook: MatrixF32 {
                    rows: 2,
                    cols: 1,
                    values: vec![0.0, 1.0],
                },
                out_proj: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![1.0],
                },
                out_bias: vec![0.0],
            }],
            post_module: FishAudioPostModule {
                rope_cosines: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![1.0],
                },
                rope_sines: MatrixF32 {
                    rows: 1,
                    cols: 1,
                    values: vec![0.0],
                },
                layers: Vec::new(),
                output_norm: FishAudioNormLayer {
                    scales: vec![1.0],
                    biases: None,
                    epsilon: 1e-5,
                    subtract_mean: true,
                },
                model_dim: 1,
                hidden_dim: 1,
            },
            decoder: FishAudioDecoderGraph {
                first_conv: one_by_one.clone(),
                upsample_blocks: Vec::new(),
                decoder_blocks: Vec::new(),
                final_snake_alpha: vec![1.0],
                final_conv: one_by_one,
                upsample_factor: 1,
            },
            codebook_size: 2,
            semantic_codebook_size: 2,
            input_dim: 1,
            total_codebooks: 2,
            upsample_factor: 1,
        });

        let runtime = super::NanoCodecFsqRuntime::new(config);
        let tokens = crate::audio::AudioTokenGrid::new(
            vec![
                // semantic codebook
                0, 1, 0, // residual codebook
                1, 0, 1,
            ]
            .into_boxed_slice(),
            1,
            2,
            3,
            vec![3usize].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .expect("token grid");

        let pcm = runtime.decode(&tokens).expect("decode");
        assert_eq!(pcm.sample_rate(), 44_100);
        assert_eq!(pcm.channels(), 1);
        assert_eq!(pcm.lengths(), &[3usize]);
        assert_eq!(pcm.samples().len(), 3);
    }

    #[test]
    fn transpose_weight_conversion_matches_expected_layout() {
        let weight_oih = Tensor3Json {
            shape: [2, 2, 2],
            values: vec![
                // out=0
                1.0, 2.0, // in_group=0
                3.0, 4.0, // in_group=1
                // out=1
                5.0, 6.0, // in_group=0
                7.0, 8.0, // in_group=1
            ],
        };

        let converted = convert_lalamo_transpose_weight_oih_to_iog(&weight_oih, 4, 2, 2).expect("convert");
        assert_eq!(converted.shape, [4, 1, 2]);
        assert_eq!(
            converted.values,
            vec![
                1.0, 2.0, // in=0, out_group=0
                3.0, 4.0, // in=1, out_group=0
                5.0, 6.0, // in=2, out_group=0
                7.0, 8.0, // in=3, out_group=0
            ]
        );
    }
}
