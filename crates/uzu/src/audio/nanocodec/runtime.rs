use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;

use super::{
    decoder::{
        CausalConv1dJson, CausalConvTranspose1dJson, NanoCodecDecoderGraph, NanoCodecHiFiGanResBlockJson,
        NanoCodecHiFiGanResLayerJson, NanoCodecResidualBlockJson, NanoCodecUpsampleStageJson, Tensor3Json,
    },
    fsq::compute_dim_base_index,
};
use crate::{
    DataType,
    array::{Array, ArrayContextExt, size_for_shape},
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid, AudioTokenPacking},
    backends::{
        common::{
            Backend, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
            CommandBufferInitial, CommandBufferPending, Context, CopyEncoder, Kernels,
            kernel::{
                ActivationKernel, AudioAddKernel, AudioCausalConv1dGroupedKernel,
                AudioCausalConv1dGroupedResidualKernel, AudioCausalConv1dKernel,
                AudioCausalConvTranspose1dCausalPadKernel, AudioConv1dKernel, AudioFsqDecodeKernel,
                AudioFsqEncodeKernel, AudioHalfSnakeKernel, AudioNormNcsKernel, AudioQuantizerDecodeKernel,
                AudioTransposeNscToNcsKernel,
            },
        },
        metal::Metal,
    },
    config::{
        ConfigDataType, DescriptAudioCodecConfig, DescriptAudioConvNeXtNormConfig, EmbeddingConfig,
        EmbeddingConfigCommon, InnerModelConfig, NanoCodecAudioDecoderConfig, TtsAudioDecoderConfig, TtsConfig,
    },
    encodable_block::{EncodingParameters, LayerExecutables, RMSNorm, Rope},
    forward_pass::{
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, RopeType, SharedBuffers},
    },
    parameters::ParameterLoader,
    utils::array_io::{
        read_array_to_f32_vec, write_f32_slice_into_array as write_f32_slice_to_array, write_i32_slice_into_array,
    },
};

mod loaders;

#[cfg(test)]
use loaders::{
    convert_lalamo_transpose_weight_oih_to_iog, resolve_descript_audio_codec_vocoder_data_type,
};
use loaders::{build_nanocodec_decoder_graph_from_lalamo_config, load_audio_runtime_from_tts_config};
include!("runtime/profile.rs");
include!("runtime/stream.rs");
include!("runtime/structured.rs");

#[cfg(test)]
mod tests {
    include!("runtime/tests.rs");
}

type MetalCommandBuffer = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;
type MetalPendingCommandBuffer = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Pending;
type MetalCompletedCommandBuffer = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Completed;

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

fn checked_div_ceil(
    numerator: usize,
    denominator: usize,
) -> AudioResult<usize> {
    if denominator == 0 {
        return Err(AudioError::Runtime("division by zero".to_string()));
    }
    let addend = denominator.saturating_sub(1);
    numerator
        .checked_add(addend)
        .ok_or(AudioError::Runtime("ceil-division overflow".to_string()))
        .map(|value| value / denominator)
}

fn scale_lengths_i32_in_place(
    source: &[i32],
    destination: &mut [i32],
    factor: usize,
) -> AudioResult<()> {
    if destination.len() != source.len() {
        return Err(AudioError::Runtime(format!(
            "scaled length buffer mismatch: expected {}, got {}",
            source.len(),
            destination.len()
        )));
    }
    for (dst, &src) in destination.iter_mut().zip(source.iter()) {
        *dst = checked_mul_i32(src, factor)?;
    }
    Ok(())
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
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
    #[serde(default)]
    output_packing: RuntimePacking,
}

fn build_runtime_config_from_nanocodec_audio_decoder(
    config: &NanoCodecAudioDecoderConfig
) -> AudioResult<RuntimeConfigJson> {
    Ok(RuntimeConfigJson {
        sample_rate: config.samplerate,
        num_groups: config.quantizer_config.num_groups,
        num_levels_per_group: config.quantizer_config.quantizer_config.num_levels.clone(),
        eps: config.quantizer_config.quantizer_config.eps.unwrap_or_else(default_eps),
        output_packing: RuntimePacking::CodebookMajor,
    })
}

fn parse_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<TtsConfig> {
    serde_json::from_value(tts_config.clone())
        .map_err(|err| AudioError::Runtime(format!("failed to parse TTS config: {err}")))
}

fn default_negative_slope() -> f32 {
    0.01
}

fn default_decoder_eps() -> f32 {
    1e-9
}

enum LoadedTtsAudioRuntimeConfig {
    Standard(NanoCodecAudioDecoderConfig),
    StructuredDecoder {
        runtime: RuntimeConfigJson,
        decoder: StructuredAudioCodecGraph,
    },
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
    structured_decoder: Option<StructuredAudioCodecGraph>,
}

impl NanoCodecFsqRuntimeConfig {
    pub fn from_tts_config(tts_config: &TtsConfig) -> AudioResult<Self> {
        match &tts_config.audio_decoder_config {
            TtsAudioDecoderConfig::NanoCodecConfig {
                config,
            } => {
                let parsed = build_runtime_config_from_nanocodec_audio_decoder(config)?;
                Self::from_runtime_config_json(parsed)
            },
            TtsAudioDecoderConfig::DescriptAudioCodecConfig {
                ..
            } => Err(AudioError::Runtime(
                "DescriptAudioCodecConfig requires model weights; use from_tts_config_and_model_path".to_string(),
            )),
        }
    }

    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        Self::from_tts_config(&parse_tts_config_value(tts_config)?)
    }

    pub fn from_tts_config_and_model_path(
        tts_config: &TtsConfig,
        model_path: &Path,
    ) -> AudioResult<Self> {
        match load_audio_runtime_from_tts_config(tts_config, model_path)? {
            LoadedTtsAudioRuntimeConfig::Standard(config) => {
                let runtime_config = build_runtime_config_from_nanocodec_audio_decoder(&config)?;
                let mut runtime = Self::from_runtime_config_json(runtime_config)?;
                let weights_path = model_path.join("model.safetensors");
                if !weights_path.is_file() {
                    return Err(AudioError::Runtime(format!(
                        "missing exported NanoCodec decoder weights '{}'",
                        weights_path.display()
                    )));
                }
                runtime.decoder = Some(build_nanocodec_decoder_graph_from_lalamo_config(&config, &weights_path)?);
                Ok(runtime)
            },
            LoadedTtsAudioRuntimeConfig::StructuredDecoder {
                runtime: runtime_config,
                decoder,
            } => {
                let mut runtime = Self::from_runtime_config_json(runtime_config)?;
                runtime.structured_decoder = Some(decoder);
                Ok(runtime)
            },
        }
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Self::from_tts_config_and_model_path(&parse_tts_config_value(tts_config)?, model_path)
    }

    fn from_runtime_config_json(parsed: RuntimeConfigJson) -> AudioResult<Self> {
        Self::new(
            parsed.sample_rate,
            parsed.num_groups,
            parsed.num_levels_per_group.into_boxed_slice(),
            parsed.eps,
            parsed.output_packing.into(),
        )
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
            structured_decoder: None,
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

    pub fn semantic_codec_cardinality(&self) -> Option<usize> {
        self.structured_decoder.as_ref().map(|decoder| decoder.semantic_codebook_size)
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

    fn structured_decoder(&self) -> Option<&StructuredAudioCodecGraph> {
        self.structured_decoder.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct NanoCodecFsqRuntime {
    config: NanoCodecFsqRuntimeConfig,
    options: NanoCodecFsqRuntimeOptions,
    last_decode_profile: Arc<RwLock<Option<AudioDecodeProfile>>>,
}

impl NanoCodecFsqRuntime {
    pub fn new(config: NanoCodecFsqRuntimeConfig) -> Self {
        Self::new_with_options(config, NanoCodecFsqRuntimeOptions::default())
    }

    pub fn new_with_options(
        config: NanoCodecFsqRuntimeConfig,
        options: NanoCodecFsqRuntimeOptions,
    ) -> Self {
        if options.capture_single_decode {
            <Metal as Backend>::Context::enable_capture();
        }
        Self {
            config,
            options,
            last_decode_profile: Arc::new(RwLock::new(None)),
        }
    }

    pub fn from_tts_config(tts_config: &TtsConfig) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config(tts_config)?))
    }

    pub fn from_tts_config_value(tts_config: &serde_json::Value) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value(tts_config)?))
    }

    pub fn from_tts_config_and_model_path(
        tts_config: &TtsConfig,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_and_model_path(tts_config, model_path)?))
    }

    pub fn from_tts_config_value_and_model_path(
        tts_config: &serde_json::Value,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(tts_config, model_path)?))
    }

    pub fn from_tts_config_value_and_model_path_with_options(
        tts_config: &serde_json::Value,
        model_path: &Path,
        options: NanoCodecFsqRuntimeOptions,
    ) -> AudioResult<Self> {
        Ok(Self::new_with_options(
            NanoCodecFsqRuntimeConfig::from_tts_config_value_and_model_path(tts_config, model_path)?,
            options,
        ))
    }

    pub fn from_tts_config_and_model_path_with_options(
        tts_config: &TtsConfig,
        model_path: &Path,
        options: NanoCodecFsqRuntimeOptions,
    ) -> AudioResult<Self> {
        Ok(Self::new_with_options(
            NanoCodecFsqRuntimeConfig::from_tts_config_and_model_path(tts_config, model_path)?,
            options,
        ))
    }

    pub fn config(&self) -> &NanoCodecFsqRuntimeConfig {
        &self.config
    }

    pub fn options(&self) -> NanoCodecFsqRuntimeOptions {
        self.options
    }

    pub fn last_decode_profile(&self) -> Option<AudioDecodeProfile> {
        self.last_decode_profile.read().ok().and_then(|profile| profile.as_ref().cloned())
    }

    fn validate_fishaudio_token_delta(
        &self,
        tokens: &AudioTokenGrid,
        fishaudio: &StructuredAudioCodecGraph,
    ) -> AudioResult<()> {
        let semantic_cardinality = u32::try_from(fishaudio.semantic_codebook_size).map_err(|_| {
            AudioError::Runtime("FishAudio semantic codebook cardinality exceeds u32 range".to_string())
        })?;
        let residual_cardinality = u32::try_from(fishaudio.codebook_size).map_err(|_| {
            AudioError::Runtime("FishAudio residual codebook cardinality exceeds u32 range".to_string())
        })?;
        let codebook_major = tokens.to_packing(AudioTokenPacking::CodebookMajor);
        let frames = codebook_major.frames();
        for batch in 0..codebook_major.batch_size() {
            for codebook in 0..codebook_major.codebooks() {
                let cardinality = if codebook == 0 {
                    semantic_cardinality
                } else {
                    residual_cardinality
                };
                for frame in 0..frames {
                    let token = codebook_major.get(batch, codebook, frame);
                    if token >= cardinality {
                        return Err(AudioError::InvalidCodecToken {
                            token,
                            cardinality,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn decode_structured_stream_delta(
        &self,
        state: &mut AudioDecodeStreamState,
        fishaudio: &StructuredAudioCodecGraph,
        input_frames: usize,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if state.total_frames() == 0 {
            state.record_last_step_stats(input_frames, 0, 0);
            return Ok(super::decoder::DecodedPaddedAudio {
                samples: Vec::new(),
                channels: 1,
                frames: 0,
                lengths: vec![0usize; state.batch_size],
            });
        }

        let Some(context_frames) = fishaudio.streaming_decode_context_frames()? else {
            let full_grid = state.to_full_grid()?;
            let full_pcm = self.decode(&full_grid)?;
            state.record_last_step_stats(input_frames, 0, state.total_frames());
            return state.extract_delta_padded(&full_pcm);
        };
        let mut window_start = state.total_frames();
        for &emitted in &state.emitted_semantic_lengths {
            window_start = window_start.min(emitted.saturating_sub(context_frames));
        }
        let window_end = state.total_frames();
        let batch_size = state.batch_size;
        let codebooks = state.codebooks;
        let (window_tokens, window_lengths, window_frames) = state.flatten_window(window_start, window_end)?;
        let (decoded_window, decode_profile) = fishaudio.decode_padded(
            self.options,
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
            *last_decode_profile = decode_profile;
        }
        let audio_offset_frames = window_start
            .checked_mul(fishaudio.upsample_factor)
            .ok_or(AudioError::Runtime("stream audio offset overflow".to_string()))?;
        state.record_last_step_stats(input_frames, window_start, window_frames);
        state.extract_delta_from_padded_with_offset(&decoded_window, audio_offset_frames, fishaudio.upsample_factor)
    }

    pub fn begin_decode_stream(
        &self,
        batch_size: usize,
        codebooks: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        self.begin_decode_stream_with_options(batch_size, codebooks, AudioDecodeStreamingMode::IncrementalStateful, 256)
    }

    pub fn begin_decode_stream_with_options(
        &self,
        batch_size: usize,
        codebooks: usize,
        mode: AudioDecodeStreamingMode,
        max_workspace_frames: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        if codebooks != self.config.num_groups() {
            return Err(AudioError::Runtime(format!(
                "stream codebook mismatch: expected {}, got {}",
                self.config.num_groups(),
                codebooks
            )));
        }
        AudioDecodeStreamState::new(batch_size, codebooks, max_workspace_frames, mode)
    }

    pub fn decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        _is_final: bool,
    ) -> AudioResult<super::decoder::DecodedPaddedAudio> {
        if new_tokens.codebooks() != state.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                state.codebooks,
                new_tokens.codebooks()
            )));
        }
        if new_tokens.batch_size() != state.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                state.batch_size,
                new_tokens.batch_size()
            )));
        }
        if new_tokens.frames() == 0 {
            state.record_last_step_stats(0, state.total_frames(), 0);
            return Ok(super::decoder::DecodedPaddedAudio {
                samples: Vec::new(),
                channels: if let Some(decoder) = self.config.decoder() {
                    decoder.output_channels()
                } else if self.config.structured_decoder().is_some() {
                    1
                } else {
                    self.config.channels()
                },
                frames: 0,
                lengths: vec![0; state.batch_size],
            });
        }

        state.append_delta(new_tokens)?;
        if let Some(fishaudio) = self.config.structured_decoder() {
            self.validate_fishaudio_token_delta(new_tokens, fishaudio)?;
            return match state.mode {
                AudioDecodeStreamingMode::IncrementalStateful => {
                    self.decode_structured_stream_delta(state, fishaudio, new_tokens.frames())
                },
                AudioDecodeStreamingMode::PrefixFallback => {
                    let full_grid = state.to_full_grid()?;
                    let full_pcm = self.decode(&full_grid)?;
                    state.record_last_step_stats(new_tokens.frames(), 0, state.total_frames());
                    state.extract_delta_padded(&full_pcm)
                },
            };
        }

        let full_grid = state.to_full_grid()?;
        let full_pcm = self.decode(&full_grid)?;
        state.record_last_step_stats(new_tokens.frames(), 0, state.total_frames());
        state.extract_delta_padded(&full_pcm)
    }

    pub(crate) fn submit_decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> AudioResult<Option<PendingStreamPcmChunk>> {
        if is_final || !self.options.async_stream_delivery_enabled() {
            return Ok(None);
        }
        if new_tokens.codebooks() != state.codebooks {
            return Err(AudioError::Runtime(format!(
                "stream delta codebook mismatch: expected {}, got {}",
                state.codebooks,
                new_tokens.codebooks()
            )));
        }
        if new_tokens.batch_size() != state.batch_size {
            return Err(AudioError::Runtime(format!(
                "stream delta batch mismatch: expected {}, got {}",
                state.batch_size,
                new_tokens.batch_size()
            )));
        }
        if new_tokens.frames() == 0 {
            return Ok(None);
        }

        let Some(fishaudio) = self.config.structured_decoder() else {
            return Ok(None);
        };
        if state.mode != AudioDecodeStreamingMode::IncrementalStateful {
            return Ok(None);
        }
        let Some(context_frames) = fishaudio.streaming_decode_context_frames()? else {
            return Ok(None);
        };

        self.validate_fishaudio_token_delta(new_tokens, fishaudio)?;
        state.append_delta(new_tokens)?;

        let mut window_start = state.total_frames();
        for &emitted in &state.emitted_semantic_lengths {
            window_start = window_start.min(emitted.saturating_sub(context_frames));
        }
        let window_end = state.total_frames();
        let batch_size = state.batch_size;
        let codebooks = state.codebooks;
        let previous_audio_lengths = state.emitted_audio_lengths.clone().into_boxed_slice();
        let semantic_lengths = state.semantic_lengths.clone().into_boxed_slice();
        let (window_tokens, window_lengths, window_frames) = state.flatten_window(window_start, window_end)?;
        let submitted = fishaudio.submit_decode_padded(
            self.options,
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        let audio_offset_frames = window_start
            .checked_mul(fishaudio.upsample_factor)
            .ok_or(AudioError::Runtime("stream audio offset overflow".to_string()))?;
        state.record_last_step_stats(new_tokens.frames(), window_start, window_frames);
        state.mark_submitted_audio_window(&semantic_lengths, fishaudio.upsample_factor)?;

        Ok(Some(PendingStreamPcmChunk {
            runtime: self.clone(),
            submitted,
            previous_audio_lengths,
            semantic_lengths,
            audio_offset_frames,
            upsample_factor: fishaudio.upsample_factor,
            step_stats: state.last_step_stats(),
        }))
    }

    pub fn decoded_padded_to_pcm_batch(
        &self,
        decoded: &super::decoder::DecodedPaddedAudio,
    ) -> AudioResult<AudioPcmBatch> {
        let samples = unpack_padded_to_pcm(
            &decoded.samples,
            decoded.lengths.len(),
            decoded.channels,
            decoded.frames,
            &decoded.lengths,
        )?;
        AudioPcmBatch::new(
            samples.into_boxed_slice(),
            self.config.sample_rate(),
            decoded.channels,
            decoded.lengths.clone().into_boxed_slice(),
        )
    }

    pub fn end_decode_stream(
        &self,
        _state: AudioDecodeStreamState,
    ) -> AudioResult<()> {
        Ok(())
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
        if self.config.decoder().is_some() || self.config.structured_decoder().is_some() {
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

        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?
            .start_encoding();

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;
        {
            let input_buffer = input.buffer();
            let input_buffer = input_buffer.borrow();
            let tokens_buffer = tokens.buffer();
            let mut tokens_buffer = tokens_buffer.borrow_mut();
            let lengths_buffer = lengths.buffer();
            let lengths_buffer = lengths_buffer.borrow();

            command_buffer.with_compute_encoder(|compute_encoder| {
                kernel.encode(
                    &*input_buffer,
                    &mut *tokens_buffer,
                    &*lengths_buffer,
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
        }

        let command_buffer = command_buffer.end_encoding().submit();
        command_buffer
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for FSQ encode command buffer: {err}")))?;

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
        if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
            *last_decode_profile = None;
        }
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
            } else if self.config.structured_decoder().is_some() {
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
            } else if let Some(decoder) = self.config.structured_decoder() {
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
        if let Some(fishaudio) = self.config.structured_decoder() {
            let semantic_cardinality = u32::try_from(fishaudio.semantic_codebook_size).map_err(|_| {
                AudioError::Runtime("FishAudio semantic codebook cardinality exceeds u32 range".to_string())
            })?;
            let residual_cardinality = u32::try_from(fishaudio.codebook_size).map_err(|_| {
                AudioError::Runtime("FishAudio residual codebook cardinality exceeds u32 range".to_string())
            })?;

            for batch in 0..batch_size {
                for codebook in 0..codebook_major.codebooks() {
                    let cardinality = if codebook == 0 {
                        semantic_cardinality
                    } else {
                        residual_cardinality
                    };
                    for frame in 0..frames {
                        let token = codebook_major.get(batch, codebook, frame);
                        if token >= cardinality {
                            return Err(AudioError::InvalidCodecToken {
                                token,
                                cardinality,
                            });
                        }
                    }
                }
            }
            let (decoded, decode_profile) = fishaudio.decode_padded(
                self.options,
                codebook_major.tokens(),
                &lengths_usize,
                batch_size,
                codebook_major.codebooks(),
                frames,
            )?;
            if let Ok(mut last_decode_profile) = self.last_decode_profile.write() {
                *last_decode_profile = decode_profile;
            }
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

        let mut command_buffer = context
            .create_command_buffer()
            .map_err(|err| AudioError::Runtime(format!("failed to create command buffer: {err}")))?
            .start_encoding();

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;
        {
            let tokens_buffer = tokens_array.buffer();
            let tokens_buffer = tokens_buffer.borrow();
            let output_buffer = output.buffer();
            let mut output_buffer = output_buffer.borrow_mut();
            let lengths_buffer = lengths_array.buffer();
            let lengths_buffer = lengths_buffer.borrow();

            command_buffer.with_compute_encoder(|compute_encoder| {
                kernel.encode(
                    &*tokens_buffer,
                    &mut *output_buffer,
                    &*lengths_buffer,
                    num_groups_i32,
                    frames_i32,
                    codebook_dim_i32,
                    self.config.num_levels_per_group(),
                    self.config.dim_base_index(),
                    batch_size_i32,
                    compute_encoder,
                );
            });
        }

        let command_buffer = command_buffer.end_encoding().submit();
        command_buffer
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for FSQ decode command buffer: {err}")))?;

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
