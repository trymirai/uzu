use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    path::Path,
    rc::Rc,
};

use serde::Deserialize;

use crate::{
    DataType,
    array::{ArrayContextExt, allocation_as_slice, size_for_shape},
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid},
    backends::common::{
        Backend, Context, Encoder, Kernels, Pending,
        kernel::{
            ActivationKernel, AudioAddKernel, AudioCausalConv1dGroupedKernel, AudioCausalConv1dGroupedResidualKernel,
            AudioCausalConv1dKernel, AudioCausalConvTranspose1dCausalPadKernel, AudioConv1dKernel,
            AudioFsqDecodeKernel, AudioFsqEncodeKernel, AudioHalfSnakeKernel, AudioNormNcsKernel,
            AudioQuantizerDecodeKernel,
        },
    },
    config::{
        ConfigDataType, DescriptAudioCodecConfig, DescriptAudioConvNeXtNormConfig, EmbeddingConfig,
        EmbeddingConfigCommon, InnerModelConfig, TtsAudioDecoderConfig, TtsConfig,
    },
    encodable_block::{Decoder, EncodingParameters, LayerExecutables, RMSNorm},
    forward_pass::{model_shape::ModelShape, state::SharedBuffers},
    parameters::ParameterLoader,
};

mod loaders;
mod profile;
mod stream;
mod structured;
mod support;

use loaders::load_audio_runtime_from_tts_config;
pub(crate) use profile::PendingStreamPcmChunk;
pub use stream::{AudioDecodeStepStats, AudioDecodeStreamState};
use stream::{extract_delta_from_padded_with_offset_snapshot, pack_pcm_to_padded, unpack_padded_to_pcm};
use structured::{StructuredAudioCodecGraph, StructuredAudioRuntimeResources};
use support::{DecodedPaddedAudio, checked_product, convert_lengths_to_i32, usize_to_i32};

fn default_eps() -> f32 {
    1e-3
}

fn compute_dim_base_index(num_levels: &[i32]) -> AudioResult<Box<[i32]>> {
    if num_levels.is_empty() {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let mut out = vec![0_i32; num_levels.len()];
    let mut base = 1_i32;

    for (index, &levels) in num_levels.iter().enumerate() {
        if levels <= 1 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        out[index] = base;
        base = base.checked_mul(levels).ok_or(AudioError::Runtime("dim_base_index overflow".to_string()))?;
    }

    Ok(out.into_boxed_slice())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeConfigJson {
    sample_rate: u32,
    num_groups: usize,
    num_levels_per_group: Vec<i32>,
    #[serde(default = "default_eps")]
    eps: f32,
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
    structured_decoder: Option<StructuredAudioCodecGraph>,
}

impl NanoCodecFsqRuntimeConfig {
    pub fn from_tts_config_and_model_path(
        tts_config: &TtsConfig,
        model_path: &Path,
    ) -> AudioResult<Self> {
        let (runtime_config, decoder) = load_audio_runtime_from_tts_config(tts_config, model_path)?;
        let mut runtime = Self::from_runtime_config_json(runtime_config)?;
        runtime.structured_decoder = Some(decoder);
        Ok(runtime)
    }

    fn from_runtime_config_json(parsed: RuntimeConfigJson) -> AudioResult<Self> {
        Self::new(parsed.sample_rate, parsed.num_groups, parsed.num_levels_per_group.into_boxed_slice(), parsed.eps)
    }

    pub fn new(
        sample_rate: u32,
        num_groups: usize,
        num_levels_per_group: Box<[i32]>,
        eps: f32,
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
            dim_base_index,
            codebook_dim_per_group,
            channels,
            codec_cardinality,
            eps,
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

    fn structured_decoder(&self) -> Option<&StructuredAudioCodecGraph> {
        self.structured_decoder.as_ref()
    }
}

pub struct NanoCodecFsqRuntime<B: Backend> {
    config: NanoCodecFsqRuntimeConfig,
    structured_resources: RefCell<Option<Rc<StructuredAudioRuntimeResources<B>>>>,
}

impl<B: Backend> Clone for NanoCodecFsqRuntime<B> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            structured_resources: self.structured_resources.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for NanoCodecFsqRuntime<B> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("NanoCodecFsqRuntime").field("config", &self.config).finish()
    }
}

impl<B: Backend> NanoCodecFsqRuntime<B> {
    pub fn new(config: NanoCodecFsqRuntimeConfig) -> Self {
        Self {
            config,
            structured_resources: RefCell::new(None),
        }
    }

    pub fn from_tts_config_and_model_path(
        tts_config: &TtsConfig,
        model_path: &Path,
    ) -> AudioResult<Self> {
        Ok(Self::new(NanoCodecFsqRuntimeConfig::from_tts_config_and_model_path(tts_config, model_path)?))
    }

    pub fn config(&self) -> &NanoCodecFsqRuntimeConfig {
        &self.config
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
        let frames = tokens.frames();
        for batch in 0..tokens.batch_size() {
            for codebook in 0..tokens.codebooks() {
                let cardinality = if codebook == 0 {
                    semantic_cardinality
                } else {
                    residual_cardinality
                };
                for frame in 0..frames {
                    let token = tokens.get(batch, codebook, frame);
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

    fn fsq_decode_padded(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<DecodedPaddedAudio> {
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
            return Ok(DecodedPaddedAudio {
                samples: Vec::new(),
                channels: self.config.channels(),
                frames: 0,
                lengths: lengths_usize,
            });
        }

        let mut tokens_i32 = vec![0_i32; tokens.tokens().len()];
        for (index, &token) in tokens.tokens().iter().enumerate() {
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
        let kernel = <B::Kernels as Kernels>::AudioFsqDecodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq decode kernel: {err}")))?;

        let tokens_allocation = context
            .create_array_from(
                &[batch_size, self.config.num_groups(), frames],
                &tokens_i32,
                "nanocodec_fsq_decode_tokens",
            )
            .into_allocation();

        let lengths_allocation =
            context.create_array_from(&[batch_size], &lengths_i32, "nanocodec_fsq_decode_lengths").into_allocation();

        let mut output = context
            .create_array_uninitialized(
                &[batch_size, self.config.channels(), frames],
                DataType::F32,
                "nanocodec_fsq_decode_output",
            )
            .into_allocation();

        let mut encoder = Encoder::new(context.as_ref())
            .map_err(|err| AudioError::Runtime(format!("failed to create encoder: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;
        kernel.encode(
            &tokens_allocation,
            &mut output,
            &lengths_allocation,
            num_groups_i32,
            frames_i32,
            codebook_dim_i32,
            self.config.num_levels_per_group(),
            self.config.dim_base_index(),
            batch_size_i32,
            &mut encoder,
        );

        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for FSQ decode command buffer: {err}")))?;

        Ok(DecodedPaddedAudio {
            samples: allocation_as_slice::<f32, B>(&output).to_vec(),
            channels: self.config.channels(),
            frames,
            lengths: lengths_usize,
        })
    }

    fn fsq_encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> AudioResult<AudioTokenGrid> {
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
            )?;
            return Ok(grid);
        }

        let context = Self::create_context()?;
        let kernel = <B::Kernels as Kernels>::AudioFsqEncodeKernel::new(&context, DataType::F32)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize fsq encode kernel: {err}")))?;

        let input = context
            .create_array_from(
                &[batch_size, self.config.channels(), frames],
                &padded_input,
                "nanocodec_fsq_encode_input",
            )
            .into_allocation();

        let lengths =
            context.create_array_from(&[batch_size], &lengths_i32, "nanocodec_fsq_encode_lengths").into_allocation();

        let mut tokens = context
            .create_array_uninitialized(
                &[batch_size, self.config.num_groups(), frames],
                DataType::I32,
                "nanocodec_fsq_encode_tokens",
            )
            .into_allocation();

        let mut encoder = Encoder::new(context.as_ref())
            .map_err(|err| AudioError::Runtime(format!("failed to create encoder: {err}")))?;

        let num_groups_i32 = usize_to_i32(self.config.num_groups(), "num_groups")?;
        let frames_i32 = usize_to_i32(frames, "frames")?;
        let codebook_dim_i32 = usize_to_i32(self.config.codebook_dim_per_group(), "codebook_dim_per_group")?;
        let batch_size_i32 = usize_to_i32(batch_size, "batch_size")?;
        kernel.encode(
            &input,
            &mut tokens,
            &lengths,
            num_groups_i32,
            frames_i32,
            codebook_dim_i32,
            self.config.num_levels_per_group(),
            self.config.dim_base_index(),
            self.config.eps(),
            batch_size_i32,
            &mut encoder,
        );

        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|err| AudioError::Runtime(format!("failed to wait for FSQ encode command buffer: {err}")))?;

        let encoded_tokens = allocation_as_slice::<i32, B>(&tokens).to_vec();
        let mut tokens_u32 = vec![0_u32; encoded_tokens.len()];
        for (index, &token) in encoded_tokens.iter().enumerate() {
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

        AudioTokenGrid::new(
            tokens_u32.into_boxed_slice(),
            batch_size,
            self.config.num_groups(),
            frames,
            lengths_usize.into_boxed_slice(),
        )
    }

    pub fn begin_decode_stream(
        &self,
        batch_size: usize,
        codebooks: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        self.begin_decode_stream_with_options(batch_size, codebooks, 256)
    }

    pub fn begin_decode_stream_with_options(
        &self,
        batch_size: usize,
        codebooks: usize,
        max_workspace_frames: usize,
    ) -> AudioResult<AudioDecodeStreamState> {
        if codebooks != self.config.num_groups() {
            return Err(AudioError::Runtime(format!(
                "stream codebook mismatch: expected {}, got {}",
                self.config.num_groups(),
                codebooks
            )));
        }
        AudioDecodeStreamState::new(batch_size, codebooks, max_workspace_frames)
    }

    pub(crate) fn decoded_padded_to_pcm_batch(
        &self,
        decoded: &DecodedPaddedAudio,
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
        state: AudioDecodeStreamState,
    ) -> AudioResult<()> {
        drop(state);
        Ok(())
    }

    fn create_context() -> AudioResult<Rc<B::Context>> {
        B::Context::new().map_err(|err| AudioError::Runtime(format!("failed to create audio context: {err}")))
    }

    fn structured_resources(&self) -> AudioResult<Rc<StructuredAudioRuntimeResources<B>>> {
        if let Some(existing) = self.structured_resources.borrow().as_ref() {
            return Ok(existing.clone());
        }
        let context = Self::create_context()?;
        let created = Rc::new(StructuredAudioRuntimeResources::new(context));
        *self.structured_resources.borrow_mut() = Some(created.clone());
        Ok(created)
    }

    fn decode_structured_stream_delta(
        &self,
        state: &mut AudioDecodeStreamState,
        fishaudio: &StructuredAudioCodecGraph,
        input_frames: usize,
    ) -> AudioResult<DecodedPaddedAudio> {
        if state.total_frames() == 0 {
            state.record_last_step_stats(input_frames, 0, 0);
            return Ok(DecodedPaddedAudio {
                samples: Vec::new(),
                channels: 1,
                frames: 0,
                lengths: vec![0usize; state.batch_size],
            });
        }

        let resources = self.structured_resources()?;
        let Some(context_frames) = fishaudio.streaming_decode_context_frames(resources.as_ref())? else {
            let full_grid = state.to_full_grid()?;
            let full_padded = self.decode_padded(&full_grid)?;
            state.record_last_step_stats(input_frames, 0, state.total_frames());
            return state.extract_delta_from_padded_with_offset(&full_padded, 0, fishaudio.upsample_factor);
        };
        let mut window_start = state.total_frames();
        for &emitted in &state.emitted_semantic_lengths {
            window_start = window_start.min(emitted.saturating_sub(context_frames));
        }
        let window_end = state.total_frames();
        let batch_size = state.batch_size;
        let codebooks = state.codebooks;
        let (window_tokens, window_lengths, window_frames) = state.flatten_window(window_start, window_end)?;
        let decoded_window = fishaudio.decode_padded(
            resources.as_ref(),
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        let audio_offset_frames = window_start
            .checked_mul(fishaudio.upsample_factor)
            .ok_or(AudioError::Runtime("stream audio offset overflow".to_string()))?;
        state.record_last_step_stats(input_frames, window_start, window_frames);
        state.extract_delta_from_padded_with_offset(&decoded_window, audio_offset_frames, fishaudio.upsample_factor)
    }

    pub(crate) fn decode_padded(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<DecodedPaddedAudio> {
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

        if batch_size == 0 || frames == 0 {
            let channels = if self.config.structured_decoder().is_some() {
                1
            } else {
                self.config.channels()
            };
            let out_lengths = if let Some(decoder) = self.config.structured_decoder() {
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
            return Ok(DecodedPaddedAudio {
                samples: Vec::new(),
                channels,
                frames: out_lengths.iter().copied().max().unwrap_or(0),
                lengths: out_lengths,
            });
        }

        if let Some(fishaudio) = self.config.structured_decoder() {
            self.validate_fishaudio_token_delta(tokens, fishaudio)?;
            let resources = self.structured_resources()?;
            let decoded = fishaudio.decode_padded(
                resources.as_ref(),
                tokens.tokens(),
                &lengths_usize,
                batch_size,
                tokens.codebooks(),
                frames,
            )?;
            return Ok(decoded);
        }

        self.fsq_decode_padded(tokens)
    }

    pub(crate) fn decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> AudioResult<DecodedPaddedAudio> {
        let _ = is_final;
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
            return Ok(DecodedPaddedAudio {
                samples: Vec::new(),
                channels: if self.config.structured_decoder().is_some() {
                    1
                } else {
                    self.config.channels()
                },
                frames: 0,
                lengths: vec![0; state.batch_size],
            });
        }

        if let Some(fishaudio) = self.config.structured_decoder() {
            self.validate_fishaudio_token_delta(new_tokens, fishaudio)?;
            state.append_delta(new_tokens)?;
            return self.decode_structured_stream_delta(state, fishaudio, new_tokens.frames());
        }

        state.append_delta(new_tokens)?;
        let full_grid = state.to_full_grid()?;
        let full_padded = self.fsq_decode_padded(&full_grid)?;
        state.record_last_step_stats(new_tokens.frames(), 0, state.total_frames());
        state.extract_delta_from_padded_with_offset(&full_padded, 0, 1)
    }

    pub(crate) fn submit_decode_stream_step(
        &self,
        state: &mut AudioDecodeStreamState,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> AudioResult<Option<PendingStreamPcmChunk<B>>> {
        if is_final {
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
        let resources = self.structured_resources()?;
        let Some(context_frames) = fishaudio.streaming_decode_context_frames(resources.as_ref())? else {
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
            resources.as_ref(),
            window_tokens,
            window_lengths,
            batch_size,
            codebooks,
            window_frames,
        )?;
        // The scratch buffers inside the workspace were encoded into the
        // submitted command buffer. Reset the workspace so the next decode
        // pass allocates fresh buffers instead of reusing ones the in-flight
        // command buffer may still be reading.
        resources.reset_for_pending();
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
}

impl<B: Backend> AudioCodecRuntime for NanoCodecFsqRuntime<B> {
    fn encode(
        &self,
        pcm: &AudioPcmBatch,
    ) -> AudioResult<AudioTokenGrid> {
        if self.config.structured_decoder().is_some() {
            return Err(AudioError::Runtime("encode is not supported when a decoder graph is configured".to_string()));
        }
        self.fsq_encode(pcm)
    }

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> AudioResult<AudioPcmBatch> {
        let decoded = self.decode_padded(tokens)?;
        self.decoded_padded_to_pcm_batch(&decoded)
    }
}
