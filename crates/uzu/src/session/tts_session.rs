#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

mod audio_backend;
mod backend_factory;
mod fishaudio;

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    ops::DerefMut,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use half::{bf16, f16};
use minijinja::{Environment, context};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tokenizers::Tokenizer;

use crate::{
    DataType,
    array::{ArrayCell, ArrayContextExt},
    audio::{
        AudioCodecRuntime, AudioGenerationContext, AudioPcmBatch, AudioTokenGrid, AudioTokenPacking,
        nanocodec::{AudioDecodeStepStats, AudioDecodeStreamState, AudioDecodeStreamingMode},
    },
    backends::{
        common::{
            Backend, Context as BackendContext, Kernels, NativeBuffer,
            kernel::{
                BufferArg, EmbeddingRowsSumKernel, RepetitionPenaltyKernel, TensorAddScaleKernel, TensorCopyKernel,
                TokenCopySampledKernel, TokenCopyToResultsKernel,
                kv_cache_update::KVCacheUpdate,
                matmul::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel, MatmulKernels},
            },
        },
        metal::Metal,
    },
    config::{InnerModelConfig, ModelMetadata, ModelType, TtsModelConfig, TtsTextDecoderConfig},
    encodable_block::{Decoder, EncodableBlock, EncodingParameters, Sampling as GpuSampling},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, SharedBuffers},
    },
    parameters::{ParameterLoader, read_safetensors_metadata},
    session::{
        config::{
            TextDecoderFollowupStrategy, TextDecoderRuntimeConfig, TextSamplingConfig, TtsChunkPolicy, TtsRunConfig,
            TtsSessionOptions, TtsVocoderStreamingMode,
        },
        parameter::{SamplingMethod, SamplingProcessingOrder},
        types::{Error, Input},
    },
};

use backend_factory::{build_audio_decoder_backend, build_text_decoder_backend};

const DEFAULT_STUB_SPEAKER_ID: &str = "speaker:0";
const DEFAULT_STUB_STYLE: &str = "interleave";
const DEFAULT_STUB_SEED: u64 = 123;
const DEFAULT_TTS_RANDOM_SEED: u64 = 123;
const DEFAULT_CHUNK_EMA_ALPHA: f64 = 0.2;
const DEFAULT_CHUNK_HYSTERESIS_FRACTION: f64 = 0.25;

fn unable_to_create_context<E: std::error::Error + 'static>(err: E) -> Error {
    Error::UnableToCreateContext(Box::new(err))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtsExecutionStats {
    pub semantic_decode_seconds: f64,
    pub audio_decode_seconds: f64,
    pub callback_seconds: f64,
    pub first_chunk_seconds: f64,
    pub command_buffers_submitted: usize,
    pub host_waits: usize,
    pub semantic_frames: usize,
    pub first_chunk_frames: usize,
    pub emitted_chunks: usize,
    pub audio_decode_calls: usize,
    pub audio_input_frames: usize,
    pub audio_decoded_window_frames: usize,
    pub audio_max_decoded_window_frames: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RunnerInstrumentation {
    command_buffers_submitted: usize,
    host_waits: usize,
}

#[derive(Debug, Clone, Copy)]
struct AdaptiveChunkController {
    ema_ms_per_frame: Option<f64>,
    current_chunk_frames: usize,
}

pub struct TtsSession {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    model_metadata: ModelMetadata,
    tokenizer: Tokenizer,
    audio: AudioGenerationContext,
    audio_decoder: Box<dyn AudioDecoderBackend>,
    prompt_template: String,
    drop_initial_newline: bool,
    text_decoder: RefCell<Box<dyn SemanticDecoderBackend>>,
    last_execution_stats: RefCell<Option<TtsExecutionStats>>,
}

trait SemanticDecoderBackend {
    fn default_seed(&self) -> u64;

    fn generate_semantic_tokens(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error>;

    fn generate_semantic_tokens_with_callback(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<(), Error>,
    ) -> Result<AudioTokenGrid, Error>;

    fn take_instrumentation(&mut self) -> RunnerInstrumentation;
}

trait AudioDecoderBackend {
    fn codec_cardinality(&self) -> usize;

    fn num_codebooks(&self) -> usize;

    fn sample_rate(&self) -> u32;

    fn decode(
        &self,
        tokens: &AudioTokenGrid,
    ) -> Result<AudioPcmBatch, Error>;

    fn begin_stream(
        &self,
        batch_size: usize,
        codebooks: usize,
        mode: AudioDecodeStreamingMode,
        max_workspace_frames: usize,
    ) -> Result<Box<dyn AudioDecoderStreamBackend>, Error>;
}

trait AudioDecoderStreamBackend {
    fn decode_step(
        &mut self,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> Result<AudioPcmBatch, Error>;

    fn decode_step_pending(
        &mut self,
        new_tokens: &AudioTokenGrid,
        is_final: bool,
    ) -> Result<Box<dyn PendingAudioChunkBackend>, Error> {
        let pcm = self.decode_step(new_tokens, is_final)?;
        Ok(Box::new(ImmediatePendingAudioChunk {
            pcm: Some(pcm),
            step_stats: self.last_step_stats(),
        }))
    }

    fn last_step_stats(&self) -> Option<AudioDecodeStepStats>;

    fn finish(self: Box<Self>) -> Result<(), Error>;
}

trait PendingAudioChunkBackend {
    fn is_ready(&self) -> bool;

    fn step_stats(&self) -> Option<AudioDecodeStepStats>;

    fn resolve(self: Box<Self>) -> Result<AudioPcmBatch, Error>;
}

struct ImmediatePendingAudioChunk {
    pcm: Option<AudioPcmBatch>,
    step_stats: Option<AudioDecodeStepStats>,
}

impl PendingAudioChunkBackend for ImmediatePendingAudioChunk {
    fn is_ready(&self) -> bool {
        true
    }

    fn step_stats(&self) -> Option<AudioDecodeStepStats> {
        self.step_stats
    }

    fn resolve(mut self: Box<Self>) -> Result<AudioPcmBatch, Error> {
        self.pcm.take().ok_or(Error::GenerateFailed)
    }
}

struct NanoCodecPendingAudioChunk {
    inner: Option<crate::audio::nanocodec::runtime::PendingStreamPcmChunk>,
}

impl PendingAudioChunkBackend for NanoCodecPendingAudioChunk {
    fn is_ready(&self) -> bool {
        self.inner.as_ref().is_none_or(|pending| pending.is_ready())
    }

    fn step_stats(&self) -> Option<AudioDecodeStepStats> {
        self.inner.as_ref().map(|pending| pending.step_stats())
    }

    fn resolve(mut self: Box<Self>) -> Result<AudioPcmBatch, Error> {
        let pending = self.inner.take().ok_or(Error::GenerateFailed)?;
        pending.resolve().map_err(Error::from)
    }
}

struct PendingStreamingChunk {
    submitted_at: Instant,
    ready_frames: usize,
    next_chunk_frames: usize,
    chunk: Box<dyn PendingAudioChunkBackend>,
}

#[derive(Clone, Copy)]
struct StubTextDecoderRuntime {
    num_codebooks: usize,
    codebook_size: usize,
    default_seed: u64,
}

type MetalContext = <Metal as Backend>::Context;
type MetalCommandBuffer = <Metal as Backend>::CommandBuffer;

struct TokenDecoderRunner {
    context: Rc<MetalContext>,
    command_buffer: Rc<RefCell<MetalCommandBuffer>>,
    cache_layers: Rc<RefCell<CacheLayers<Metal>>>,
    shared_buffers: Rc<RefCell<SharedBuffers<Metal>>>,
    scratch_buffers: ScratchBuffers<Metal>,
    model_shape: ModelShape,
    decoder_config: Rc<crate::config::DecoderConfig>,
    executables: Decoder<Metal>,
    sampler: GpuSampling<Metal>,
    repetition_penalty: <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel,
    kv_cache_update: KVCacheUpdate<Metal>,
    tensor_copy: <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel,
    tensor_add_scale: <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel,
    token_copy_sampled: <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel,
    token_copy_results: <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel,
    async_chain_positions: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_seeds: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_results: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_repetition_tokens: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_repetition_counts: Rc<RefCell<<Metal as Backend>::NativeBuffer>>,
    async_chain_capacity: usize,
    repetition_capacity: usize,
    repetition_tokens: ArrayCell<Metal>,
    repetition_counts: ArrayCell<Metal>,
    repetition_window_raw: Vec<u32>,
    single_hidden_capture: ArrayCell<Metal>,
    single_override_embedding: ArrayCell<Metal>,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    should_fill_attention_bias: bool,
    next_position: usize,
    instrumentation: RunnerInstrumentation,
}

type PreInjectionEncodeCallback<'a> =
    dyn FnMut(&TokenDecoderRunner, &ForwardPassState<Metal>, &mut MetalCommandBuffer) -> Result<(), Error> + 'a;

#[derive(Debug, Clone)]
struct MatrixF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

enum EmbeddingInjection {
    None,
    AddPreloaded {
        post_scale: Option<f32>,
    },
    OverrideFirstRowInternal,
}

struct TextSamplingState {
    rng: StdRng,
    method: SamplingMethod,
    repetition_penalty: f32,
}

impl TextSamplingState {
    fn from_config(
        seed: u64,
        config: &TextSamplingConfig,
    ) -> Self {
        Self::with_params(seed, config.temperature, config.top_p, config.repetition_penalty)
    }

    fn with_params(
        seed: u64,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Self {
        let method = if temperature <= 0.0 || top_p <= 0.0 {
            SamplingMethod::Greedy
        } else {
            SamplingMethod::Stochastic {
                temperature: Some(temperature),
                top_k: None,
                top_p: Some(top_p),
                min_p: None,
                repetition_penalty: Some(repetition_penalty),
                processing_order: SamplingProcessingOrder::FiltersThenTemperature,
            }
        };
        Self {
            rng: StdRng::seed_from_u64(seed),
            method,
            repetition_penalty,
        }
    }

    fn method(&self) -> SamplingMethod {
        self.method
    }

    fn next_seed(&mut self) -> u64 {
        self.rng.random::<u64>()
    }

    fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    fn uses_repetition_penalty(&self) -> bool {
        matches!(self.method, SamplingMethod::Stochastic { .. }) && (self.repetition_penalty - 1.0).abs() > 1e-6
    }
}

fn text_decoder_prefill_step_size(
    config: &TextDecoderRuntimeConfig,
    context_length: usize,
) -> usize {
    config.prefill_step_size.min(context_length.max(1)).max(1)
}

fn normalize_rendered_prompt(
    mut rendered: String,
    template: &str,
    drop_initial_newline: bool,
) -> String {
    if drop_initial_newline && rendered.starts_with('\n') {
        rendered.remove(0);
    }
    if template.ends_with('\n') && rendered.ends_with('\n') {
        rendered.pop();
    }
    rendered
}

fn push_recent_token(
    history: &mut Vec<u32>,
    token: u32,
    max_window: usize,
) {
    if max_window == 0 {
        return;
    }
    if history.len() == max_window {
        history.remove(0);
    }
    history.push(token);
}

fn write_repetition_window(
    destination: &mut [u32],
    source: &[u32],
) -> usize {
    let mut count = 0usize;
    for &token in source.iter().rev() {
        if destination[..count].contains(&token) {
            continue;
        }
        if count >= destination.len() {
            break;
        }
        destination[count] = token;
        count += 1;
    }
    count
}

fn write_repetition_window_tail(
    destination: &mut [u32],
    source: &[u32],
) -> usize {
    destination.fill(0);
    let tail = tail_repetition_window(source, destination.len());
    write_repetition_window(destination, tail)
}

fn tail_repetition_window(
    source: &[u32],
    max_window: usize,
) -> &[u32] {
    if source.len() <= max_window {
        source
    } else {
        &source[source.len() - max_window..]
    }
}

impl AdaptiveChunkController {
    fn new(config: &TtsRunConfig) -> Self {
        Self {
            ema_ms_per_frame: None,
            current_chunk_frames: config.min_chunk_frames.max(1),
        }
    }

    fn target_frames(
        &self,
        config: &TtsRunConfig,
    ) -> usize {
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        match config.chunk_policy {
            TtsChunkPolicy::Fixed => min_frames,
            TtsChunkPolicy::Adaptive => {
                let Some(ema_ms_per_frame) = self.ema_ms_per_frame else {
                    return min_frames;
                };
                let raw = (config.target_emit_latency_ms as f64 / ema_ms_per_frame).round();
                let candidate = raw.max(min_frames as f64).min(max_frames as f64) as usize;
                if self.current_chunk_frames == 0 {
                    return candidate;
                }
                // With full-prefix fallback vocoder decode, shrinking chunk size causes severe re-decode thrash.
                // Keep chunk size monotonic for the duration of a synthesis run.
                if candidate <= self.current_chunk_frames {
                    return self.current_chunk_frames;
                }
                let change = ((candidate as f64 - self.current_chunk_frames as f64).abs()
                    / self.current_chunk_frames as f64)
                    .max(0.0);
                if change < DEFAULT_CHUNK_HYSTERESIS_FRACTION {
                    self.current_chunk_frames
                } else {
                    candidate
                }
            },
        }
    }

    fn observe(
        &mut self,
        frames: usize,
        decode_elapsed: std::time::Duration,
        next_chunk_frames: usize,
    ) {
        if frames == 0 {
            return;
        }
        let ms_per_frame = (decode_elapsed.as_secs_f64() * 1000.0) / frames as f64;
        self.ema_ms_per_frame = Some(match self.ema_ms_per_frame {
            Some(previous) => previous * (1.0 - DEFAULT_CHUNK_EMA_ALPHA) + ms_per_frame * DEFAULT_CHUNK_EMA_ALPHA,
            None => ms_per_frame,
        });
        self.current_chunk_frames = next_chunk_frames.max(1);
    }

    fn adapt_up_for_realtime(
        &mut self,
        config: &TtsRunConfig,
        generated_frames: usize,
        sample_rate: u32,
        decode_elapsed: std::time::Duration,
        emitted_audio_frames: usize,
    ) {
        if generated_frames == 0 || emitted_audio_frames == 0 || sample_rate == 0 {
            return;
        }
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        let decode_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let produced_audio_ms = (emitted_audio_frames as f64) * 1000.0 / f64::from(sample_rate);
        if produced_audio_ms <= 0.0 {
            return;
        }
        let realtime_ratio = decode_ms / produced_audio_ms;
        if realtime_ratio <= 1.0 {
            return;
        }
        let scaled = ((generated_frames as f64) * realtime_ratio * 1.1).ceil() as usize;
        let clamped = scaled.clamp(min_frames, max_frames);
        self.current_chunk_frames = self.current_chunk_frames.max(clamped);
    }

    fn promote_to_max_chunk(
        &mut self,
        config: &TtsRunConfig,
    ) {
        let min_frames = config.min_chunk_frames.max(1);
        let max_frames = config.max_chunk_frames.max(min_frames);
        self.current_chunk_frames = max_frames;
    }
}

fn next_startup_target_frames(
    current_target_frames: usize,
    startup_cap_frames: usize,
) -> usize {
    let startup_cap_frames = startup_cap_frames.max(1);
    current_target_frames.max(1).saturating_mul(2).min(startup_cap_frames)
}

struct StreamingTokenAccumulator {
    by_codebook: Vec<Vec<u32>>,
}

impl StreamingTokenAccumulator {
    fn new(num_codebooks: usize) -> Result<Self, Error> {
        if num_codebooks == 0 {
            return Err(Error::UnableToLoadConfig);
        }
        Ok(Self {
            by_codebook: vec![Vec::new(); num_codebooks],
        })
    }

    fn push_frame(
        &mut self,
        frame_codes: &[u32],
    ) -> Result<(), Error> {
        if frame_codes.len() != self.by_codebook.len() {
            return Err(Error::GenerateFailed);
        }
        for (codebook, &token) in self.by_codebook.iter_mut().zip(frame_codes.iter()) {
            codebook.push(token);
        }
        Ok(())
    }

    fn frames(&self) -> usize {
        self.by_codebook.first().map_or(0, Vec::len)
    }

    #[cfg(test)]
    fn to_grid(&self) -> Result<AudioTokenGrid, Error> {
        let frames = self.frames();
        let mut tokens = Vec::with_capacity(self.by_codebook.len() * frames);
        for codebook in &self.by_codebook {
            if codebook.len() != frames {
                return Err(Error::GenerateFailed);
            }
            tokens.extend_from_slice(codebook);
        }

        AudioTokenGrid::new(
            tokens.into_boxed_slice(),
            1,
            self.by_codebook.len(),
            frames,
            vec![frames].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .map_err(Error::from)
    }

    fn to_grid_range(
        &self,
        frame_start: usize,
        frame_end: usize,
    ) -> Result<AudioTokenGrid, Error> {
        let frames = self.frames();
        if frame_start > frame_end || frame_end > frames {
            return Err(Error::GenerateFailed);
        }
        let range_frames = frame_end - frame_start;
        let mut tokens = Vec::with_capacity(self.by_codebook.len() * range_frames);
        for codebook in &self.by_codebook {
            if codebook.len() != frames {
                return Err(Error::GenerateFailed);
            }
            tokens.extend_from_slice(&codebook[frame_start..frame_end]);
        }

        AudioTokenGrid::new(
            tokens.into_boxed_slice(),
            1,
            self.by_codebook.len(),
            range_frames,
            vec![range_frames].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .map_err(Error::from)
    }
}

fn slice_codebook_major_grid_range(
    grid: &AudioTokenGrid,
    frame_start: usize,
    frame_end: usize,
) -> Result<AudioTokenGrid, Error> {
    if frame_start > frame_end || frame_end > grid.frames() {
        return Err(Error::GenerateFailed);
    }
    let range_frames = frame_end.saturating_sub(frame_start);
    let packed = grid.to_packing(AudioTokenPacking::CodebookMajor);
    let batch_size = packed.batch_size();
    let codebooks = packed.codebooks();
    let frames = packed.frames();
    if frames != grid.frames() {
        return Err(Error::GenerateFailed);
    }

    let mut tokens = Vec::with_capacity(batch_size * codebooks * range_frames);
    let row_stride = frames;
    for batch in 0..batch_size {
        for codebook in 0..codebooks {
            let row_index = batch
                .checked_mul(codebooks)
                .and_then(|value| value.checked_add(codebook))
                .ok_or(Error::GenerateFailed)?;
            let row_start = row_index.checked_mul(row_stride).ok_or(Error::GenerateFailed)?;
            let src_start = row_start.checked_add(frame_start).ok_or(Error::GenerateFailed)?;
            let src_end = row_start.checked_add(frame_end).ok_or(Error::GenerateFailed)?;
            tokens.extend_from_slice(&packed.tokens()[src_start..src_end]);
        }
    }

    let mut lengths = Vec::with_capacity(batch_size);
    for &length in packed.lengths() {
        lengths.push(length.saturating_sub(frame_start).min(range_frames));
    }

    AudioTokenGrid::new(
        tokens.into_boxed_slice(),
        batch_size,
        codebooks,
        range_frames,
        lengths.into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .map_err(Error::from)
}

fn accumulate_audio_decode_step_stats(
    decode_calls: &mut usize,
    input_frames: &mut usize,
    decoded_window_frames: &mut usize,
    max_decoded_window_frames: &mut usize,
    step: Option<AudioDecodeStepStats>,
) {
    let Some(step) = step else {
        return;
    };
    if step.input_frames == 0 && step.decoded_window_frames == 0 {
        return;
    }
    *decode_calls = decode_calls.saturating_add(1);
    *input_frames = input_frames.saturating_add(step.input_frames);
    *decoded_window_frames = decoded_window_frames.saturating_add(step.decoded_window_frames);
    *max_decoded_window_frames = (*max_decoded_window_frames).max(step.decoded_window_frames);
}

fn maybe_flush_pending_stream_chunk<F>(
    pending_chunk: &mut Option<PendingStreamingChunk>,
    force: bool,
    count_in_loop: bool,
    on_chunk: &mut F,
    callback_seconds: &mut f64,
    audio_decode_seconds_in_loop: &mut f64,
    audio_decode_seconds: &mut f64,
    output_samples: &mut Vec<f32>,
    output_frames: &mut usize,
    output_sample_rate: &mut u32,
    output_channels: &mut usize,
    emitted_chunks: &mut usize,
    first_emit_pending: &mut bool,
    first_chunk_seconds: &mut Option<f64>,
    first_chunk_frames: &mut usize,
    stream_start: Instant,
    startup_target_frames: &mut usize,
    startup_cap_frames: usize,
    chunk_controller: &mut AdaptiveChunkController,
    config: &TtsRunConfig,
    audio_decode_calls: &mut usize,
    audio_input_frames: &mut usize,
    audio_decoded_window_frames: &mut usize,
    audio_max_decoded_window_frames: &mut usize,
) -> Result<(), Error>
where
    F: FnMut(&AudioPcmBatch),
{
    let should_flush = pending_chunk.as_ref().map(|pending| force || pending.chunk.is_ready()).unwrap_or(false);
    if !should_flush {
        return Ok(());
    }

    let pending = pending_chunk.take().ok_or(Error::GenerateFailed)?;
    accumulate_audio_decode_step_stats(
        audio_decode_calls,
        audio_input_frames,
        audio_decoded_window_frames,
        audio_max_decoded_window_frames,
        pending.chunk.step_stats(),
    );
    let partial_pcm = pending.chunk.resolve()?;
    let decode_elapsed = pending.submitted_at.elapsed();
    if count_in_loop {
        *audio_decode_seconds_in_loop += decode_elapsed.as_secs_f64();
    }
    *audio_decode_seconds += decode_elapsed.as_secs_f64();

    let partial_sample_rate = partial_pcm.sample_rate();
    let callback_start = Instant::now();
    let emitted_frames = if partial_pcm.lengths().len() == 1 {
        partial_pcm.lengths()[0]
    } else {
        return Err(Error::GenerateFailed);
    };
    if emitted_frames > 0 {
        on_chunk(&partial_pcm);
    }
    *callback_seconds += callback_start.elapsed().as_secs_f64();
    chunk_controller.observe(pending.ready_frames, decode_elapsed, pending.next_chunk_frames);

    if emitted_frames > 0 {
        chunk_controller.adapt_up_for_realtime(
            config,
            pending.ready_frames,
            partial_sample_rate,
            decode_elapsed,
            emitted_frames,
        );
        output_samples.extend_from_slice(partial_pcm.samples());
        *output_frames = (*output_frames).saturating_add(emitted_frames);
        *output_sample_rate = partial_pcm.sample_rate();
        *output_channels = partial_pcm.channels();
        *emitted_chunks = (*emitted_chunks).saturating_add(1);
        if *first_emit_pending {
            *first_emit_pending = false;
            *first_chunk_seconds = Some(stream_start.elapsed().as_secs_f64());
            *first_chunk_frames = emitted_frames;
            chunk_controller.promote_to_max_chunk(config);
        }
    } else if *first_emit_pending {
        *startup_target_frames = next_startup_target_frames(*startup_target_frames, startup_cap_frames);
    }

    Ok(())
}

impl SemanticDecoderBackend for StubTextDecoderRuntime {
    fn default_seed(&self) -> u64 {
        self.default_seed
    }

    fn generate_semantic_tokens(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        generate_stub_semantic_grid(self, text_tokens, codec_cardinality, seed, max_semantic_frames)
    }

    fn generate_semantic_tokens_with_callback(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<(), Error>,
    ) -> Result<AudioTokenGrid, Error> {
        let grid = generate_stub_semantic_grid(self, text_tokens, codec_cardinality, seed, max_semantic_frames)?;
        for frame in 0..grid.frames() {
            let mut frame_codes = Vec::with_capacity(self.num_codebooks);
            for codebook in 0..self.num_codebooks {
                frame_codes.push(grid.get(0, codebook, frame));
            }
            on_frame(&frame_codes)?;
        }
        Ok(grid)
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        RunnerInstrumentation::default()
    }
}

impl TtsSession {
    pub fn new(model_path: PathBuf) -> Result<Self, Error> {
        Self::new_with_options(model_path, TtsSessionOptions::default())
    }

    pub fn new_with_options(
        model_path: PathBuf,
        options: TtsSessionOptions,
    ) -> Result<Self, Error> {
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::UnableToLoadConfig);
        }
        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(std::io::BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        Self::from_model_metadata_with_options(model_path, model_metadata, options)
    }

    pub fn last_execution_stats(&self) -> Option<TtsExecutionStats> {
        self.last_execution_stats.borrow().clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    pub fn synthesize(
        &self,
        input: Input,
    ) -> Result<AudioPcmBatch, Error> {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_with_seed(input, seed)
    }

    pub fn synthesize_with_seed(
        &self,
        input: Input,
        seed: u64,
    ) -> Result<AudioPcmBatch, Error> {
        self.synthesize_with_seed_and_config(input, seed, &TtsRunConfig::default())
    }

    pub fn synthesize_with_config(
        &self,
        input: Input,
        config: &TtsRunConfig,
    ) -> Result<AudioPcmBatch, Error> {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_with_seed_and_config(input, seed, config)
    }

    pub fn synthesize_with_seed_and_config(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
    ) -> Result<AudioPcmBatch, Error> {
        config.validate().map_err(|_| Error::GenerateFailed)?;

        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        let semantic_start = Instant::now();
        let semantic_tokens = self.generate_semantic_tokens(&text_tokens, seed, config.max_semantic_frames)?;
        let semantic_decode_seconds = semantic_start.elapsed().as_secs_f64();
        let instrumentation = self.take_text_decoder_instrumentation();

        let audio_start = Instant::now();
        let mut audio_decode_calls = 0usize;
        let mut audio_input_frames = 0usize;
        let mut audio_decoded_window_frames = 0usize;
        let mut audio_max_decoded_window_frames = 0usize;
        let pcm = match config.non_streaming_mode {
            crate::session::config::TtsNonStreamingMode::FullDecode => {
                audio_decode_calls = usize::from(semantic_tokens.frames() > 0);
                audio_input_frames = semantic_tokens.frames();
                audio_decoded_window_frames = semantic_tokens.frames();
                audio_max_decoded_window_frames = semantic_tokens.frames();
                self.audio_decoder.decode(&semantic_tokens)?
            },
            crate::session::config::TtsNonStreamingMode::ChunkedIfNeeded => {
                let total_frames = semantic_tokens.frames();
                let chunked_threshold = config.max_stream_workspace_frames.max(config.max_chunk_frames.max(1));
                if total_frames < chunked_threshold {
                    audio_decode_calls = usize::from(total_frames > 0);
                    audio_input_frames = total_frames;
                    audio_decoded_window_frames = total_frames;
                    audio_max_decoded_window_frames = total_frames;
                    self.audio_decoder.decode(&semantic_tokens)?
                } else {
                    let chunk_frames = config.max_chunk_frames.max(config.min_chunk_frames.max(1));
                    let workspace_frames = config.max_stream_workspace_frames.max(chunk_frames);
                    let mut stream = self.audio_decoder.begin_stream(
                        semantic_tokens.batch_size(),
                        semantic_tokens.codebooks(),
                        AudioDecodeStreamingMode::IncrementalStateful,
                        workspace_frames,
                    )?;

                    let mut all_samples = Vec::<f32>::new();
                    let mut accumulated_lengths = vec![0usize; semantic_tokens.batch_size()];
                    let mut sample_rate = self.audio_decoder.sample_rate();
                    let mut channels = 1usize;

                    let mut frame_start = 0usize;
                    while frame_start < total_frames {
                        let frame_end = (frame_start + chunk_frames).min(total_frames);
                        let delta_grid = slice_codebook_major_grid_range(&semantic_tokens, frame_start, frame_end)?;
                        let partial_pcm = stream.decode_step(&delta_grid, frame_end == total_frames)?;
                        accumulate_audio_decode_step_stats(
                            &mut audio_decode_calls,
                            &mut audio_input_frames,
                            &mut audio_decoded_window_frames,
                            &mut audio_max_decoded_window_frames,
                            stream.last_step_stats(),
                        );
                        if partial_pcm.lengths().len() != accumulated_lengths.len() {
                            return Err(Error::GenerateFailed);
                        }
                        for (acc, &len) in accumulated_lengths.iter_mut().zip(partial_pcm.lengths().iter()) {
                            *acc = acc.saturating_add(len);
                        }
                        if !partial_pcm.samples().is_empty() {
                            all_samples.extend_from_slice(partial_pcm.samples());
                        }
                        sample_rate = partial_pcm.sample_rate();
                        channels = partial_pcm.channels();
                        frame_start = frame_end;
                    }
                    stream.finish()?;
                    AudioPcmBatch::new(
                        all_samples.into_boxed_slice(),
                        sample_rate,
                        channels,
                        accumulated_lengths.into_boxed_slice(),
                    )
                    .map_err(Error::from)?
                }
            },
        };
        let audio_decode_seconds = audio_start.elapsed().as_secs_f64();

        self.record_last_execution_stats(TtsExecutionStats {
            semantic_decode_seconds,
            audio_decode_seconds,
            callback_seconds: 0.0,
            first_chunk_seconds: 0.0,
            command_buffers_submitted: instrumentation.command_buffers_submitted,
            host_waits: instrumentation.host_waits,
            semantic_frames: semantic_tokens.frames(),
            first_chunk_frames: 0,
            emitted_chunks: usize::from(config.streaming_enabled),
            audio_decode_calls,
            audio_input_frames,
            audio_decoded_window_frames,
            audio_max_decoded_window_frames,
        });

        Ok(pcm)
    }

    pub fn generate_semantic_tokens_with_seed(
        &self,
        input: Input,
        seed: u64,
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_with_seed_and_config(input, seed, &TtsRunConfig::default())
    }

    pub fn generate_semantic_tokens_with_seed_and_config(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
    ) -> Result<AudioTokenGrid, Error> {
        config.validate().map_err(|_| Error::GenerateFailed)?;
        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        self.generate_semantic_tokens(&text_tokens, seed, config.max_semantic_frames)
    }

    pub fn synthesize_streaming<F>(
        &self,
        input: Input,
        chunk_frames: usize,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_streaming_with_seed(input, seed, chunk_frames, on_chunk)
    }

    pub fn synthesize_streaming_with_seed<F>(
        &self,
        input: Input,
        seed: u64,
        chunk_frames: usize,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let config = TtsRunConfig::fixed_chunk_frames(chunk_frames);
        self.synthesize_streaming_with_seed_and_config(input, seed, &config, on_chunk)
    }

    pub fn synthesize_streaming_with_config<F>(
        &self,
        input: Input,
        config: &TtsRunConfig,
        on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        let seed = self.text_decoder.borrow().default_seed();
        self.synthesize_streaming_with_seed_and_config(input, seed, config, on_chunk)
    }

    pub fn synthesize_streaming_with_seed_and_config<F>(
        &self,
        input: Input,
        seed: u64,
        config: &TtsRunConfig,
        mut on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        config.validate().map_err(|_| Error::GenerateFailed)?;
        if !config.streaming_enabled {
            let pcm = self.synthesize_with_seed_and_config(input, seed, config)?;
            on_chunk(&pcm);
            return Ok(pcm);
        }

        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        let mut streamed_tokens = StreamingTokenAccumulator::new(self.audio_decoder.num_codebooks())?;
        let streaming_mode = match config.vocoder_streaming_mode {
            TtsVocoderStreamingMode::IncrementalStateful => AudioDecodeStreamingMode::IncrementalStateful,
            TtsVocoderStreamingMode::PrefixFallback => AudioDecodeStreamingMode::PrefixFallback,
        };
        let mut audio_stream = self.audio_decoder.begin_stream(
            1,
            self.audio_decoder.num_codebooks(),
            streaming_mode,
            config.max_stream_workspace_frames,
        )?;
        let mut emitted_chunks = 0usize;
        let mut callback_seconds = 0.0_f64;
        let mut audio_decode_seconds_in_loop = 0.0_f64;
        let mut audio_decode_seconds = 0.0_f64;
        let mut last_decoded_frames = 0usize;
        let mut first_chunk_seconds = None::<f64>;
        let mut first_chunk_frames = 0usize;
        let mut first_emit_pending = true;
        let mut output_samples = Vec::<f32>::new();
        let mut output_frames = 0usize;
        let mut output_sample_rate = self.audio_decoder.sample_rate();
        let mut output_channels = 1usize;
        let mut chunk_controller = AdaptiveChunkController::new(config);
        let mut pending_chunk = None::<PendingStreamingChunk>;
        let mut audio_decode_calls = 0usize;
        let mut audio_input_frames = 0usize;
        let mut audio_decoded_window_frames = 0usize;
        let mut audio_max_decoded_window_frames = 0usize;
        let stream_start = Instant::now();
        let startup_cap_frames = config.max_chunk_frames.max(config.min_chunk_frames.max(1));
        let initial_chunk_frames = config.initial_chunk_frames.max(1).min(startup_cap_frames);
        let mut startup_target_frames = initial_chunk_frames;

        let semantic_start = Instant::now();
        let semantic_tokens = self.generate_semantic_tokens_with_callback(
            &text_tokens,
            seed,
            config.max_semantic_frames,
            &mut |codes| {
                maybe_flush_pending_stream_chunk(
                    &mut pending_chunk,
                    false,
                    true,
                    &mut on_chunk,
                    &mut callback_seconds,
                    &mut audio_decode_seconds_in_loop,
                    &mut audio_decode_seconds,
                    &mut output_samples,
                    &mut output_frames,
                    &mut output_sample_rate,
                    &mut output_channels,
                    &mut emitted_chunks,
                    &mut first_emit_pending,
                    &mut first_chunk_seconds,
                    &mut first_chunk_frames,
                    stream_start,
                    &mut startup_target_frames,
                    startup_cap_frames,
                    &mut chunk_controller,
                    config,
                    &mut audio_decode_calls,
                    &mut audio_input_frames,
                    &mut audio_decoded_window_frames,
                    &mut audio_max_decoded_window_frames,
                )?;
                streamed_tokens.push_frame(codes)?;
                let ready_frames = streamed_tokens.frames().saturating_sub(last_decoded_frames);
                let adaptive_target_frames = chunk_controller.target_frames(config);
                let target_frames = if first_emit_pending {
                    startup_target_frames
                } else {
                    adaptive_target_frames
                };
                if ready_frames >= target_frames && pending_chunk.is_none() {
                    let decode_start = Instant::now();
                    let partial_grid = streamed_tokens.to_grid_range(last_decoded_frames, streamed_tokens.frames())?;
                    last_decoded_frames = streamed_tokens.frames();
                    let pending_next_chunk_frames = if first_emit_pending {
                        adaptive_target_frames
                    } else {
                        target_frames
                    };
                    pending_chunk = Some(PendingStreamingChunk {
                        submitted_at: decode_start,
                        ready_frames,
                        next_chunk_frames: pending_next_chunk_frames,
                        chunk: audio_stream.decode_step_pending(&partial_grid, false)?,
                    });
                    maybe_flush_pending_stream_chunk(
                        &mut pending_chunk,
                        false,
                        true,
                        &mut on_chunk,
                        &mut callback_seconds,
                        &mut audio_decode_seconds_in_loop,
                        &mut audio_decode_seconds,
                        &mut output_samples,
                        &mut output_frames,
                        &mut output_sample_rate,
                        &mut output_channels,
                        &mut emitted_chunks,
                        &mut first_emit_pending,
                        &mut first_chunk_seconds,
                        &mut first_chunk_frames,
                        stream_start,
                        &mut startup_target_frames,
                        startup_cap_frames,
                        &mut chunk_controller,
                        config,
                        &mut audio_decode_calls,
                        &mut audio_input_frames,
                        &mut audio_decoded_window_frames,
                        &mut audio_max_decoded_window_frames,
                    )?;
                }
                Ok(())
            },
        )?;
        let semantic_loop_seconds = semantic_start.elapsed().as_secs_f64();

        maybe_flush_pending_stream_chunk(
            &mut pending_chunk,
            true,
            false,
            &mut on_chunk,
            &mut callback_seconds,
            &mut audio_decode_seconds_in_loop,
            &mut audio_decode_seconds,
            &mut output_samples,
            &mut output_frames,
            &mut output_sample_rate,
            &mut output_channels,
            &mut emitted_chunks,
            &mut first_emit_pending,
            &mut first_chunk_seconds,
            &mut first_chunk_frames,
            stream_start,
            &mut startup_target_frames,
            startup_cap_frames,
            &mut chunk_controller,
            config,
            &mut audio_decode_calls,
            &mut audio_input_frames,
            &mut audio_decoded_window_frames,
            &mut audio_max_decoded_window_frames,
        )?;

        if last_decoded_frames < semantic_tokens.frames() {
            let final_decode_start = Instant::now();
            let final_delta_grid = streamed_tokens.to_grid_range(last_decoded_frames, semantic_tokens.frames())?;
            let final_pcm = audio_stream.decode_step(&final_delta_grid, true)?;
            accumulate_audio_decode_step_stats(
                &mut audio_decode_calls,
                &mut audio_input_frames,
                &mut audio_decoded_window_frames,
                &mut audio_max_decoded_window_frames,
                audio_stream.last_step_stats(),
            );
            audio_decode_seconds += final_decode_start.elapsed().as_secs_f64();
            let callback_start = Instant::now();
            let emitted_frames = if final_pcm.lengths().len() == 1 {
                final_pcm.lengths()[0]
            } else {
                return Err(Error::GenerateFailed);
            };
            if emitted_frames > 0 {
                on_chunk(&final_pcm);
            }
            callback_seconds += callback_start.elapsed().as_secs_f64();
            if emitted_frames > 0 {
                output_samples.extend_from_slice(final_pcm.samples());
                output_frames = output_frames.saturating_add(emitted_frames);
                output_sample_rate = final_pcm.sample_rate();
                output_channels = final_pcm.channels();
                emitted_chunks += 1;
                if first_emit_pending {
                    first_chunk_seconds = Some(stream_start.elapsed().as_secs_f64());
                    first_chunk_frames = emitted_frames;
                }
            }
        }
        let semantic_decode_seconds =
            (semantic_loop_seconds - audio_decode_seconds_in_loop - callback_seconds).max(0.0);
        audio_stream.finish()?;

        let full_pcm = AudioPcmBatch::new(
            output_samples.into_boxed_slice(),
            output_sample_rate,
            output_channels,
            vec![output_frames].into_boxed_slice(),
        )
        .map_err(Error::from)?;

        let instrumentation = self.take_text_decoder_instrumentation();
        self.record_last_execution_stats(TtsExecutionStats {
            semantic_decode_seconds,
            audio_decode_seconds,
            callback_seconds,
            first_chunk_seconds: first_chunk_seconds.unwrap_or(0.0),
            command_buffers_submitted: instrumentation.command_buffers_submitted,
            host_waits: instrumentation.host_waits,
            semantic_frames: semantic_tokens.frames(),
            first_chunk_frames,
            emitted_chunks,
            audio_decode_calls,
            audio_input_frames,
            audio_decoded_window_frames,
            audio_max_decoded_window_frames,
        });

        Ok(full_pcm)
    }

    fn from_model_metadata_with_options(
        model_path: PathBuf,
        model_metadata: ModelMetadata,
        options: TtsSessionOptions,
    ) -> Result<Self, Error> {
        if model_metadata.model_type != ModelType::TtsModel {
            return Err(Error::UnableToLoadConfig);
        }

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| Error::UnableToLoadTokenizer)?;

        let tts_model_config = model_metadata.model_config.as_tts().ok_or(Error::UnableToLoadConfig)?.clone();
        let audio =
            tts_model_config.create_audio_generation_context_with_model_path_and_options(&model_path, options.audio_runtime)?;
        let text_decoder = build_text_decoder_backend(&tts_model_config, &audio, &model_path, &options)?;
        let audio_decoder = build_audio_decoder_backend(&audio)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            audio,
            audio_decoder,
            prompt_template: tts_model_config.message_processor_config.prompt_template.clone(),
            drop_initial_newline: tts_model_config.message_processor_config.drop_initial_newline,
            text_decoder: RefCell::new(text_decoder),
            last_execution_stats: RefCell::new(None),
        })
    }

    fn render_prompt(
        &self,
        input: &Input,
    ) -> Result<String, Error> {
        let messages = input
            .get_messages()
            .into_iter()
            .map(|message| {
                HashMap::from([
                    (String::from("content"), message.content),
                    (String::from("speaker_id"), String::from(DEFAULT_STUB_SPEAKER_ID)),
                    (String::from("style"), String::from(DEFAULT_STUB_STYLE)),
                ])
            })
            .collect::<Vec<_>>();

        let template_name = "tts_prompt_template";
        let mut environment = Environment::new();
        environment
            .add_template(template_name, self.prompt_template.as_str())
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;
        let template = environment.get_template(template_name).map_err(|_| Error::UnableToLoadPromptTemplate)?;

        let result = template
            .render(context!(
                messages => messages
            ))
            .map_err(|_| Error::UnableToRenderPromptTemplate)?;

        Ok(normalize_rendered_prompt(result, self.prompt_template.as_str(), self.drop_initial_newline))
    }

    fn generate_semantic_tokens(
        &self,
        text_tokens: &[u64],
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality = self.audio_decoder.codec_cardinality();
        decoder.generate_semantic_tokens(text_tokens, codec_cardinality, seed, max_semantic_frames)
    }

    fn generate_semantic_tokens_with_callback<F>(
        &self,
        text_tokens: &[u64],
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut F,
    ) -> Result<AudioTokenGrid, Error>
    where
        F: FnMut(&[u32]) -> Result<(), Error>,
    {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality = self.audio_decoder.codec_cardinality();
        decoder.generate_semantic_tokens_with_callback(
            text_tokens,
            codec_cardinality,
            seed,
            max_semantic_frames,
            on_frame,
        )
    }

    fn take_text_decoder_instrumentation(&self) -> RunnerInstrumentation {
        self.text_decoder.borrow_mut().take_instrumentation()
    }

    fn record_last_execution_stats(
        &self,
        stats: TtsExecutionStats,
    ) {
        self.last_execution_stats.borrow_mut().replace(stats);
    }
}

fn semantic_token_to_code(
    semantic_token: u64,
    semantic_begin: i64,
    semantic_end: i64,
    token_upper_bound: usize,
) -> u32 {
    if semantic_begin > semantic_end || token_upper_bound == 0 {
        return 0;
    }

    let semantic = semantic_token as i64;
    if semantic < semantic_begin || semantic > semantic_end {
        return 0;
    }

    let relative = usize::try_from(semantic - semantic_begin).unwrap_or(0);
    let clamped = relative.min(token_upper_bound.saturating_sub(1));
    u32::try_from(clamped).unwrap_or(0)
}

fn build_semantic_sampling_mask_row(
    vocab_size: usize,
    semantic_begin: i64,
    semantic_end: i64,
    im_end: i64,
) -> Result<Box<[u32]>, Error> {
    if vocab_size == 0 || semantic_begin > semantic_end {
        return Err(Error::UnableToLoadConfig);
    }

    let max_token_id = i64::try_from(vocab_size.saturating_sub(1)).map_err(|_| Error::UnableToLoadConfig)?;
    if semantic_begin < 0 || semantic_end < 0 || semantic_end > max_token_id || im_end < 0 || im_end > max_token_id {
        return Err(Error::UnableToLoadConfig);
    }

    let row_words = vocab_size.div_ceil(32);
    let mut mask = vec![0_u32; row_words];
    for token_index in semantic_begin..=semantic_end {
        let token = usize::try_from(token_index).map_err(|_| Error::UnableToLoadConfig)?;
        let word = token / 32;
        let bit = token % 32;
        mask[word] |= 1_u32 << bit;
    }
    let im_end_token = usize::try_from(im_end).map_err(|_| Error::UnableToLoadConfig)?;
    let word = im_end_token / 32;
    let bit = im_end_token % 32;
    mask[word] |= 1_u32 << bit;
    Ok(mask.into_boxed_slice())
}

fn clear_token_in_sampling_mask(
    mask: &mut [u32],
    token: i64,
) -> Result<(), Error> {
    if token < 0 {
        return Err(Error::UnableToLoadConfig);
    }
    let token = usize::try_from(token).map_err(|_| Error::UnableToLoadConfig)?;
    let word = token / 32;
    let bit = token % 32;
    if word >= mask.len() {
        return Err(Error::UnableToLoadConfig);
    }
    mask[word] &= !(1_u32 << bit);
    Ok(())
}

fn expand_token_mask_for_sampling_row(
    row_mask: &[u32],
    token_count: usize,
) -> Result<Box<[u32]>, Error> {
    if token_count == 0 || row_mask.is_empty() {
        return Err(Error::GenerateFailed);
    }
    if token_count == 1 {
        return Ok(row_mask.to_vec().into_boxed_slice());
    }

    let row_words = row_mask.len();
    let total_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
    let mut expanded = vec![u32::MAX; total_words];
    let offset = (token_count - 1).checked_mul(row_words).ok_or(Error::GenerateFailed)?;
    expanded[offset..offset + row_words].copy_from_slice(row_mask);
    Ok(expanded.into_boxed_slice())
}

impl TokenDecoderRunner {
    fn new_with_context(
        context: Rc<MetalContext>,
        model_path: &Path,
        decoder_config: Rc<crate::config::DecoderConfig>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
        runtime_config: &TextDecoderRuntimeConfig,
        repetition_window_size: usize,
    ) -> Result<Self, Error> {
        let command_buffer =
            Rc::new(RefCell::new(context.create_command_buffer().expect("Failed to create command buffer")));

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = text_decoder_prefill_step_size(runtime_config, decoder_config.context_length).max(32);
        let should_fill_attention_bias =
            model_shape.sliding_window_length_per_layer.iter().any(|value| value.is_some());
        let activation_data_type = model_shape.activation_data_type();

        let weights_path = model_path.join("model.safetensors");
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(context.as_ref(), &decoder_config, &model_shape)));
        shared_buffers.borrow_mut().update_data_with_transformer_subtree(&root_loader_view, transformer_subtree);

        let scratch_buffers =
            ScratchBuffers::new(context.as_ref(), &decoder_config, &model_shape, max_prefix_length, max_suffix_length);
        let executables = Decoder::new_with_subtrees(
            context.clone(),
            decoder_config.clone(),
            &root_loader_view,
            transformer_subtree,
            embedding_subtree,
            readout_subtree,
        );
        let logits_data_type = scratch_buffers.logits.borrow().data_type();
        let sampler =
            GpuSampling::new(context.as_ref(), logits_data_type, max_suffix_length, decoder_config.vocab_size)
                .map_err(unable_to_create_context)?;
        let repetition_penalty =
            <<Metal as Backend>::Kernels as Kernels>::RepetitionPenaltyKernel::new(context.as_ref(), logits_data_type)
                .map_err(unable_to_create_context)?;
        let tensor_copy =
            <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let tensor_add_scale =
            <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(unable_to_create_context)?;
        let token_copy_sampled =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;
        let token_copy_results =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel::new(context.as_ref())
                .map_err(unable_to_create_context)?;
        let async_chain_capacity = max_suffix_length.max(1);
        let async_chain_positions = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<i32>())
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_seeds = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u64>())
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_results = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(unable_to_create_context)?,
        ));
        let repetition_capacity = repetition_window_size.max(1);
        let async_chain_repetition_tokens = Rc::new(RefCell::new(
            context
                .create_buffer(
                    async_chain_capacity
                        .checked_mul(repetition_capacity)
                        .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                        .ok_or_else(|| unable_to_create_context(std::io::Error::other("async chain repetition buffer size overflow")))?,
                )
                .map_err(unable_to_create_context)?,
        ));
        let async_chain_repetition_counts = Rc::new(RefCell::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(unable_to_create_context)?,
        ));
        let repetition_tokens =
            RefCell::new(context.create_array(&[repetition_capacity], DataType::U32, "tts_repetition_tokens"));
        let repetition_counts = RefCell::new(context.create_array(&[1], DataType::U32, "tts_repetition_counts"));
        let single_hidden_capture = RefCell::new(context.create_array(
            &[1, decoder_config.model_dim],
            activation_data_type,
            "tts_single_hidden_capture",
        ));
        let single_override_embedding = RefCell::new(context.create_array(
            &[1, decoder_config.model_dim],
            activation_data_type,
            "tts_single_override_embedding",
        ));

        let cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.as_ref(),
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));

        let intermediate_data_type: DataType = decoder_config.output_norm_config.scale_precision.into();
        let kv_cache_update = KVCacheUpdate::new(context.as_ref(), intermediate_data_type, max_prefix_length)
            .map_err(unable_to_create_context)?;

        Ok(Self {
            context,
            command_buffer,
            cache_layers,
            shared_buffers,
            scratch_buffers,
            model_shape,
            decoder_config,
            executables,
            sampler,
            repetition_penalty,
            kv_cache_update,
            tensor_copy,
            tensor_add_scale,
            token_copy_sampled,
            token_copy_results,
            async_chain_positions,
            async_chain_seeds,
            async_chain_results,
            async_chain_repetition_tokens,
            async_chain_repetition_counts,
            async_chain_capacity,
            repetition_capacity,
            repetition_tokens,
            repetition_counts,
            repetition_window_raw: Vec::with_capacity(repetition_capacity),
            single_hidden_capture,
            single_override_embedding,
            single_token_vocab_masks: HashMap::new(),
            two_token_vocab_masks: HashMap::new(),
            should_fill_attention_bias,
            next_position: 0,
            instrumentation: RunnerInstrumentation::default(),
        })
    }

    fn reset(&mut self) {
        self.cache_layers.borrow_mut().clear();
        self.next_position = 0;
        self.clear_repetition_window();
    }

    fn clear_repetition_window(&mut self) {
        self.repetition_window_raw.clear();
        self.repetition_counts.borrow_mut().as_slice_mut::<u32>()[0] = 0;
    }

    fn set_repetition_window(
        &mut self,
        previous_tokens: &[u32],
    ) -> Result<(), Error> {
        let mut tokens = self.repetition_tokens.borrow_mut();
        let raw_window = tail_repetition_window(previous_tokens, tokens.shape()[0]);
        self.repetition_window_raw.clear();
        self.repetition_window_raw.extend_from_slice(raw_window);
        let count = write_repetition_window_tail(tokens.as_slice_mut::<u32>(), previous_tokens);
        self.repetition_counts.borrow_mut().as_slice_mut::<u32>()[0] =
            u32::try_from(count).map_err(|_| Error::GenerateFailed)?;
        Ok(())
    }

    fn encode_repetition_penalty_if_needed_with_buffers<'tokens, 'counts>(
        &self,
        sampling: &TextSamplingState,
        count: u32,
        previous_tokens: impl BufferArg<'tokens, <Metal as Backend>::NativeBuffer>,
        previous_counts: impl BufferArg<'counts, <Metal as Backend>::NativeBuffer>,
        batch_size: u32,
        max_previous_tokens: u32,
    ) -> Result<(), Error> {
        if !sampling.uses_repetition_penalty() {
            return Ok(());
        }
        if count == 0 {
            return Ok(());
        }
        let logits = self.scratch_buffers.logits.borrow();
        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            self.repetition_penalty.encode(
                logits.buffer(),
                previous_tokens,
                previous_counts,
                batch_size,
                self.decoder_config.vocab_size as u32,
                max_previous_tokens,
                sampling.repetition_penalty(),
                encoder,
            );
        });
        Ok(())
    }

    fn encode_repetition_penalty_if_needed(
        &self,
        sampling: &TextSamplingState,
    ) -> Result<(), Error> {
        let repetition_tokens = self.repetition_tokens.borrow();
        let repetition_counts = self.repetition_counts.borrow();
        let count = repetition_counts.as_slice::<u32>()[0];
        self.encode_repetition_penalty_if_needed_with_buffers(
            sampling,
            count,
            repetition_tokens.buffer(),
            repetition_counts.buffer(),
            1_u32,
            repetition_tokens.shape()[0] as u32,
        )
    }

    fn populate_async_chain_repetition_windows(
        &mut self,
        repetition_windows: Option<&[Vec<u32>]>,
        first_codebook_index: usize,
        followup_count: usize,
    ) -> Result<(), Error> {
        let tokens_ptr = self.async_chain_repetition_tokens.borrow().cpu_ptr().as_ptr() as *mut u32;
        let counts_ptr = self.async_chain_repetition_counts.borrow().cpu_ptr().as_ptr() as *mut u32;
        for pass in 0..followup_count {
            let row = first_codebook_index + pass;
            let previous_tokens =
                repetition_windows.and_then(|windows| windows.get(row).map(Vec::as_slice)).unwrap_or(&[]);
            let slot_offset = pass.checked_mul(self.repetition_capacity).ok_or(Error::GenerateFailed)?;
            let slot = unsafe { std::slice::from_raw_parts_mut(tokens_ptr.add(slot_offset), self.repetition_capacity) };
            let count = write_repetition_window_tail(slot, previous_tokens);
            unsafe {
                *counts_ptr.add(pass) = u32::try_from(count).map_err(|_| Error::GenerateFailed)?;
            }
        }
        Ok(())
    }

    fn prefill_without_sampling(
        &mut self,
        token_ids: &[u64],
    ) -> Result<(), Error> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut sampling = TextSamplingState::with_params(0, 0.0, 1.0, 1.0);
        let _ = self.decode_next_step(token_ids, EmbeddingInjection::None, None, &mut sampling, None, false, None)?;
        Ok(())
    }

    fn decode_next_token_with_hidden_capture(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
    ) -> Result<u64, Error> {
        self.decode_next_step(token_ids, embedding_injection, None, sampling, precomputed_token_bitmask, true, None)
    }

    fn decode_next_token_with_hidden_capture_and_pre_injection(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_>>,
    ) -> Result<u64, Error> {
        self.decode_next_step(
            token_ids,
            embedding_injection,
            None,
            sampling,
            precomputed_token_bitmask,
            true,
            pre_injection_encode,
        )
    }

    fn decode_next_token(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
    ) -> Result<u64, Error> {
        self.decode_next_step(token_ids, embedding_injection, vocab_limit, sampling, None, false, None)
    }

    fn decode_followup_tokens_batched(
        &mut self,
        first_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        first_codebook_index: usize,
        repetition_windows: Option<&[Vec<u32>]>,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if followup_count == 0 {
            return Ok(());
        }

        if followup_count > self.async_chain_capacity {
            return Err(Error::GenerateFailed);
        }

        let vocab_mask_limit = if let Some(limit_raw) = vocab_limit {
            let limit = limit_raw.min(self.decoder_config.vocab_size);
            if limit == 0 || limit >= self.decoder_config.vocab_size {
                None
            } else {
                self.prepare_two_token_vocab_mask(limit)?;
                self.prepare_single_token_vocab_mask(limit)?;
                Some(limit)
            }
        } else {
            None
        };

        self.populate_async_chain_repetition_windows(repetition_windows, first_codebook_index, followup_count)?;

        {
            let positions_ptr = self.async_chain_positions.borrow().cpu_ptr().as_ptr() as *mut i32;
            for pass in 0..followup_count {
                unsafe {
                    *positions_ptr.add(pass) = (self.next_position + pass) as i32;
                }
            }
        }

        {
            let seeds_ptr = self.async_chain_seeds.borrow().cpu_ptr().as_ptr() as *mut u64;
            if matches!(sampling.method(), SamplingMethod::Stochastic { .. }) {
                for pass in 0..followup_count {
                    unsafe {
                        *seeds_ptr.add(pass) = sampling.next_seed();
                    }
                }
            } else {
                for pass in 0..followup_count {
                    unsafe {
                        *seeds_ptr.add(pass) = 0;
                    }
                }
            }
        }

        self.command_buffer =
            Rc::new(RefCell::new(self.context.create_command_buffer().expect("Failed to create command buffer")));
        for pass in 0..followup_count {
            let token_ids = [if pass == 0 {
                first_token
            } else {
                0
            }];
            let token_bitmask = vocab_mask_limit.and_then(|limit| self.get_single_token_vocab_mask(limit));
            let mut state = ForwardPassState::new_llm(
                self.context.clone(),
                &self.decoder_config,
                &self.model_shape,
                &self.scratch_buffers,
                self.cache_layers.clone(),
                self.shared_buffers.clone(),
                &token_ids,
                &[self.next_position + pass],
                token_bitmask,
                &[0],
                1,
                0,
                1,
                false,
                None,
                pass > 0,
                self.should_fill_attention_bias,
                Some((self.async_chain_positions.clone(), pass)),
                Some((self.async_chain_seeds.clone(), pass)),
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling.method());
            }

            let encoding_parameters = EncodingParameters::new();
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.executables
                    .embed
                    .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                for layer in self.executables.layers.iter() {
                    layer
                        .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                        .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                }
                self.executables
                    .norm
                    .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                self.executables
                    .readout
                    .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            let count =
                unsafe { *(self.async_chain_repetition_counts.borrow().cpu_ptr().as_ptr() as *const u32).add(pass) };
            let tokens_offset = pass * self.repetition_capacity * std::mem::size_of::<u32>();
            let counts_offset = pass * std::mem::size_of::<u32>();
            self.encode_repetition_penalty_if_needed_with_buffers(
                sampling,
                count,
                (self.async_chain_repetition_tokens.clone(), tokens_offset),
                (self.async_chain_repetition_counts.clone(), counts_offset),
                1_u32,
                self.repetition_capacity as u32,
            )?;
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.sampler
                    .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }

            let sampling_output = state.sampling_output().ok_or(Error::GenerateFailed)?;
            let sampling_output_binding = sampling_output.borrow();
            let sampling_output_buffer = sampling_output_binding.buffer();
            let token_ids_binding = self.scratch_buffers.token_ids.borrow();
            let token_ids_buffer = token_ids_binding.buffer();

            self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
                if pass + 1 < followup_count {
                    self.token_copy_sampled.encode(
                        sampling_output_buffer.clone(),
                        token_ids_buffer.clone(),
                        encoder,
                    );
                }
                let results_offset = pass * std::mem::size_of::<u32>();
                self.token_copy_results.encode(
                    sampling_output_buffer.clone(),
                    (self.async_chain_results.clone(), results_offset),
                    encoder,
                );
            });

            self.cache_layers.borrow_mut().update_after_acceptance(
                &[0],
                None,
                self.command_buffer.borrow_mut().deref_mut(),
                &self.kv_cache_update,
            );
            self.cache_layers.borrow_mut().register_accepted_tokens(&[self.next_position + pass]);
        }

        self.next_position = self.next_position.saturating_add(followup_count);
        self.submit_and_wait_current_command_buffer()?;

        let results_ptr = self.async_chain_results.borrow().cpu_ptr().as_ptr() as *const u32;
        for pass in 0..followup_count {
            let sampled = unsafe { *results_ptr.add(pass) };
            on_token(pass, u64::from(sampled))?;
        }
        Ok(())
    }

    fn decode_followup_tokens_sequential(
        &mut self,
        first_codebook_index: usize,
        mut previous_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        repetition_windows: Option<&[Vec<u32>]>,
        mut on_token: impl FnMut(usize, u64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for pass in 0..followup_count {
            if let Some(windows) = repetition_windows {
                let row = first_codebook_index + pass;
                self.set_repetition_window(windows.get(row).map_or(&[], Vec::as_slice))?;
            } else {
                self.clear_repetition_window();
            }
            let sampled = self.decode_next_token(&[previous_token], EmbeddingInjection::None, vocab_limit, sampling)?;
            on_token(pass, sampled)?;
            previous_token = sampled;
        }
        Ok(())
    }

    fn prepare_single_token_vocab_mask(
        &mut self,
        vocab_limit: usize,
    ) -> Result<(), Error> {
        let limit = vocab_limit.min(self.decoder_config.vocab_size);
        if limit == 0 || limit >= self.decoder_config.vocab_size {
            return Ok(());
        }
        if self.single_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.decoder_config.vocab_size.div_ceil(32);
        let mut mask = vec![0_u32; row_words];
        for token_index in 0..limit {
            let word = token_index / 32;
            let bit = token_index % 32;
            mask[word] |= 1_u32 << bit;
        }
        self.single_token_vocab_masks.insert(limit, mask.into_boxed_slice());
        Ok(())
    }

    fn prepare_two_token_vocab_mask(
        &mut self,
        vocab_limit: usize,
    ) -> Result<(), Error> {
        let limit = vocab_limit.min(self.decoder_config.vocab_size);
        if limit == 0 || limit >= self.decoder_config.vocab_size {
            return Ok(());
        }
        if self.two_token_vocab_masks.contains_key(&limit) {
            return Ok(());
        }
        let row_words = self.decoder_config.vocab_size.div_ceil(32);
        let mut mask = vec![0_u32; row_words.checked_mul(2).ok_or(Error::GenerateFailed)?];
        for token_index in 0..limit {
            let word = token_index / 32;
            let bit = token_index % 32;
            mask[row_words + word] |= 1_u32 << bit;
        }
        self.two_token_vocab_masks.insert(limit, mask.into_boxed_slice());
        Ok(())
    }

    fn get_single_token_vocab_mask(
        &self,
        vocab_limit: usize,
    ) -> Option<&[u32]> {
        self.single_token_vocab_masks.get(&vocab_limit).map(|mask| mask.as_ref())
    }

    fn get_two_token_vocab_mask(
        &self,
        vocab_limit: usize,
    ) -> Option<&[u32]> {
        self.two_token_vocab_masks.get(&vocab_limit).map(|mask| mask.as_ref())
    }

    fn decode_next_step(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        sampling: &mut TextSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
        capture_hidden: bool,
        mut pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_>>,
    ) -> Result<u64, Error> {
        objc2::rc::autoreleasepool(|_| {
            if token_ids.is_empty() {
                return Err(Error::GenerateFailed);
            }

            let token_count = token_ids.len();
            let sampling_start = token_count - 1;
            let sampling_length = 1usize;

            let mut single_position = [0_usize; 1];
            let mut two_positions = [0_usize; 2];
            let positions_storage;
            let positions: &[usize] = if token_count == 1 {
                single_position[0] = self.next_position;
                &single_position
            } else if token_count == 2 {
                two_positions[0] = self.next_position;
                two_positions[1] = self.next_position + 1;
                &two_positions
            } else {
                positions_storage = (self.next_position..self.next_position + token_count).collect::<Vec<_>>();
                positions_storage.as_slice()
            };

            let mut single_seed = [0_u64; 1];
            let mut two_seeds = [0_u64; 2];
            let mut token_seeds_storage;
            let token_seeds: &mut [u64] = if token_count == 1 {
                &mut single_seed
            } else if token_count == 2 {
                &mut two_seeds
            } else {
                token_seeds_storage = vec![0_u64; token_count];
                token_seeds_storage.as_mut_slice()
            };
            if matches!(sampling.method(), SamplingMethod::Stochastic { .. }) {
                token_seeds[sampling_start] = sampling.next_seed();
            }

            enum TokenBitmaskSource<'a> {
                None,
                Borrowed(&'a [u32]),
                Owned(Vec<u32>),
            }

            let row_words = self.decoder_config.vocab_size.div_ceil(32);
            let token_bitmask_source = if let Some(mask) = precomputed_token_bitmask {
                let expected_words = token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?;
                if mask.len() != expected_words {
                    return Err(Error::GenerateFailed);
                }
                TokenBitmaskSource::Borrowed(mask)
            } else if let Some(limit_raw) = vocab_limit {
                let limit = limit_raw.min(self.decoder_config.vocab_size);
                if limit == 0 {
                    return Err(Error::GenerateFailed);
                }
                if limit >= self.decoder_config.vocab_size {
                    TokenBitmaskSource::None
                } else if token_count == 1 {
                    if let Some(mask) = self.get_single_token_vocab_mask(limit) {
                        TokenBitmaskSource::Borrowed(mask)
                    } else {
                        let mut mask = vec![0_u32; row_words];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[word] |= 1_u32 << bit;
                        }
                        TokenBitmaskSource::Owned(mask)
                    }
                } else if token_count == 2 {
                    if let Some(mask) = self.get_two_token_vocab_mask(limit) {
                        TokenBitmaskSource::Borrowed(mask)
                    } else {
                        let mut mask = vec![0_u32; token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?];
                        for token_index in 0..limit {
                            let word = token_index / 32;
                            let bit = token_index % 32;
                            mask[sampling_start * row_words + word] |= 1_u32 << bit;
                        }
                        TokenBitmaskSource::Owned(mask)
                    }
                } else {
                    let mut mask = vec![0_u32; token_count.checked_mul(row_words).ok_or(Error::GenerateFailed)?];
                    for token_index in 0..limit {
                        let word = token_index / 32;
                        let bit = token_index % 32;
                        mask[sampling_start * row_words + word] |= 1_u32 << bit;
                    }
                    TokenBitmaskSource::Owned(mask)
                }
            } else {
                TokenBitmaskSource::None
            };

            let token_bitmask: Option<&[u32]> = match &token_bitmask_source {
                TokenBitmaskSource::None => None,
                TokenBitmaskSource::Borrowed(mask) => Some(*mask),
                TokenBitmaskSource::Owned(mask) => Some(mask.as_slice()),
            };

            let mut state = ForwardPassState::new_llm(
                self.context.clone(),
                &self.decoder_config,
                &self.model_shape,
                &self.scratch_buffers,
                self.cache_layers.clone(),
                self.shared_buffers.clone(),
                token_ids,
                positions,
                token_bitmask,
                token_seeds,
                token_count,
                sampling_start,
                sampling_length,
                false,
                None,
                false,
                self.should_fill_attention_bias,
                None,
                None,
            );
            if let Some(method) = state.sampling_method_mut() {
                *method = Some(sampling.method());
            }

            let encoding_parameters = EncodingParameters::new();
            let mut single_accepted = [0_usize; 1];
            let two_accepted = [0_usize, 1_usize];
            let accepted_suffix_indices_storage;
            let accepted_suffix_indices: &[usize] = if token_count == 1 {
                single_accepted[0] = 0;
                &single_accepted
            } else if token_count == 2 {
                &two_accepted
            } else {
                accepted_suffix_indices_storage = (0..token_count).collect::<Vec<_>>();
                accepted_suffix_indices_storage.as_slice()
            };

            if matches!(embedding_injection, EmbeddingInjection::OverrideFirstRowInternal) && capture_hidden {
                return Err(Error::GenerateFailed);
            }
            self.command_buffer =
                Rc::new(RefCell::new(self.context.create_command_buffer().expect("Failed to create command buffer")));
            {
                let mut command_buffer = self.command_buffer.borrow_mut();
                self.executables
                    .embed
                    .encode(&mut state, &encoding_parameters, command_buffer.deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
                if let Some(pre_encode) = pre_injection_encode.as_mut() {
                    pre_encode(self, &state, command_buffer.deref_mut())?;
                }
            }
            match embedding_injection {
                EmbeddingInjection::None => {},
                EmbeddingInjection::AddPreloaded {
                    post_scale,
                } => {
                    self.encode_add_scale_from_single_bias(&state, token_count, post_scale.unwrap_or(1.0))?;
                },
                EmbeddingInjection::OverrideFirstRowInternal => {
                    self.encode_override_first_row_from_device(&state, &self.single_override_embedding)?;
                },
            }
            for layer in self.executables.layers.iter() {
                layer
                    .encode(&mut state, &encoding_parameters, self.command_buffer.borrow_mut().deref_mut())
                    .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            }
            if capture_hidden {
                self.encode_capture_last_hidden_into_single_buffer(&state, token_count)?;
            }
            self.executables
                .norm
                .encode(&mut state, &encoding_parameters, self.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.executables
                .readout
                .encode(
                    &mut state,
                    &encoding_parameters,
                    self.command_buffer.borrow_mut().deref_mut(),
                )
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.encode_repetition_penalty_if_needed(sampling)?;
            self.sampler
                .encode(&mut state, &encoding_parameters, self.command_buffer.borrow_mut().deref_mut())
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
            self.cache_layers.borrow_mut().update_after_acceptance(
                accepted_suffix_indices,
                None,
                self.command_buffer.borrow_mut().deref_mut(),
                &self.kv_cache_update,
            );
            self.submit_and_wait_current_command_buffer()?;
            let token = read_sampled_token_from_sampling_output(&state)?;
            self.cache_layers.borrow_mut().register_accepted_tokens(positions);
            self.next_position = self.next_position.saturating_add(token_count);
            Ok(token)
        })
    }

    fn encode_capture_last_hidden_into_single_buffer(
        &self,
        state: &ForwardPassState<Metal>,
        token_count: usize,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let bytes_per_element = main.data_type().size_in_bytes();
        let row_offset = (token_count - 1)
            .checked_mul(model_dim)
            .and_then(|value| value.checked_mul(bytes_per_element))
            .ok_or(Error::GenerateFailed)?;
        let src_offset = main.offset().checked_add(row_offset).ok_or(Error::GenerateFailed)?;
        let capture = self.single_hidden_capture.borrow();
        if capture.shape() != [1, model_dim] || capture.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            self.tensor_copy.encode((main.buffer(), src_offset), capture.buffer(), model_dim_u32, encoder);
        });
        Ok(())
    }

    fn encode_override_first_row_from_device(
        &self,
        state: &ForwardPassState<Metal>,
        override_embedding: &ArrayCell<Metal>,
    ) -> Result<(), Error> {
        let model_dim = self.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let override_embedding = override_embedding.borrow();
        if override_embedding.shape() != [1, model_dim] || override_embedding.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            self.tensor_copy.encode(
                (override_embedding.buffer(), override_embedding.offset()),
                (main.buffer(), main.offset()),
                model_dim_u32,
                encoder,
            );
        });
        Ok(())
    }

    fn encode_add_scale_from_single_bias(
        &self,
        state: &ForwardPassState<Metal>,
        token_count: usize,
        scale: f32,
    ) -> Result<(), Error> {
        if token_count == 0 {
            return Err(Error::GenerateFailed);
        }
        let model_dim = self.decoder_config.model_dim;
        let model_dim_u32 = u32::try_from(model_dim).map_err(|_| Error::GenerateFailed)?;
        let total_len = token_count.checked_mul(model_dim).ok_or(Error::GenerateFailed)?;
        let total_len_u32 = u32::try_from(total_len).map_err(|_| Error::GenerateFailed)?;

        let main = state.arrays(&[ArrayId::Main])[0].clone();
        let main = main.borrow();
        let bias = self.single_override_embedding.borrow();
        if bias.shape() != [1, model_dim] || bias.data_type() != main.data_type() {
            return Err(Error::GenerateFailed);
        }

        self.command_buffer.borrow_mut().with_compute_encoder(|encoder| {
            self.tensor_add_scale.encode(
                (main.buffer(), main.offset()),
                bias.buffer(),
                (main.buffer(), main.offset()),
                model_dim_u32,
                total_len_u32,
                scale,
                encoder,
            );
        });
        Ok(())
    }

    fn submit_and_wait_current_command_buffer(&mut self) -> Result<(), Error> {
        self.command_buffer.borrow_mut().submit();
        self.instrumentation.command_buffers_submitted += 1;
        self.command_buffer
            .borrow()
            .wait_until_completed()
            .map_err(|err| Error::CommandBufferFailed(Box::new(err)))?;
        self.instrumentation.host_waits += 1;
        Ok(())
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
    }

    fn clear_instrumentation(&mut self) {
        self.instrumentation = RunnerInstrumentation::default();
    }
}

fn read_sampled_token_from_sampling_output(state: &ForwardPassState<Metal>) -> Result<u64, Error> {
    let output = state.sampling_output().ok_or(Error::GenerateFailed)?;
    let output = output.borrow();
    let tokens = output.as_slice::<u32>();
    let token = tokens.first().copied().ok_or(Error::GenerateFailed)?;
    Ok(u64::from(token))
}

fn write_f32_slice_into_array(
    array: &mut crate::array::Array<Metal>,
    values: &[f32],
) -> Result<(), Error> {
    if array.num_elements() != values.len() {
        return Err(Error::GenerateFailed);
    }
    match array.data_type() {
        DataType::F32 => {
            array.as_slice_mut::<f32>().copy_from_slice(values);
            Ok(())
        },
        DataType::F16 => {
            for (dst, &src) in array.as_slice_mut::<f16>().iter_mut().zip(values.iter()) {
                *dst = f16::from_f32(src);
            }
            Ok(())
        },
        DataType::BF16 => {
            for (dst, &src) in array.as_slice_mut::<bf16>().iter_mut().zip(values.iter()) {
                *dst = bf16::from_f32(src);
            }
            Ok(())
        },
        _ => Err(Error::GenerateFailed),
    }
}

impl MatrixF32 {
    #[cfg(test)]
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

    #[cfg(test)]
    fn matmul_into(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Option<()> {
        if input.len() != self.cols || output.len() != self.rows {
            return None;
        }
        for (row_index, row) in self.values.chunks_exact(self.cols).enumerate() {
            let mut acc = 0.0_f32;
            for (&w, &x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            output[row_index] = acc;
        }
        Some(())
    }
}

fn load_matrix_f32(
    weights_path: &Path,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<MatrixF32, Error> {
    let file = File::open(weights_path).map_err(|_| Error::UnableToLoadWeights)?;
    let (global_offset, metadata) = read_safetensors_metadata(&file).map_err(|_| Error::UnableToLoadWeights)?;
    let tensor = metadata.tensors.get(key).ok_or(Error::UnableToLoadWeights)?;

    if tensor.shape.len() != 2 {
        return Err(Error::UnableToLoadWeights);
    }
    let rows = tensor.shape[0];
    let cols = tensor.shape[1];
    if rows != expected_rows || cols != expected_cols {
        return Err(Error::UnableToLoadConfig);
    }

    let (begin, end) = tensor.data_offsets;
    let size = end.checked_sub(begin).ok_or(Error::UnableToLoadWeights)?;
    let offset = global_offset.checked_add(begin).ok_or(Error::UnableToLoadWeights)?;
    let data_type: DataType = tensor.dtype.into();
    let expected_size = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(data_type.size_in_bytes()))
        .ok_or(Error::UnableToLoadWeights)?;
    if size != expected_size {
        return Err(Error::UnableToLoadWeights);
    }

    let mut bytes = vec![0_u8; size];
    file.read_exact_at(&mut bytes, offset as u64).map_err(|_| Error::UnableToLoadWeights)?;

    let values = match data_type {
        DataType::F32 => decode_f32_bytes(&bytes),
        DataType::F16 => decode_f16_bytes_to_f32(&bytes),
        DataType::BF16 => decode_bf16_bytes_to_f32(&bytes),
        _ => return Err(Error::UnableToLoadWeights),
    };
    if values.len() != rows.checked_mul(cols).ok_or(Error::UnableToLoadWeights)? {
        return Err(Error::UnableToLoadWeights);
    }

    Ok(MatrixF32 {
        rows,
        cols,
        values,
    })
}

fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn decode_f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|chunk| f32::from(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))).collect()
}

fn decode_bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|chunk| f32::from(bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))).collect()
}

fn load_stub_seed(weights_path: PathBuf) -> Option<u64> {
    let file = File::open(weights_path).ok()?;
    let (global_offset, metadata) = read_safetensors_metadata(&file).ok()?;
    let tensor = metadata.tensors.get("text_decoder.seed")?;

    let (begin, end) = tensor.data_offsets;
    let size = end.checked_sub(begin)?;
    let data_type: DataType = tensor.dtype.into();
    let offset = global_offset.checked_add(begin)?;

    match data_type {
        DataType::I32 if size == 4 => {
            let mut bytes = [0_u8; 4];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            let value = i32::from_le_bytes(bytes);
            (value >= 0).then_some(value as u64)
        },
        DataType::I64 if size == 8 => {
            let mut bytes = [0_u8; 8];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            let value = i64::from_le_bytes(bytes);
            (value >= 0).then_some(value as u64)
        },
        DataType::U64 if size == 8 => {
            let mut bytes = [0_u8; 8];
            file.read_exact_at(&mut bytes, offset as u64).ok()?;
            Some(u64::from_le_bytes(bytes))
        },
        _ => None,
    }
}

fn generate_stub_tokens(
    num_codebooks: usize,
    frames: usize,
    token_upper_bound: usize,
    seed: u64,
) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = Vec::with_capacity(num_codebooks * frames);
    for _codebook in 0..num_codebooks {
        for _frame in 0..frames {
            tokens.push(rng.random_range(0..token_upper_bound) as u32);
        }
    }
    tokens
}

fn generate_stub_semantic_grid(
    stub: &StubTextDecoderRuntime,
    text_tokens: &[u64],
    codec_cardinality: usize,
    seed: u64,
    max_semantic_frames: usize,
) -> Result<AudioTokenGrid, Error> {
    let frames = text_tokens.len().min(max_semantic_frames.max(1));
    let token_upper_bound = stub.codebook_size.min(codec_cardinality);
    if token_upper_bound == 0 {
        return Err(Error::UnableToLoadConfig);
    }

    let tokens = generate_stub_tokens(stub.num_codebooks, frames, token_upper_bound, seed);
    AudioTokenGrid::new(
        tokens.into_boxed_slice(),
        1,
        stub.num_codebooks,
        frames,
        vec![frames].into_boxed_slice(),
        AudioTokenPacking::CodebookMajor,
    )
    .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveChunkController, DEFAULT_CHUNK_EMA_ALPHA, DEFAULT_CHUNK_HYSTERESIS_FRACTION, DEFAULT_STUB_SEED,
        DEFAULT_TTS_RANDOM_SEED, MatrixF32, StreamingTokenAccumulator, TextSamplingState,
        TextDecoderFollowupStrategy, build_semantic_sampling_mask_row, clear_token_in_sampling_mask,
        expand_token_mask_for_sampling_row, generate_stub_tokens, load_stub_seed, normalize_rendered_prompt,
        semantic_token_to_code,
    };
    use crate::audio::AudioTokenPacking;
    use crate::session::config::{TextDecoderRuntimeConfig, TextSamplingConfig, TtsChunkPolicy, TtsRunConfig};
    use crate::session::parameter::SamplingMethod;

    #[test]
    fn missing_seed_file_uses_default_path() {
        let seed = load_stub_seed("/does/not/exist/model.safetensors".into()).unwrap_or(DEFAULT_STUB_SEED);
        assert_eq!(seed, DEFAULT_STUB_SEED);
    }

    #[test]
    fn stub_token_generation_is_seeded_and_bounded() {
        let a = generate_stub_tokens(3, 8, 17, 123);
        let b = generate_stub_tokens(3, 8, 17, 123);
        let c = generate_stub_tokens(3, 8, 17, 456);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|&value| value < 17));
    }

    #[test]
    fn semantic_token_to_code_respects_bounds() {
        assert_eq!(semantic_token_to_code(5, 5, 9, 4), 0);
        assert_eq!(semantic_token_to_code(7, 5, 9, 4), 2);
        assert_eq!(semantic_token_to_code(9, 5, 9, 4), 3);
        assert_eq!(semantic_token_to_code(11, 5, 9, 4), 0);
        assert_eq!(semantic_token_to_code(7, 10, 9, 4), 0);
        assert_eq!(semantic_token_to_code(7, 5, 9, 0), 0);
    }

    #[test]
    fn semantic_sampling_mask_row_includes_band_and_im_end() {
        let mask = build_semantic_sampling_mask_row(96, 64, 79, 12).expect("mask");
        let bit = |token: usize| -> bool {
            let word = token / 32;
            let offset = token % 32;
            ((mask[word] >> offset) & 1) == 1
        };

        assert!(bit(12), "im_end must be selectable");
        assert!(bit(64) && bit(79), "semantic range endpoints must be selectable");
        assert!(!bit(63), "tokens below semantic range must be masked");
        assert!(!bit(80), "tokens above semantic range must be masked");
    }

    #[test]
    fn rendered_prompt_drops_template_newlines() {
        let template = "\n{% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}\n";
        let rendered = "\n<|interleave|><|speaker:0|>hello\n".to_string();
        let normalized = normalize_rendered_prompt(rendered, template, true);
        assert_eq!(normalized, "<|interleave|><|speaker:0|>hello");
    }

    #[test]
    fn expanded_sampling_mask_targets_only_sampling_row() {
        let row_mask = vec![0xFFFF_0000u32, 0x0000_FFFFu32];
        let expanded = expand_token_mask_for_sampling_row(&row_mask, 3).expect("expanded");
        assert_eq!(expanded.len(), 6);
        assert_eq!(&expanded[0..2], &[u32::MAX, u32::MAX]);
        assert_eq!(&expanded[2..4], &[u32::MAX, u32::MAX]);
        assert_eq!(&expanded[4..6], row_mask.as_slice());
    }

    #[test]
    fn text_decoder_defaults_are_stable() {
        let config = TextDecoderRuntimeConfig::default();
        assert_eq!(DEFAULT_TTS_RANDOM_SEED, 123);
        assert_eq!(config.min_frames_before_im_end, 8);
        assert_eq!(config.prefill_step_size, 128);
        assert_eq!(config.followup_strategy, TextDecoderFollowupStrategy::AsyncChain);
        assert_eq!(config.sampling.temperature, 0.8008);
        assert_eq!(config.sampling.top_p, 0.8008);
        assert_eq!(config.sampling.repetition_penalty, 1.1);
    }

    #[test]
    fn clearing_token_from_sampling_mask_removes_bit() {
        let mut mask = build_semantic_sampling_mask_row(96, 64, 79, 12).expect("mask").into_vec();
        clear_token_in_sampling_mask(&mut mask, 12).expect("clear");
        let word = 12 / 32;
        let bit = 12 % 32;
        assert_eq!((mask[word] >> bit) & 1, 0);
    }

    #[test]
    fn fishaudio_sampling_seed_stream_is_deterministic() {
        let config = TextSamplingConfig::default();
        let mut a = TextSamplingState::from_config(123, &config);
        let mut b = TextSamplingState::from_config(123, &config);
        let seeds_a = (0..8).map(|_| a.next_seed()).collect::<Vec<_>>();
        let seeds_b = (0..8).map(|_| b.next_seed()).collect::<Vec<_>>();
        assert_eq!(seeds_a, seeds_b);
    }

    #[test]
    fn fishaudio_sampling_top_p_zero_switches_to_greedy() {
        let sampler = TextSamplingState::with_params(999, 0.8, 0.0, 1.1);
        assert_eq!(sampler.method(), SamplingMethod::Greedy);
    }

    #[test]
    fn fishaudio_sampling_positive_top_p_uses_stochastic_mode() {
        let sampler = TextSamplingState::with_params(999, 0.8, 0.8, 1.1);
        assert!(matches!(sampler.method(), SamplingMethod::Stochastic { .. }));
    }

    #[test]
    fn matrix_f32_row_and_matmul_work() {
        let matrix = MatrixF32 {
            rows: 2,
            cols: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        assert_eq!(matrix.row(0), Some([1.0_f32, 2.0, 3.0].as_slice()));
        assert_eq!(matrix.row(1), Some([4.0_f32, 5.0, 6.0].as_slice()));
        assert_eq!(matrix.row(2), None);
        let mut out = [0.0_f32; 2];
        assert_eq!(matrix.matmul_into(&[1.0, 0.0, 1.0], &mut out), Some(()));
        assert_eq!(out, [4.0, 10.0]);
        assert_eq!(matrix.matmul_into(&[1.0, 0.0], &mut out), None);
    }

    #[test]
    fn streaming_token_accumulator_builds_codebook_major_grid() {
        let mut accumulator = StreamingTokenAccumulator::new(3).expect("accumulator");
        accumulator.push_frame(&[1, 10, 100]).expect("frame 0");
        accumulator.push_frame(&[2, 20, 200]).expect("frame 1");

        let grid = accumulator.to_grid().expect("grid");
        assert_eq!(grid.batch_size(), 1);
        assert_eq!(grid.codebooks(), 3);
        assert_eq!(grid.frames(), 2);
        assert_eq!(grid.packing(), AudioTokenPacking::CodebookMajor);
        assert_eq!(grid.tokens(), &[1, 2, 10, 20, 100, 200]);
    }

    #[test]
    fn streaming_token_accumulator_rejects_wrong_frame_width() {
        let mut accumulator = StreamingTokenAccumulator::new(2).expect("accumulator");
        let error = accumulator.push_frame(&[1]).expect_err("must reject wrong width");
        assert!(matches!(error, crate::session::types::Error::GenerateFailed));
    }

    #[test]
    fn adaptive_chunk_controller_applies_hysteresis() {
        let config = TtsRunConfig {
            min_chunk_frames: 16,
            max_chunk_frames: 256,
            target_emit_latency_ms: 80,
            chunk_policy: TtsChunkPolicy::Adaptive,
            ..TtsRunConfig::default()
        };
        let mut controller = AdaptiveChunkController::new(&config);
        assert_eq!(controller.target_frames(&config), 16);

        controller.observe(16, std::time::Duration::from_millis(32), 16);
        let first = controller.target_frames(&config);
        assert!(first >= 16);

        // Small fluctuations under hysteresis threshold should keep the current chunk size.
        let near_first = ((first as f64) * (1.0 + DEFAULT_CHUNK_HYSTERESIS_FRACTION / 2.0)).round() as usize;
        controller.current_chunk_frames = first;
        controller.ema_ms_per_frame = Some(config.target_emit_latency_ms as f64 / near_first as f64);
        assert_eq!(controller.target_frames(&config), first);

        // Larger changes should trigger chunk-size updates.
        let far_target =
            ((first as f64) * (1.0 + DEFAULT_CHUNK_HYSTERESIS_FRACTION + DEFAULT_CHUNK_EMA_ALPHA)).round() as usize;
        controller.ema_ms_per_frame = Some(config.target_emit_latency_ms as f64 / far_target as f64);
        assert_ne!(controller.target_frames(&config), first);
    }

    #[test]
    fn adaptive_chunk_controller_does_not_shrink_mid_run() {
        let config = TtsRunConfig {
            min_chunk_frames: 16,
            max_chunk_frames: 256,
            target_emit_latency_ms: 80,
            chunk_policy: TtsChunkPolicy::Adaptive,
            ..TtsRunConfig::default()
        };
        let mut controller = AdaptiveChunkController::new(&config);
        controller.current_chunk_frames = 96;
        controller.ema_ms_per_frame = Some(40.0); // candidate would be 16
        assert_eq!(controller.target_frames(&config), 96);
    }

    #[test]
    fn startup_target_frames_backoff_caps_at_startup_cap() {
        assert_eq!(super::next_startup_target_frames(1, 64), 2);
        assert_eq!(super::next_startup_target_frames(2, 64), 4);
        assert_eq!(super::next_startup_target_frames(8, 64), 16);
        assert_eq!(super::next_startup_target_frames(32, 64), 64);
        assert_eq!(super::next_startup_target_frames(64, 64), 64);
        assert_eq!(super::next_startup_target_frames(128, 64), 64);
    }

    #[test]
    fn adaptive_chunk_controller_scales_up_when_decode_lags_realtime() {
        let config = TtsRunConfig {
            min_chunk_frames: 16,
            max_chunk_frames: 256,
            ..TtsRunConfig::default()
        };
        let mut controller = AdaptiveChunkController::new(&config);
        controller.current_chunk_frames = 16;

        controller.adapt_up_for_realtime(&config, 16, 44_100, std::time::Duration::from_millis(500), 8_192);
        assert!(controller.current_chunk_frames > 16);

        let after_up = controller.current_chunk_frames;
        controller.adapt_up_for_realtime(&config, 16, 44_100, std::time::Duration::from_millis(20), 8_192);
        assert_eq!(controller.current_chunk_frames, after_up);
    }

    #[test]
    fn adaptive_chunk_controller_promotes_to_max_chunk() {
        let config = TtsRunConfig {
            min_chunk_frames: 16,
            max_chunk_frames: 256,
            ..TtsRunConfig::default()
        };
        let mut controller = AdaptiveChunkController::new(&config);
        controller.current_chunk_frames = 32;
        controller.promote_to_max_chunk(&config);
        assert_eq!(controller.current_chunk_frames, 256);
    }
}
