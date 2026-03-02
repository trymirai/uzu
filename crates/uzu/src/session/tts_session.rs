#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    ops::Deref,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use half::{bf16, f16};
use minijinja::{Environment, context};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    DataType,
    array::{ArrayCell, ArrayContextExt},
    audio::{AudioCodecRuntime, AudioPcmBatch, AudioTokenGrid, AudioTokenPacking, nanocodec::AudioDecodeStreamingMode},
    backends::{
        common::{
            Backend, CommandBuffer, Context as BackendContext, Kernels, NativeBuffer,
            kernel::{
                EmbeddingRowsSumKernel, TensorAddScaleKernel, TensorCopyKernel, TokenCopySampledKernel,
                TokenCopyToResultsKernel,
                kv_cache_update::KVCacheUpdate,
                matmul::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel, MatmulKernels},
            },
        },
        metal::Metal,
    },
    config::{ConfigDataType, InnerModelConfig, LinearConfig, ModelMetadata, ModelType, TtsModelConfig},
    encodable_block::{Decoder, EncodableBlock, EncodingParameters, Sampling as GpuSampling},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, SharedBuffers},
    },
    parameters::{ParameterLoader, read_safetensors_metadata},
    session::{
        config::{TtsChunkPolicy, TtsRunConfig, TtsVocoderStreamingMode},
        parameter::SamplingMethod,
        types::{Error, Input},
    },
};

const DEFAULT_STUB_SPEAKER_ID: &str = "speaker:0";
const DEFAULT_STUB_STYLE: &str = "interleave";
const DEFAULT_STUB_SEED: u64 = 123;
const DEFAULT_FISHAUDIO_RANDOM_SEED: u64 = 123;
const DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE: usize = 1024;
const DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE: usize = 16;
const DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE: f32 = 0.8008;
const DEFAULT_FISHAUDIO_SAMPLING_TOP_P: f32 = 0.8008;
const DEFAULT_FISHAUDIO_MIN_FRAMES_BEFORE_IM_END: usize = 8;
const DEFAULT_FISHAUDIO_PREFILL_STEP_SIZE: usize = 128;
const DEFAULT_CHUNK_EMA_ALPHA: f64 = 0.2;
const DEFAULT_CHUNK_HYSTERESIS_FRACTION: f64 = 0.25;

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
    runtime: crate::audio::NanoCodecFsqRuntime,
    prompt_template: String,
    drop_initial_newline: bool,
    text_decoder: RefCell<TextDecoderRuntime>,
    last_execution_stats: RefCell<Option<TtsExecutionStats>>,
}

enum TextDecoderRuntime {
    Stub(StubTextDecoderRuntime),
    FishAudio(FishAudioTextDecoderRuntime),
}

#[derive(Clone, Copy)]
struct StubTextDecoderRuntime {
    num_codebooks: usize,
    codebook_size: usize,
    default_seed: u64,
}

struct FishAudioTextDecoderRuntime {
    slow_runner: TokenDecoderRunner,
    fast_runner: Option<TokenDecoderRunner>,
    gpu_path: FishAudioGpuPath,
    semantic_token_begin_id: i64,
    semantic_token_end_id: i64,
    semantic_cardinality: usize,
    im_end_token_id: i64,
    codebook_size: usize,
    num_codebooks: usize,
    slow_model_dim: usize,
    fast_model_dim: usize,
    max_seq_len: usize,
    scale_codebook_embeddings: bool,
    fast_vocab_limit: usize,
    apply_semantic_sampling_mask: bool,
    semantic_sampling_mask_row: Box<[u32]>,
    semantic_sampling_mask_without_im_end_row: Box<[u32]>,
    min_frames_before_im_end: usize,
    current_codes_scratch: Vec<u32>,
    instrumentation: RunnerInstrumentation,
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
    kv_cache_update: KVCacheUpdate<Metal>,
    tensor_copy: <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel,
    tensor_add_scale: <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel,
    token_copy_sampled: <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel,
    token_copy_results: <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel,
    async_chain_positions: Rc<<Metal as Backend>::NativeBuffer>,
    async_chain_seeds: Rc<<Metal as Backend>::NativeBuffer>,
    async_chain_results: Rc<<Metal as Backend>::NativeBuffer>,
    async_chain_capacity: usize,
    single_hidden_capture: ArrayCell<Metal>,
    single_override_embedding: ArrayCell<Metal>,
    single_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    two_token_vocab_masks: HashMap<usize, Box<[u32]>>,
    should_fill_attention_bias: bool,
    next_position: usize,
    instrumentation: RunnerInstrumentation,
}

type PreInjectionEncodeCallback<'a> =
    dyn FnMut(&TokenDecoderRunner, &ForwardPassState<Metal>, &MetalCommandBuffer) -> Result<(), Error> + 'a;

struct MatrixF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

struct FishAudioGpuPath {
    embedding_rows_sum: <<Metal as Backend>::Kernels as Kernels>::EmbeddingRowsSumKernel,
    projection: <<Metal as Backend>::Kernels as MatmulKernels>::FullPrecisionMatmulKernel,
    tensor_copy: <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel,
    codebook_embeddings: ArrayCell<Metal>,
    codebook_row_indices: ArrayCell<Metal>,
    projection_weights: Option<ArrayCell<Metal>>,
}

enum EmbeddingInjection {
    None,
    AddPreloaded {
        post_scale: Option<f32>,
    },
    OverrideFirstRowInternal,
}

struct FishAudioSamplingState {
    rng: StdRng,
    method: SamplingMethod,
}

impl FishAudioSamplingState {
    fn new(seed: u64) -> Self {
        Self::with_params(seed, fishaudio_sampling_temperature(), fishaudio_sampling_top_p())
    }

    fn with_params(
        seed: u64,
        temperature: f32,
        top_p: f32,
    ) -> Self {
        let method = if temperature <= 0.0 || top_p <= 0.0 {
            SamplingMethod::Greedy
        } else {
            SamplingMethod::Stochastic {
                temperature: Some(temperature),
                top_k: None,
                top_p: Some(top_p),
                min_p: None,
            }
        };
        Self {
            rng: StdRng::seed_from_u64(seed),
            method,
        }
    }

    fn method(&self) -> SamplingMethod {
        self.method
    }

    fn next_seed(&mut self) -> u64 {
        self.rng.random::<u64>()
    }
}

#[derive(Debug, Deserialize)]
struct TtsConfigJson {
    text_decoder_config: TextDecoderConfigJson,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct FishAudioTextDecoderConfigJson {
    slow_embeddings_config: FishAudioEmbeddingConfigJson,
    slow_model_config: crate::config::TransformerConfig,
    slow_readout_config: FishAudioLinearConfigJson,
    fast_embeddings_config: FishAudioEmbeddingConfigJson,
    fast_model_config: crate::config::TransformerConfig,
    fast_readout_config: FishAudioLinearConfigJson,
    codebook_embeddings_config: FishAudioEmbeddingConfigJson,
    fast_model_projection_config: Option<FishAudioLinearConfigJson>,
    semantic_token_begin_id: i64,
    semantic_token_end_id: i64,
    im_end_token_id: i64,
    codebook_size: usize,
    vocab_size: usize,
    slow_model_dim: usize,
    fast_model_dim: usize,
    num_codebooks: usize,
    max_seq_len: usize,
    scale_codebook_embeddings: bool,
    #[serde(default = "default_short_logits_size")]
    short_logits_size: usize,
    #[serde(default = "default_repeat_window_size")]
    repeat_window_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FishAudioLinearConfigJson {
    Tagged(LinearConfig),
    UntaggedFullPrecision {
        #[serde(rename = "precision")]
        _precision: crate::ConfigDataType,
    },
}

impl FishAudioLinearConfigJson {
    fn is_full_precision(&self) -> bool {
        matches!(
            self,
            FishAudioLinearConfigJson::Tagged(LinearConfig::FullPrecision { .. })
                | FishAudioLinearConfigJson::UntaggedFullPrecision { .. }
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FishAudioEmbeddingConfigJson {
    Tagged(crate::config::EmbeddingConfig),
    UntaggedFullPrecision {
        input_scale: Option<f32>,
        logit_soft_cap: Option<f32>,
        precision: crate::ConfigDataType,
    },
}

impl FishAudioEmbeddingConfigJson {
    fn to_embedding_config(&self) -> crate::config::EmbeddingConfig {
        match self {
            FishAudioEmbeddingConfigJson::Tagged(config) => config.clone(),
            FishAudioEmbeddingConfigJson::UntaggedFullPrecision {
                input_scale,
                logit_soft_cap,
                precision,
            } => crate::config::EmbeddingConfig::Untied {
                common: crate::config::EmbeddingConfigCommon {
                    input_scale: *input_scale,
                    logit_soft_cap: *logit_soft_cap,
                },
                precision: *precision,
            },
        }
    }
}

fn default_short_logits_size() -> usize {
    DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE
}

fn default_repeat_window_size() -> usize {
    DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE
}

fn fishaudio_sampling_temperature() -> f32 {
    std::env::var("UZU_FISHAUDIO_SAMPLING_TEMPERATURE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE)
}

fn fishaudio_sampling_top_p() -> f32 {
    std::env::var("UZU_FISHAUDIO_SAMPLING_TOP_P")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0)
        .unwrap_or(DEFAULT_FISHAUDIO_SAMPLING_TOP_P)
}

fn fishaudio_min_frames_before_im_end() -> usize {
    std::env::var("UZU_FISHAUDIO_MIN_FRAMES_BEFORE_IM_END")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FISHAUDIO_MIN_FRAMES_BEFORE_IM_END)
}

fn fishaudio_max_new_tokens_override() -> Option<usize> {
    std::env::var("UZU_FISHAUDIO_MAX_NEW_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn fishaudio_force_semantic_sampling_mask() -> Option<bool> {
    std::env::var("UZU_FISHAUDIO_FORCE_SEMANTIC_MASK").ok().and_then(|value| {
        let value = value.trim().to_ascii_lowercase();
        match value.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn fishaudio_prefill_step_size(context_length: usize) -> usize {
    std::env::var("UZU_FISHAUDIO_PREFILL_STEP_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_FISHAUDIO_PREFILL_STEP_SIZE)
        .min(context_length.max(1))
        .max(1)
}

fn promote_transformer_norm_accumulation_to_f32(transformer: &mut crate::config::TransformerConfig) {
    for layer in transformer.layer_configs.iter_mut() {
        if let Some(norm) = layer.pre_attention_norm_config.as_mut() {
            norm.accumulation_precision = ConfigDataType::Float32;
        }
        layer.pre_mlp_norm_config.accumulation_precision = ConfigDataType::Float32;
        if let Some(norm) = layer.post_attention_norm_config.as_mut() {
            norm.accumulation_precision = ConfigDataType::Float32;
        }
        if let Some(norm) = layer.post_mlp_norm_config.as_mut() {
            norm.accumulation_precision = ConfigDataType::Float32;
        }
        if let crate::config::MixerConfig::Attention(attention) = &mut layer.mixer_config {
            if let Some(norm) = attention.query_norm_config.as_mut() {
                norm.accumulation_precision = ConfigDataType::Float32;
            }
            if let Some(norm) = attention.key_norm_config.as_mut() {
                norm.accumulation_precision = ConfigDataType::Float32;
            }
        }
    }
    transformer.output_norm_config.accumulation_precision = ConfigDataType::Float32;
}

impl FishAudioGpuPath {
    fn new(
        context: &MetalContext,
        data_type: DataType,
        codebook_embeddings: &MatrixF32,
        fast_model_projection: &Option<MatrixF32>,
        num_codebooks: usize,
        codebook_size: usize,
        slow_model_dim: usize,
        fast_model_dim: usize,
    ) -> Result<Self, Error> {
        if codebook_embeddings.rows != num_codebooks.checked_mul(codebook_size).ok_or(Error::UnableToLoadConfig)?
            || codebook_embeddings.cols != slow_model_dim
        {
            return Err(Error::UnableToLoadConfig);
        }
        if let Some(projection) = fast_model_projection
            && (projection.rows != fast_model_dim || projection.cols != slow_model_dim)
        {
            return Err(Error::UnableToLoadConfig);
        }

        let embedding_rows_sum =
            <<Metal as Backend>::Kernels as Kernels>::EmbeddingRowsSumKernel::new(context, data_type)
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let projection =
            <<<Metal as Backend>::Kernels as MatmulKernels>::FullPrecisionMatmulKernel as FullPrecisionMatmulKernel>::new(
                context, data_type,
            )
            .map_err(|_| Error::UnableToCreateBackendContext)?;
        let tensor_copy = <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel::new(context, data_type)
            .map_err(|_| Error::UnableToCreateBackendContext)?;

        let mut codebook_embeddings_gpu = context.create_array(
            &[num_codebooks.checked_mul(codebook_size).ok_or(Error::UnableToLoadConfig)?, slow_model_dim],
            data_type,
            "tts_codebook_embeddings_gpu",
        );
        write_f32_slice_into_array(&mut codebook_embeddings_gpu, &codebook_embeddings.values)?;

        let codebook_row_indices = context.create_array(&[num_codebooks], DataType::U64, "tts_codebook_row_indices");

        let projection_weights = if let Some(projection) = fast_model_projection {
            let mut weights =
                context.create_array(&[fast_model_dim, slow_model_dim], data_type, "tts_fast_projection_weights");
            write_f32_slice_into_array(&mut weights, &projection.values)?;
            Some(RefCell::new(weights))
        } else {
            None
        };

        Ok(Self {
            embedding_rows_sum,
            projection,
            tensor_copy,
            codebook_embeddings: RefCell::new(codebook_embeddings_gpu),
            codebook_row_indices: RefCell::new(codebook_row_indices),
            projection_weights,
        })
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

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum TextDecoderConfigJson {
    StubTextDecoderConfig {
        num_codebooks: usize,
        codebook_size: usize,
    },
    FishAudioTextDecoderConfig {
        #[serde(flatten)]
        config: FishAudioTextDecoderConfigJson,
    },
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

impl TtsSession {
    pub fn new(model_path: PathBuf) -> Result<Self, Error> {
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

        Self::from_model_metadata(model_path, model_metadata)
    }

    pub fn runtime(&self) -> &crate::audio::NanoCodecFsqRuntime {
        &self.runtime
    }

    pub fn last_execution_stats(&self) -> Option<TtsExecutionStats> {
        self.last_execution_stats.borrow().clone()
    }

    pub fn synthesize(
        &self,
        input: Input,
    ) -> Result<AudioPcmBatch, Error> {
        let seed = match *self.text_decoder.borrow().deref() {
            TextDecoderRuntime::Stub(stub) => stub.default_seed,
            TextDecoderRuntime::FishAudio(_) => DEFAULT_FISHAUDIO_RANDOM_SEED,
        };
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
        let seed = match *self.text_decoder.borrow().deref() {
            TextDecoderRuntime::Stub(stub) => stub.default_seed,
            TextDecoderRuntime::FishAudio(_) => DEFAULT_FISHAUDIO_RANDOM_SEED,
        };
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
        let pcm = match config.non_streaming_mode {
            crate::session::config::TtsNonStreamingMode::FullDecode => {
                self.runtime.decode(&semantic_tokens).map_err(Error::from)?
            },
            crate::session::config::TtsNonStreamingMode::ChunkedIfNeeded => {
                let total_frames = semantic_tokens.frames();
                let chunked_threshold = config.max_stream_workspace_frames.max(config.max_chunk_frames.max(1));
                if total_frames < chunked_threshold {
                    self.runtime.decode(&semantic_tokens).map_err(Error::from)?
                } else {
                    let chunk_frames = config.max_chunk_frames.max(config.min_chunk_frames.max(1));
                    let workspace_frames = config.max_stream_workspace_frames.max(chunk_frames);
                    let mut stream_state = self
                        .runtime
                        .begin_decode_stream_with_options(
                            semantic_tokens.batch_size(),
                            semantic_tokens.codebooks(),
                            AudioDecodeStreamingMode::IncrementalStateful,
                            workspace_frames,
                        )
                        .map_err(Error::from)?;

                    let mut all_samples = Vec::<f32>::new();
                    let mut accumulated_lengths = vec![0usize; semantic_tokens.batch_size()];
                    let mut sample_rate = self.runtime.config().sample_rate();
                    let mut channels = 1usize;

                    let mut frame_start = 0usize;
                    while frame_start < total_frames {
                        let frame_end = (frame_start + chunk_frames).min(total_frames);
                        let delta_grid = slice_codebook_major_grid_range(&semantic_tokens, frame_start, frame_end)?;
                        let decoded_delta = self
                            .runtime
                            .decode_stream_step(&mut stream_state, &delta_grid, frame_end == total_frames)
                            .map_err(Error::from)?;
                        let partial_pcm =
                            self.runtime.decoded_padded_to_pcm_batch(&decoded_delta).map_err(Error::from)?;
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
                    self.runtime.end_decode_stream(stream_state).map_err(Error::from)?;
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
        let seed = match *self.text_decoder.borrow().deref() {
            TextDecoderRuntime::Stub(stub) => stub.default_seed,
            TextDecoderRuntime::FishAudio(_) => DEFAULT_FISHAUDIO_RANDOM_SEED,
        };
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
        let seed = match *self.text_decoder.borrow().deref() {
            TextDecoderRuntime::Stub(stub) => stub.default_seed,
            TextDecoderRuntime::FishAudio(_) => DEFAULT_FISHAUDIO_RANDOM_SEED,
        };
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

        let mut streamed_tokens = StreamingTokenAccumulator::new(self.runtime.config().num_groups())?;
        let streaming_mode = match config.vocoder_streaming_mode {
            TtsVocoderStreamingMode::IncrementalStateful => AudioDecodeStreamingMode::IncrementalStateful,
            TtsVocoderStreamingMode::PrefixFallback => AudioDecodeStreamingMode::PrefixFallback,
        };
        let mut audio_stream_state = self
            .runtime
            .begin_decode_stream_with_options(
                1,
                self.runtime.config().num_groups(),
                streaming_mode,
                config.max_stream_workspace_frames,
            )
            .map_err(Error::from)?;
        let mut emitted_chunks = 0usize;
        let mut callback_seconds = 0.0_f64;
        let mut audio_decode_seconds = 0.0_f64;
        let mut last_decoded_frames = 0usize;
        let mut first_chunk_seconds = None::<f64>;
        let mut first_chunk_frames = 0usize;
        let mut first_emit_pending = true;
        let mut output_samples = Vec::<f32>::new();
        let mut output_frames = 0usize;
        let mut output_sample_rate = self.runtime.config().sample_rate();
        let mut output_channels = 1usize;
        let mut chunk_controller = AdaptiveChunkController::new(config);
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
                streamed_tokens.push_frame(codes)?;
                let ready_frames = streamed_tokens.frames().saturating_sub(last_decoded_frames);
                let adaptive_target_frames = chunk_controller.target_frames(config);
                let target_frames = if first_emit_pending {
                    startup_target_frames
                } else {
                    adaptive_target_frames
                };
                if ready_frames >= target_frames {
                    let decode_start = Instant::now();
                    let partial_grid = streamed_tokens.to_grid_range(last_decoded_frames, streamed_tokens.frames())?;
                    let decoded_delta = self
                        .runtime
                        .decode_stream_step(&mut audio_stream_state, &partial_grid, false)
                        .map_err(Error::from)?;
                    let partial_pcm = self.runtime.decoded_padded_to_pcm_batch(&decoded_delta).map_err(Error::from)?;
                    let decode_elapsed = decode_start.elapsed();
                    audio_decode_seconds += decode_elapsed.as_secs_f64();
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
                    callback_seconds += callback_start.elapsed().as_secs_f64();
                    last_decoded_frames = streamed_tokens.frames();
                    let next_chunk_frames = if first_emit_pending {
                        adaptive_target_frames
                    } else {
                        target_frames
                    };
                    chunk_controller.observe(ready_frames, decode_elapsed, next_chunk_frames);
                    if emitted_frames > 0 {
                        chunk_controller.adapt_up_for_realtime(
                            config,
                            ready_frames,
                            partial_sample_rate,
                            decode_elapsed,
                            emitted_frames,
                        );
                        output_samples.extend_from_slice(partial_pcm.samples());
                        output_frames = output_frames.saturating_add(emitted_frames);
                        output_sample_rate = partial_pcm.sample_rate();
                        output_channels = partial_pcm.channels();
                        emitted_chunks += 1;
                        if first_emit_pending {
                            first_emit_pending = false;
                            first_chunk_seconds = Some(stream_start.elapsed().as_secs_f64());
                            first_chunk_frames = emitted_frames;
                            chunk_controller.promote_to_max_chunk(config);
                        }
                    } else if first_emit_pending {
                        startup_target_frames = next_startup_target_frames(startup_target_frames, startup_cap_frames);
                    }
                }
                Ok(())
            },
        )?;
        let semantic_loop_seconds = semantic_start.elapsed().as_secs_f64();

        if last_decoded_frames < semantic_tokens.frames() {
            let final_decode_start = Instant::now();
            let final_delta_grid = streamed_tokens.to_grid_range(last_decoded_frames, semantic_tokens.frames())?;
            let decoded_delta = self
                .runtime
                .decode_stream_step(&mut audio_stream_state, &final_delta_grid, true)
                .map_err(Error::from)?;
            let final_pcm = self.runtime.decoded_padded_to_pcm_batch(&decoded_delta).map_err(Error::from)?;
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
        let semantic_decode_seconds = (semantic_loop_seconds - audio_decode_seconds - callback_seconds).max(0.0);
        self.runtime.end_decode_stream(audio_stream_state).map_err(Error::from)?;

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
        });

        Ok(full_pcm)
    }

    fn from_model_metadata(
        model_path: PathBuf,
        model_metadata: ModelMetadata,
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
        let runtime = tts_model_config.create_audio_codec_runtime_with_model_path(&model_path)?;
        let text_decoder = parse_text_decoder_runtime(&tts_model_config, &runtime, &model_path)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            runtime,
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

        let mut result = template
            .render(context!(
                messages => messages
            ))
            .map_err(|_| Error::UnableToRenderPromptTemplate)?;

        if self.drop_initial_newline && result.starts_with('\n') {
            result.remove(0);
        }

        Ok(result)
    }

    fn generate_semantic_tokens(
        &self,
        text_tokens: &[u64],
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        let max_semantic_frames = max_semantic_frames.max(1);
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality =
            usize::try_from(self.runtime.config().codec_cardinality()).map_err(|_| Error::UnableToLoadConfig)?;
        match &mut *decoder {
            TextDecoderRuntime::Stub(stub) => {
                let frames = text_tokens.len().min(max_semantic_frames);
                if stub.num_codebooks != self.runtime.config().num_groups() {
                    return Err(Error::UnableToLoadConfig);
                }

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
            },
            TextDecoderRuntime::FishAudio(runtime) => {
                runtime.generate_semantic_tokens(text_tokens, codec_cardinality, seed, max_semantic_frames)
            },
        }
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
        let max_semantic_frames = max_semantic_frames.max(1);
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality =
            usize::try_from(self.runtime.config().codec_cardinality()).map_err(|_| Error::UnableToLoadConfig)?;
        match &mut *decoder {
            TextDecoderRuntime::Stub(stub) => {
                let frames = text_tokens.len().min(max_semantic_frames);
                if stub.num_codebooks != self.runtime.config().num_groups() {
                    return Err(Error::UnableToLoadConfig);
                }

                let token_upper_bound = stub.codebook_size.min(codec_cardinality);
                if token_upper_bound == 0 {
                    return Err(Error::UnableToLoadConfig);
                }

                let tokens = generate_stub_tokens(stub.num_codebooks, frames, token_upper_bound, seed);
                let grid = AudioTokenGrid::new(
                    tokens.into_boxed_slice(),
                    1,
                    stub.num_codebooks,
                    frames,
                    vec![frames].into_boxed_slice(),
                    AudioTokenPacking::CodebookMajor,
                )
                .map_err(Error::from)?;

                for frame in 0..grid.frames() {
                    let mut frame_codes = Vec::with_capacity(stub.num_codebooks);
                    for codebook in 0..stub.num_codebooks {
                        frame_codes.push(grid.get(0, codebook, frame));
                    }
                    on_frame(&frame_codes)?;
                }
                Ok(grid)
            },
            TextDecoderRuntime::FishAudio(runtime) => runtime.generate_semantic_tokens_with_callback(
                text_tokens,
                codec_cardinality,
                seed,
                max_semantic_frames,
                on_frame,
            ),
        }
    }

    fn take_text_decoder_instrumentation(&self) -> RunnerInstrumentation {
        let mut decoder = self.text_decoder.borrow_mut();
        match &mut *decoder {
            TextDecoderRuntime::Stub(_) => RunnerInstrumentation::default(),
            TextDecoderRuntime::FishAudio(runtime) => runtime.take_instrumentation(),
        }
    }

    fn record_last_execution_stats(
        &self,
        stats: TtsExecutionStats,
    ) {
        self.last_execution_stats.borrow_mut().replace(stats);
    }
}

fn parse_text_decoder_runtime(
    tts_model_config: &TtsModelConfig,
    runtime: &crate::audio::NanoCodecFsqRuntime,
    model_path: &Path,
) -> Result<TextDecoderRuntime, Error> {
    let parsed: TtsConfigJson =
        serde_json::from_value(tts_model_config.tts_config.clone()).map_err(|_| Error::UnableToLoadConfig)?;

    match parsed.text_decoder_config {
        TextDecoderConfigJson::StubTextDecoderConfig {
            num_codebooks,
            codebook_size,
        } => {
            if num_codebooks == 0 || codebook_size == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            if num_codebooks != runtime.config().num_groups() {
                return Err(Error::UnableToLoadConfig);
            }
            let default_seed = load_stub_seed(model_path.join("model.safetensors")).unwrap_or(DEFAULT_STUB_SEED);
            Ok(TextDecoderRuntime::Stub(StubTextDecoderRuntime {
                num_codebooks,
                codebook_size,
                default_seed,
            }))
        },
        TextDecoderConfigJson::FishAudioTextDecoderConfig {
            config,
        } => {
            if config.num_codebooks == 0 || config.codebook_size == 0 || config.max_seq_len == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            if config.num_codebooks != runtime.config().num_groups() {
                return Err(Error::UnableToLoadConfig);
            }
            if config.semantic_token_begin_id > config.semantic_token_end_id {
                return Err(Error::UnableToLoadConfig);
            }
            let semantic_cardinality =
                usize::try_from(config.semantic_token_end_id - config.semantic_token_begin_id + 1)
                    .map_err(|_| Error::UnableToLoadConfig)?;
            if semantic_cardinality == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            if !config.slow_readout_config.is_full_precision() {
                return Err(Error::UnableToLoadConfig);
            }
            if !config.fast_readout_config.is_full_precision() {
                return Err(Error::UnableToLoadConfig);
            }
            if config.slow_model_config.model_dim != config.slow_model_dim {
                return Err(Error::UnableToLoadConfig);
            }
            if config.fast_model_config.model_dim != config.fast_model_dim {
                return Err(Error::UnableToLoadConfig);
            }
            if config.short_logits_size == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            let semantic_sampling_mask_row = build_semantic_sampling_mask_row(
                config.vocab_size,
                config.semantic_token_begin_id,
                config.semantic_token_end_id,
                config.im_end_token_id,
            )?;
            let mut semantic_sampling_mask_without_im_end_row = semantic_sampling_mask_row.to_vec();
            clear_token_in_sampling_mask(&mut semantic_sampling_mask_without_im_end_row, config.im_end_token_id)?;
            let min_frames_before_im_end = fishaudio_min_frames_before_im_end();

            let mut slow_transformer_config = config.slow_model_config.clone();
            let mut fast_transformer_config = config.fast_model_config.clone();
            // Keep f16 activation/storage, but enforce stable normalization accumulation.
            promote_transformer_norm_accumulation_to_f32(&mut slow_transformer_config);
            promote_transformer_norm_accumulation_to_f32(&mut fast_transformer_config);

            let slow_inner_config = InnerModelConfig {
                embedding_config: config.slow_embeddings_config.to_embedding_config(),
                transformer_config: slow_transformer_config,
                vocab_size: config.vocab_size,
            };
            let fast_inner_config = InnerModelConfig {
                embedding_config: config.fast_embeddings_config.to_embedding_config(),
                transformer_config: fast_transformer_config,
                vocab_size: config.codebook_size,
            };

            let slow_decoder_config =
                Rc::new(slow_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
            let fast_decoder_config =
                Rc::new(fast_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
            let text_decoder_context = MetalContext::new().map_err(|_| Error::UnableToCreateBackendContext)?;

            let slow_runner = TokenDecoderRunner::new_with_context(
                text_decoder_context.clone(),
                model_path,
                slow_decoder_config,
                "text_decoder.transformer_slow",
                "text_decoder.embeddings_slow",
                "text_decoder.readout_slow",
            )?;

            let fast_runner = if config.num_codebooks > 1 {
                Some(TokenDecoderRunner::new_with_context(
                    text_decoder_context.clone(),
                    model_path,
                    fast_decoder_config,
                    "text_decoder.transformer_fast",
                    "text_decoder.embeddings_fast",
                    "text_decoder.readout_fast",
                )?)
            } else {
                None
            };

            let weights_path = model_path.join("model.safetensors");
            let codebook_embeddings = load_matrix_f32(
                &weights_path,
                "text_decoder.codebook_embeddings.weights",
                config.codebook_size.checked_mul(config.num_codebooks).ok_or(Error::UnableToLoadConfig)?,
                config.slow_model_dim,
            )?;

            let fast_model_projection = if config.fast_model_projection_config.is_some() {
                Some(load_matrix_f32(
                    &weights_path,
                    "text_decoder.fast_model_projection.weights",
                    config.fast_model_dim,
                    config.slow_model_dim,
                )?)
            } else {
                None
            };

            let activation_data_type = slow_runner.single_hidden_capture.borrow().data_type();
            if let Some(fast_runner_ref) = fast_runner.as_ref() {
                let fast_data_type = fast_runner_ref.single_override_embedding.borrow().data_type();
                if fast_data_type != activation_data_type {
                    return Err(Error::UnableToLoadConfig);
                }
            }
            let apply_semantic_sampling_mask =
                fishaudio_force_semantic_sampling_mask().unwrap_or(!matches!(activation_data_type, DataType::F32));

            let gpu_path = FishAudioGpuPath::new(
                text_decoder_context.as_ref(),
                activation_data_type,
                &codebook_embeddings,
                &fast_model_projection,
                config.num_codebooks,
                config.codebook_size,
                config.slow_model_dim,
                config.fast_model_dim,
            )?;

            Ok(TextDecoderRuntime::FishAudio(FishAudioTextDecoderRuntime {
                slow_runner,
                fast_runner,
                gpu_path,
                semantic_token_begin_id: config.semantic_token_begin_id,
                semantic_token_end_id: config.semantic_token_end_id,
                semantic_cardinality,
                im_end_token_id: config.im_end_token_id,
                codebook_size: config.codebook_size,
                num_codebooks: config.num_codebooks,
                slow_model_dim: config.slow_model_dim,
                fast_model_dim: config.fast_model_dim,
                max_seq_len: config.max_seq_len,
                scale_codebook_embeddings: config.scale_codebook_embeddings,
                fast_vocab_limit: config.short_logits_size.min(config.codebook_size),
                apply_semantic_sampling_mask,
                semantic_sampling_mask_row,
                semantic_sampling_mask_without_im_end_row: semantic_sampling_mask_without_im_end_row.into_boxed_slice(),
                min_frames_before_im_end,
                current_codes_scratch: vec![0_u32; config.num_codebooks],
                instrumentation: RunnerInstrumentation::default(),
            }))
        },
    }
}

impl FishAudioTextDecoderRuntime {
    fn generate_semantic_tokens(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_internal(text_tokens, codec_cardinality, seed, max_semantic_frames, None)
    }

    fn generate_semantic_tokens_with_callback<F>(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        on_frame: &mut F,
    ) -> Result<AudioTokenGrid, Error>
    where
        F: FnMut(&[u32]) -> Result<(), Error>,
    {
        self.generate_semantic_tokens_internal(
            text_tokens,
            codec_cardinality,
            seed,
            max_semantic_frames,
            Some(on_frame),
        )
    }

    fn generate_semantic_tokens_internal(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        max_semantic_frames: usize,
        mut on_frame: Option<&mut dyn FnMut(&[u32]) -> Result<(), Error>>,
    ) -> Result<AudioTokenGrid, Error> {
        if text_tokens.is_empty() {
            return AudioTokenGrid::new(
                Vec::new().into_boxed_slice(),
                1,
                self.num_codebooks,
                0,
                vec![0].into_boxed_slice(),
                AudioTokenPacking::CodebookMajor,
            )
            .map_err(Error::from);
        }
        if text_tokens.len() >= self.max_seq_len {
            return Err(Error::GenerateFailed);
        }

        let semantic_token_upper_bound = self.semantic_cardinality;
        let residual_token_upper_bound = self.codebook_size.min(codec_cardinality);
        if semantic_token_upper_bound == 0 || residual_token_upper_bound == 0 {
            return Err(Error::UnableToLoadConfig);
        }

        self.instrumentation = RunnerInstrumentation::default();
        self.slow_runner.reset();
        self.slow_runner.clear_instrumentation();
        if let Some(fast_runner) = self.fast_runner.as_mut() {
            fast_runner.clear_instrumentation();
        }
        let mut sampling = FishAudioSamplingState::new(seed);
        let semantic_sampling_mask_row = if self.apply_semantic_sampling_mask {
            Some(self.semantic_sampling_mask_row.clone())
        } else {
            None
        };
        let semantic_sampling_mask_without_im_end_row = if self.apply_semantic_sampling_mask {
            Some(self.semantic_sampling_mask_without_im_end_row.clone())
        } else {
            None
        };
        let initial_sampling_row = if let Some(mask_row) = semantic_sampling_mask_row.as_ref() {
            if self.min_frames_before_im_end > 0 {
                semantic_sampling_mask_without_im_end_row.as_deref().ok_or(Error::GenerateFailed)?
            } else {
                mask_row.as_ref()
            }
        } else {
            &[]
        };
        let prefill_step_size = fishaudio_prefill_step_size(self.max_seq_len);
        if text_tokens.len() > prefill_step_size {
            for chunk in text_tokens[..text_tokens.len() - prefill_step_size].chunks(prefill_step_size) {
                self.slow_runner.prefill_without_sampling(chunk)?;
            }
        }
        let prefill_tail_start = text_tokens.len().saturating_sub(prefill_step_size);
        let prefill_tail = &text_tokens[prefill_tail_start..];
        let prefill_mask = if initial_sampling_row.is_empty() {
            None
        } else {
            Some(expand_token_mask_for_sampling_row(initial_sampling_row, prefill_tail.len())?)
        };
        let mut current_semantic_token = self.slow_runner.decode_next_token_with_hidden_capture(
            prefill_tail,
            EmbeddingInjection::None,
            &mut sampling,
            prefill_mask.as_deref(),
        )?;
        let post_scale = if self.scale_codebook_embeddings {
            Some(1.0 / ((self.num_codebooks + 1) as f32).sqrt())
        } else {
            None
        };

        let mut max_new_tokens = self.max_seq_len.saturating_sub(text_tokens.len());
        max_new_tokens = max_new_tokens.min(max_semantic_frames.max(1));
        if let Some(limit) = fishaudio_max_new_tokens_override() {
            max_new_tokens = max_new_tokens.min(limit);
        }
        let mut by_codebook =
            (0..self.num_codebooks).map(|_| Vec::<u32>::with_capacity(max_new_tokens)).collect::<Vec<_>>();
        if self.current_codes_scratch.len() != self.num_codebooks {
            self.current_codes_scratch = vec![0_u32; self.num_codebooks];
        }

        if self.num_codebooks > 1 {
            let fast_vocab_limit = self.fast_vocab_limit.min(residual_token_upper_bound);
            if fast_vocab_limit == 0 {
                return Err(Error::UnableToLoadConfig);
            }
            if let Some(fast_runner) = self.fast_runner.as_mut() {
                fast_runner.prepare_single_token_vocab_mask(fast_vocab_limit)?;
                fast_runner.prepare_two_token_vocab_mask(fast_vocab_limit)?;
            }
        }

        for _step in 0..max_new_tokens {
            if current_semantic_token as i64 == self.im_end_token_id {
                break;
            }

            let first_code = semantic_token_to_code(
                current_semantic_token,
                self.semantic_token_begin_id,
                self.semantic_token_end_id,
                semantic_token_upper_bound,
            );
            by_codebook[0].push(first_code);
            self.current_codes_scratch[0] = first_code;

            if self.num_codebooks > 1 {
                let (slow_runner, fast_runner_opt, gpu_path) =
                    (&mut self.slow_runner, &mut self.fast_runner, &mut self.gpu_path);
                let slow_hidden_capture = &slow_runner.single_hidden_capture;
                let slow_model_dim = self.slow_model_dim;
                let fast_model_dim = self.fast_model_dim;
                let Some(fast_runner) = fast_runner_opt.as_mut() else {
                    return Err(Error::GenerateFailed);
                };

                fast_runner.reset();
                let fast_vocab_limit = self.fast_vocab_limit.min(residual_token_upper_bound);
                let mut pre_projection =
                    |runner: &TokenDecoderRunner,
                     _state: &ForwardPassState<Metal>,
                     command_buffer: &MetalCommandBuffer| {
                        Self::encode_project_slow_hidden_to_fast_on(
                            runner.context.as_ref(),
                            gpu_path,
                            slow_hidden_capture,
                            &runner.single_override_embedding,
                            slow_model_dim,
                            fast_model_dim,
                            command_buffer,
                        )
                    };
                let mut fast_token = fast_runner
                    .decode_next_token_with_override_prefix_from_internal_and_pre_injection(
                        u64::from(first_code),
                        Some(fast_vocab_limit),
                        &mut sampling,
                        Some(&mut pre_projection),
                    )?;
                if self.num_codebooks > 1 {
                    let clamped =
                        u32::try_from((fast_token as usize).min(residual_token_upper_bound.saturating_sub(1)))
                            .map_err(|_| Error::GenerateFailed)?;
                    by_codebook[1].push(clamped);
                    self.current_codes_scratch[1] = clamped;
                    fast_token = u64::from(clamped);
                }

                let followup_count = self.num_codebooks.saturating_sub(2);
                if followup_count > 0 {
                    fast_runner.decode_followup_tokens_batched(
                        fast_token,
                        followup_count,
                        Some(fast_vocab_limit),
                        &mut sampling,
                        |relative_index, sampled| {
                            let codebook_index = relative_index + 2;
                            let clamped =
                                u32::try_from((sampled as usize).min(residual_token_upper_bound.saturating_sub(1)))
                                    .map_err(|_| Error::GenerateFailed)?;
                            by_codebook[codebook_index].push(clamped);
                            self.current_codes_scratch[codebook_index] = clamped;
                            Ok(())
                        },
                    )?;
                }
            }

            if let Some(callback) = on_frame.as_mut() {
                callback(&self.current_codes_scratch)?;
            }

            let (slow_runner, gpu_path) = (&mut self.slow_runner, &mut self.gpu_path);
            let current_codes = self.current_codes_scratch.as_slice();
            let num_codebooks = self.num_codebooks;
            let codebook_size = self.codebook_size;
            let slow_model_dim = self.slow_model_dim;
            let mut pre_codebook_sum =
                |runner: &TokenDecoderRunner, _state: &ForwardPassState<Metal>, command_buffer: &MetalCommandBuffer| {
                    Self::encode_slow_codebook_sum_from_codes_on(
                        gpu_path,
                        &runner.single_override_embedding,
                        current_codes,
                        num_codebooks,
                        codebook_size,
                        slow_model_dim,
                        command_buffer,
                    )
                };
            let sampled_frames = by_codebook[0].len();
            let slow_sampling_mask = if self.apply_semantic_sampling_mask {
                if sampled_frames < self.min_frames_before_im_end {
                    semantic_sampling_mask_without_im_end_row.as_deref()
                } else {
                    semantic_sampling_mask_row.as_deref()
                }
            } else {
                None
            };
            current_semantic_token = slow_runner.decode_next_token_with_hidden_capture_and_pre_injection(
                &[current_semantic_token],
                EmbeddingInjection::AddPreloaded {
                    post_scale,
                },
                &mut sampling,
                slow_sampling_mask,
                Some(&mut pre_codebook_sum),
            )?;
        }

        self.instrumentation = self.slow_runner.take_instrumentation();
        if let Some(fast_runner) = self.fast_runner.as_mut() {
            let fast = fast_runner.take_instrumentation();
            self.instrumentation.command_buffers_submitted += fast.command_buffers_submitted;
            self.instrumentation.host_waits += fast.host_waits;
        }

        let frames = by_codebook.first().map_or(0, Vec::len);
        let mut tokens = Vec::with_capacity(self.num_codebooks * frames);
        for codebook_tokens in &by_codebook {
            if codebook_tokens.len() != frames {
                return Err(Error::GenerateFailed);
            }
            tokens.extend_from_slice(codebook_tokens);
        }

        AudioTokenGrid::new(
            tokens.into_boxed_slice(),
            1,
            self.num_codebooks,
            frames,
            vec![frames].into_boxed_slice(),
            AudioTokenPacking::CodebookMajor,
        )
        .map_err(Error::from)
    }

    fn encode_project_slow_hidden_to_fast_on(
        context: &MetalContext,
        gpu_path: &mut FishAudioGpuPath,
        slow_hidden_capture: &ArrayCell<Metal>,
        output_embedding: &ArrayCell<Metal>,
        slow_model_dim: usize,
        fast_model_dim: usize,
        command_buffer: &MetalCommandBuffer,
    ) -> Result<(), Error> {
        let model_dim_u32 = u32::try_from(slow_model_dim).map_err(|_| Error::GenerateFailed)?;

        if let Some(weights) = gpu_path.projection_weights.as_ref() {
            let hidden = slow_hidden_capture.borrow();
            let weights = weights.borrow();
            let output = output_embedding.borrow();
            if hidden.shape() != [1, slow_model_dim]
                || output.shape() != [1, fast_model_dim]
                || weights.shape() != [fast_model_dim, slow_model_dim]
            {
                return Err(Error::GenerateFailed);
            }

            command_buffer.with_compute_encoder(|encoder| {
                gpu_path.projection.encode(
                    context,
                    encoder,
                    FullPrecisionMatmulArguments {
                        a: hidden.buffer(),
                        a_offset: hidden.offset(),
                        b: weights.buffer(),
                        output: output.buffer(),
                        bias: None,
                        batch: 1,
                        input_dim: slow_model_dim,
                        output_dim: fast_model_dim,
                    },
                );
            });
            return Ok(());
        }

        if slow_model_dim != fast_model_dim {
            return Err(Error::UnableToLoadConfig);
        }

        let hidden = slow_hidden_capture.borrow();
        let output = output_embedding.borrow();
        if hidden.shape() != [1, slow_model_dim] || output.shape() != [1, fast_model_dim] {
            return Err(Error::GenerateFailed);
        }
        command_buffer.with_compute_encoder(|encoder| {
            gpu_path.tensor_copy.encode(
                (hidden.buffer(), hidden.offset()),
                (output.buffer(), output.offset()),
                model_dim_u32,
                encoder,
            );
        });
        Ok(())
    }

    fn encode_slow_codebook_sum_from_codes_on(
        gpu_path: &mut FishAudioGpuPath,
        slow_sum_embedding: &ArrayCell<Metal>,
        current_codes: &[u32],
        num_codebooks: usize,
        codebook_size: usize,
        slow_model_dim: usize,
        command_buffer: &MetalCommandBuffer,
    ) -> Result<(), Error> {
        if current_codes.len() != num_codebooks {
            return Err(Error::GenerateFailed);
        }
        if num_codebooks == 0 {
            return Err(Error::UnableToLoadConfig);
        }

        {
            let mut row_indices = gpu_path.codebook_row_indices.borrow_mut();
            if row_indices.shape() != [num_codebooks] || row_indices.data_type() != DataType::U64 {
                return Err(Error::GenerateFailed);
            }
            let indices_slice = row_indices.as_slice_mut::<u64>();
            for (codebook_index, &token) in current_codes.iter().enumerate() {
                let token = usize::try_from(token).map_err(|_| Error::GenerateFailed)?;
                if token >= codebook_size {
                    return Err(Error::GenerateFailed);
                }
                let row = codebook_index
                    .checked_mul(codebook_size)
                    .and_then(|offset| offset.checked_add(token))
                    .ok_or(Error::GenerateFailed)?;
                indices_slice[codebook_index] = u64::try_from(row).map_err(|_| Error::GenerateFailed)?;
            }
        }

        let total_vocab = num_codebooks.checked_mul(codebook_size).ok_or(Error::GenerateFailed)?;
        let total_vocab_u32 = u32::try_from(total_vocab).map_err(|_| Error::GenerateFailed)?;
        let num_codebooks_u32 = u32::try_from(num_codebooks).map_err(|_| Error::GenerateFailed)?;
        let slow_model_dim_u32 = u32::try_from(slow_model_dim).map_err(|_| Error::GenerateFailed)?;

        let codebook_row_indices = gpu_path.codebook_row_indices.borrow();
        let codebook_embeddings = gpu_path.codebook_embeddings.borrow();
        let slow_sum = slow_sum_embedding.borrow();
        if codebook_row_indices.shape() != [num_codebooks]
            || codebook_embeddings.shape() != [total_vocab, slow_model_dim]
            || slow_sum.shape() != [1, slow_model_dim]
            || codebook_embeddings.data_type() != slow_sum.data_type()
        {
            return Err(Error::GenerateFailed);
        }

        command_buffer.with_compute_encoder(|encoder| {
            gpu_path.embedding_rows_sum.encode(
                (codebook_row_indices.buffer(), codebook_row_indices.offset()),
                (codebook_embeddings.buffer(), codebook_embeddings.offset()),
                (slow_sum.buffer(), slow_sum.offset()),
                num_codebooks_u32,
                total_vocab_u32,
                slow_model_dim_u32,
                encoder,
            );
        });
        Ok(())
    }

    fn take_instrumentation(&mut self) -> RunnerInstrumentation {
        std::mem::take(&mut self.instrumentation)
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
    ) -> Result<Self, Error> {
        let command_buffer =
            Rc::new(RefCell::new(context.create_command_buffer().expect("Failed to create command buffer")));

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = fishaudio_prefill_step_size(decoder_config.context_length).max(32);
        let should_fill_attention_bias =
            model_shape.sliding_window_length_per_layer.iter().any(|value| value.is_some());

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
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let activation_data_type = model_shape.activation_data_type();
        let tensor_copy =
            <<Metal as Backend>::Kernels as Kernels>::TensorCopyKernel::new(context.as_ref(), activation_data_type)
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let tensor_add_scale =
            <<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(context.as_ref(), activation_data_type)
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let token_copy_sampled =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopySampledKernel::new(context.as_ref())
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let token_copy_results =
            <<Metal as Backend>::Kernels as Kernels>::TokenCopyToResultsKernel::new(context.as_ref())
                .map_err(|_| Error::UnableToCreateBackendContext)?;
        let async_chain_capacity = max_suffix_length.max(1);
        let async_chain_positions = Rc::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<i32>())
                .map_err(|_| Error::UnableToCreateBackendContext)?,
        );
        let async_chain_seeds = Rc::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u64>())
                .map_err(|_| Error::UnableToCreateBackendContext)?,
        );
        let async_chain_results = Rc::new(
            context
                .create_buffer(async_chain_capacity * std::mem::size_of::<u32>())
                .map_err(|_| Error::UnableToCreateBackendContext)?,
        );
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
            .map_err(|_| Error::UnableToCreateBackendContext)?;

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
            kv_cache_update,
            tensor_copy,
            tensor_add_scale,
            token_copy_sampled,
            token_copy_results,
            async_chain_positions,
            async_chain_seeds,
            async_chain_results,
            async_chain_capacity,
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
    }

    fn prefill_without_sampling(
        &mut self,
        token_ids: &[u64],
    ) -> Result<(), Error> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut sampling = FishAudioSamplingState::with_params(0, 0.0, 1.0);
        let _ = self.decode_next_step(token_ids, EmbeddingInjection::None, None, &mut sampling, None, false, None)?;
        Ok(())
    }

    fn decode_next_token_with_hidden_capture(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut FishAudioSamplingState,
        precomputed_token_bitmask: Option<&[u32]>,
    ) -> Result<u64, Error> {
        self.decode_next_step(token_ids, embedding_injection, None, sampling, precomputed_token_bitmask, true, None)
    }

    fn decode_next_token_with_hidden_capture_and_pre_injection(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut FishAudioSamplingState,
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

    fn decode_next_token_with_override_prefix_from_internal_and_pre_injection(
        &mut self,
        first_token: u64,
        vocab_limit: Option<usize>,
        sampling: &mut FishAudioSamplingState,
        pre_injection_encode: Option<&mut PreInjectionEncodeCallback<'_>>,
    ) -> Result<u64, Error> {
        self.decode_next_step(
            &[0, first_token],
            EmbeddingInjection::OverrideFirstRowInternal,
            vocab_limit,
            sampling,
            None,
            false,
            pre_injection_encode,
        )
    }

    fn decode_followup_tokens_batched(
        &mut self,
        first_token: u64,
        followup_count: usize,
        vocab_limit: Option<usize>,
        sampling: &mut FishAudioSamplingState,
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

        {
            let positions_ptr = self.async_chain_positions.cpu_ptr().as_ptr() as *mut i32;
            for pass in 0..followup_count {
                unsafe {
                    *positions_ptr.add(pass) = (self.next_position + pass) as i32;
                }
            }
        }

        {
            let seeds_ptr = self.async_chain_seeds.cpu_ptr().as_ptr() as *mut u64;
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

            let encoding_parameters = EncodingParameters::new(false, false, false);
            self.executables.embed.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            for layer in self.executables.layers.iter() {
                layer.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            }
            self.executables.norm.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            self.executables.readout.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            self.sampler.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());

            let sampling_output = state.sampling_output().ok_or(Error::GenerateFailed)?;
            let sampling_output_binding = sampling_output.borrow();
            let sampling_output_buffer = sampling_output_binding.buffer();
            let token_ids_binding = self.scratch_buffers.token_ids.borrow();
            let token_ids_buffer = token_ids_binding.buffer();

            self.command_buffer.borrow().with_compute_encoder(|encoder| {
                if pass + 1 < followup_count {
                    self.token_copy_sampled.encode(sampling_output_buffer, token_ids_buffer, encoder);
                }
                let results_offset = pass * std::mem::size_of::<u32>();
                self.token_copy_results.encode(
                    sampling_output_buffer,
                    (self.async_chain_results.as_ref(), results_offset),
                    encoder,
                );
            });

            self.cache_layers.borrow_mut().update_after_acceptance(
                &[0],
                None,
                self.command_buffer.borrow().deref(),
                &self.kv_cache_update,
            );
            self.cache_layers.borrow_mut().register_accepted_tokens(&[self.next_position + pass]);
        }

        self.next_position = self.next_position.saturating_add(followup_count);
        self.submit_and_wait_current_command_buffer();

        let results_ptr = self.async_chain_results.cpu_ptr().as_ptr() as *const u32;
        for pass in 0..followup_count {
            let sampled = unsafe { *results_ptr.add(pass) };
            on_token(pass, u64::from(sampled))?;
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
        sampling: &mut FishAudioSamplingState,
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

            let encoding_parameters = EncodingParameters::new(false, false, false);
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
            self.executables.embed.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            if let Some(pre_encode) = pre_injection_encode.as_mut() {
                pre_encode(self, &state, self.command_buffer.borrow().deref())?;
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
                layer.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            }
            if capture_hidden {
                self.encode_capture_last_hidden_into_single_buffer(&state, token_count)?;
            }
            self.executables.norm.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            self.executables.readout.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            self.sampler.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
            self.cache_layers.borrow_mut().update_after_acceptance(
                accepted_suffix_indices,
                None,
                self.command_buffer.borrow().deref(),
                &self.kv_cache_update,
            );
            self.submit_and_wait_current_command_buffer();
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

        self.command_buffer.borrow().with_compute_encoder(|encoder| {
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

        self.command_buffer.borrow().with_compute_encoder(|encoder| {
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

        self.command_buffer.borrow().with_compute_encoder(|encoder| {
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

    fn submit_and_wait_current_command_buffer(&mut self) {
        self.command_buffer.borrow().submit();
        self.instrumentation.command_buffers_submitted += 1;
        self.command_buffer.borrow().wait_until_completed();
        self.instrumentation.host_waits += 1;
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

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveChunkController, DEFAULT_CHUNK_EMA_ALPHA, DEFAULT_CHUNK_HYSTERESIS_FRACTION,
        DEFAULT_FISHAUDIO_MIN_FRAMES_BEFORE_IM_END, DEFAULT_FISHAUDIO_RANDOM_SEED,
        DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE, DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE, DEFAULT_FISHAUDIO_SAMPLING_TOP_P,
        DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE, DEFAULT_STUB_SEED, FishAudioSamplingState, MatrixF32,
        StreamingTokenAccumulator, build_semantic_sampling_mask_row, clear_token_in_sampling_mask,
        default_repeat_window_size, default_short_logits_size, expand_token_mask_for_sampling_row,
        fishaudio_min_frames_before_im_end, generate_stub_tokens, load_stub_seed, semantic_token_to_code,
    };
    use crate::audio::AudioTokenPacking;
    use crate::session::config::{TtsChunkPolicy, TtsRunConfig};
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
    fn expanded_sampling_mask_targets_only_sampling_row() {
        let row_mask = vec![0xFFFF_0000u32, 0x0000_FFFFu32];
        let expanded = expand_token_mask_for_sampling_row(&row_mask, 3).expect("expanded");
        assert_eq!(expanded.len(), 6);
        assert_eq!(&expanded[0..2], &[u32::MAX, u32::MAX]);
        assert_eq!(&expanded[2..4], &[u32::MAX, u32::MAX]);
        assert_eq!(&expanded[4..6], row_mask.as_slice());
    }

    #[test]
    fn fishaudio_decoder_defaults_match_lalamo_consts() {
        assert_eq!(default_short_logits_size(), DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE);
        assert_eq!(default_repeat_window_size(), DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE);
        assert_eq!(fishaudio_min_frames_before_im_end(), DEFAULT_FISHAUDIO_MIN_FRAMES_BEFORE_IM_END);
        assert_eq!(DEFAULT_FISHAUDIO_RANDOM_SEED, 123);
        assert_eq!(DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE, 0.8008);
        assert_eq!(DEFAULT_FISHAUDIO_SAMPLING_TOP_P, 0.8008);
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
        let mut a = FishAudioSamplingState::new(123);
        let mut b = FishAudioSamplingState::new(123);
        let seeds_a = (0..8).map(|_| a.next_seed()).collect::<Vec<_>>();
        let seeds_b = (0..8).map(|_| b.next_seed()).collect::<Vec<_>>();
        assert_eq!(seeds_a, seeds_b);
    }

    #[test]
    fn fishaudio_sampling_top_p_zero_switches_to_greedy() {
        let sampler = FishAudioSamplingState::with_params(999, 0.8, 0.0);
        assert_eq!(sampler.method(), SamplingMethod::Greedy);
    }

    #[test]
    fn fishaudio_sampling_positive_top_p_uses_stochastic_mode() {
        let sampler = FishAudioSamplingState::with_params(999, 0.8, 0.8);
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
