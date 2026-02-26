#![cfg(all(feature = "audio-runtime", feature = "metal", target_os = "macos"))]

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    ops::Deref,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
};

use half::{bf16, f16};
use minijinja::{Environment, context};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    DataType,
    audio::{AudioCodecRuntime, AudioPcmBatch, AudioTokenGrid, AudioTokenPacking},
    backends::{
        common::{Backend, CommandBuffer, Context as BackendContext, kernel::kv_cache_update::KVCacheUpdate},
        metal::Metal,
    },
    config::{InnerModelConfig, LinearConfig, ModelMetadata, ModelType, TtsModelConfig},
    encodable_block::{Decoder, EncodableBlock, EncodingParameters},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, SharedBuffers},
    },
    parameters::{ParameterLoader, read_safetensors_metadata},
    session::types::{Error, Input},
};

const DEFAULT_STUB_SPEAKER_ID: &str = "speaker:0";
const DEFAULT_STUB_STYLE: &str = "interleave";
const DEFAULT_STUB_SEED: u64 = 123;
const DEFAULT_FISHAUDIO_RANDOM_SEED: u64 = 123;
const DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE: usize = 1024;
const DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE: usize = 16;
const DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE: f32 = 0.8008;
const DEFAULT_FISHAUDIO_SAMPLING_TOP_P: f32 = 0.8008;

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
    codebook_embeddings: MatrixF32,
    fast_model_projection: Option<MatrixF32>,
    semantic_token_begin_id: i64,
    semantic_token_end_id: i64,
    semantic_cardinality: usize,
    im_end_token_id: i64,
    codebook_size: usize,
    num_codebooks: usize,
    slow_model_dim: usize,
    fast_model_dim: usize,
    max_seq_len: usize,
    short_logits_size: usize,
    scale_codebook_embeddings: bool,
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
    kv_cache_update: KVCacheUpdate<Metal>,
    next_position: usize,
}

struct MatrixF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

enum EmbeddingInjection {
    None,
    Add {
        values: Vec<f32>,
        post_scale: Option<f32>,
    },
    Override(Vec<f32>),
}

struct DecodeStep {
    token: u64,
    last_hidden: Option<Vec<f32>>,
}

struct FishAudioSamplingState {
    rng: StdRng,
    temperature: f32,
    top_p: f32,
    step_key: Option<u64>,
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
        Self {
            rng: StdRng::seed_from_u64(seed),
            temperature,
            top_p,
            step_key: None,
        }
    }

    fn begin_step(&mut self) {
        self.step_key = Some(self.rng.random::<u64>());
    }

    fn sample_index(
        &mut self,
        logits: &[f32],
    ) -> Result<usize, Error> {
        if logits.is_empty() {
            return Err(Error::GenerateFailed);
        }

        if self.temperature <= 0.0 || self.top_p <= 0.0 {
            return Ok(argmax_index(logits));
        }

        let mut sorted = logits.iter().copied().enumerate().collect::<Vec<_>>();
        sorted.sort_by(|left, right| right.1.total_cmp(&left.1));

        let max_logit = sorted[0].1;
        let mut probs = Vec::with_capacity(sorted.len());
        let mut sum = 0.0_f32;
        for (_, value) in &sorted {
            let probability = ((*value / self.temperature) - (max_logit / self.temperature)).exp();
            probs.push(probability);
            sum += probability;
        }

        if !sum.is_finite() || sum <= 0.0 {
            return Ok(sorted[0].0);
        }

        for probability in &mut probs {
            *probability /= sum;
        }

        let mut keep = probs.len();
        if self.top_p < 1.0 {
            let mut cumulative = 0.0_f32;
            keep = 0;
            for probability in &probs {
                cumulative += *probability;
                keep += 1;
                if cumulative > self.top_p {
                    break;
                }
            }
            keep = keep.max(1);
        }

        let retained = &probs[..keep];
        let retained_sum: f32 = retained.iter().sum();
        if retained_sum <= 0.0 || !retained_sum.is_finite() {
            return Ok(sorted[0].0);
        }

        let step_key = self.step_key.unwrap_or_else(|| self.rng.random::<u64>());
        let mut best_index = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for index in 0..keep {
            let probability = (retained[index] / retained_sum).max(f32::MIN_POSITIVE);
            let log_prob = probability.ln();
            let token_index = sorted[index].0 as u64;
            let gumbel = gumbel_from_step_key(step_key, token_index);
            let score = log_prob + gumbel;
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }

        Ok(sorted[best_index].0)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01_from_u64(value: u64) -> f32 {
    let mantissa = ((value >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
    mantissa.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON) as f32
}

fn gumbel_from_step_key(
    step_key: u64,
    token_index: u64,
) -> f32 {
    let mixed = splitmix64(step_key ^ token_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let u = uniform01_from_u64(mixed);
    -(-u.ln()).ln()
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
        let semantic_tokens = self.generate_semantic_tokens_with_seed(input, seed)?;
        self.runtime.decode(&semantic_tokens).map_err(Error::from)
    }

    pub fn generate_semantic_tokens_with_seed(
        &self,
        input: Input,
        seed: u64,
    ) -> Result<AudioTokenGrid, Error> {
        let prompt = self.render_prompt(&input)?;
        let text_tokens: Vec<u64> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&token| token as u64)
            .collect();

        self.generate_semantic_tokens(&text_tokens, seed)
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
        mut on_chunk: F,
    ) -> Result<AudioPcmBatch, Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        if chunk_frames == 0 {
            return Err(Error::GenerateFailed);
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
        let mut emitted_samples = 0usize;
        let semantic_tokens = self.generate_semantic_tokens_with_callback(&text_tokens, seed, &mut |codes| {
            streamed_tokens.push_frame(codes)?;
            if streamed_tokens.frames() % chunk_frames == 0 {
                let partial_grid = streamed_tokens.to_grid()?;
                let partial_pcm = self.runtime.decode(&partial_grid).map_err(Error::from)?;
                Self::emit_streaming_chunk(&partial_pcm, &mut emitted_samples, &mut on_chunk)?;
            }
            Ok(())
        })?;

        let full_pcm = self.runtime.decode(&semantic_tokens).map_err(Error::from)?;
        Self::emit_streaming_chunk(&full_pcm, &mut emitted_samples, &mut on_chunk)?;
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
    ) -> Result<AudioTokenGrid, Error> {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality =
            usize::try_from(self.runtime.config().codec_cardinality()).map_err(|_| Error::UnableToLoadConfig)?;
        match &mut *decoder {
            TextDecoderRuntime::Stub(stub) => {
                let frames = text_tokens.len();
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
                runtime.generate_semantic_tokens(text_tokens, codec_cardinality, seed)
            },
        }
    }

    fn generate_semantic_tokens_with_callback<F>(
        &self,
        text_tokens: &[u64],
        seed: u64,
        on_frame: &mut F,
    ) -> Result<AudioTokenGrid, Error>
    where
        F: FnMut(&[u32]) -> Result<(), Error>,
    {
        let mut decoder = self.text_decoder.borrow_mut();
        let codec_cardinality =
            usize::try_from(self.runtime.config().codec_cardinality()).map_err(|_| Error::UnableToLoadConfig)?;
        match &mut *decoder {
            TextDecoderRuntime::Stub(stub) => {
                let frames = text_tokens.len();
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
            TextDecoderRuntime::FishAudio(runtime) => {
                runtime.generate_semantic_tokens_with_callback(text_tokens, codec_cardinality, seed, on_frame)
            },
        }
    }

    fn emit_streaming_chunk<F>(
        pcm: &AudioPcmBatch,
        emitted_samples: &mut usize,
        on_chunk: &mut F,
    ) -> Result<(), Error>
    where
        F: FnMut(&AudioPcmBatch),
    {
        if pcm.batch_size() != 1 {
            return Err(Error::GenerateFailed);
        }

        let samples = pcm.samples();
        if *emitted_samples > samples.len() {
            return Err(Error::GenerateFailed);
        }

        let delta = &samples[*emitted_samples..];
        if delta.is_empty() {
            return Ok(());
        }

        let channels = pcm.channels();
        if delta.len() % channels != 0 {
            return Err(Error::GenerateFailed);
        }
        let frames = delta.len() / channels;
        let chunk = AudioPcmBatch::new(
            delta.to_vec().into_boxed_slice(),
            pcm.sample_rate(),
            channels,
            vec![frames].into_boxed_slice(),
        )
        .map_err(Error::from)?;

        on_chunk(&chunk);
        *emitted_samples = samples.len();
        Ok(())
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

            let slow_inner_config = InnerModelConfig {
                embedding_config: config.slow_embeddings_config.to_embedding_config(),
                transformer_config: config.slow_model_config.clone(),
                vocab_size: config.vocab_size,
            };
            let fast_inner_config = InnerModelConfig {
                embedding_config: config.fast_embeddings_config.to_embedding_config(),
                transformer_config: config.fast_model_config.clone(),
                vocab_size: config.codebook_size,
            };

            let slow_decoder_config =
                Rc::new(slow_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);
            let fast_decoder_config =
                Rc::new(fast_inner_config.to_decoder_config().map_err(|_| Error::UnableToLoadConfig)?);

            let slow_runner = TokenDecoderRunner::new(
                model_path,
                slow_decoder_config,
                "text_decoder.transformer_slow",
                "text_decoder.embeddings_slow",
                "text_decoder.readout_slow",
            )?;

            let fast_runner = if config.num_codebooks > 1 {
                Some(TokenDecoderRunner::new(
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

            Ok(TextDecoderRuntime::FishAudio(FishAudioTextDecoderRuntime {
                slow_runner,
                fast_runner,
                codebook_embeddings,
                fast_model_projection,
                semantic_token_begin_id: config.semantic_token_begin_id,
                semantic_token_end_id: config.semantic_token_end_id,
                semantic_cardinality,
                im_end_token_id: config.im_end_token_id,
                codebook_size: config.codebook_size,
                num_codebooks: config.num_codebooks,
                slow_model_dim: config.slow_model_dim,
                fast_model_dim: config.fast_model_dim,
                max_seq_len: config.max_seq_len,
                short_logits_size: config.short_logits_size,
                scale_codebook_embeddings: config.scale_codebook_embeddings,
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
    ) -> Result<AudioTokenGrid, Error> {
        self.generate_semantic_tokens_internal(text_tokens, codec_cardinality, seed, None)
    }

    fn generate_semantic_tokens_with_callback<F>(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
        on_frame: &mut F,
    ) -> Result<AudioTokenGrid, Error>
    where
        F: FnMut(&[u32]) -> Result<(), Error>,
    {
        self.generate_semantic_tokens_internal(text_tokens, codec_cardinality, seed, Some(on_frame))
    }

    fn generate_semantic_tokens_internal(
        &mut self,
        text_tokens: &[u64],
        codec_cardinality: usize,
        seed: u64,
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

        self.slow_runner.reset();
        let mut sampling = FishAudioSamplingState::new(seed);
        sampling.begin_step();
        let (mut current_semantic_token, mut current_hidden) =
            self.slow_runner.decode_next_token_with_hidden(text_tokens, EmbeddingInjection::None, &mut sampling)?;

        let max_new_tokens = self.max_seq_len.saturating_sub(text_tokens.len());
        let mut by_codebook = vec![Vec::<u32>::new(); self.num_codebooks];

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

            let mut current_codes = Vec::with_capacity(self.num_codebooks);
            current_codes.push(first_code);

            if self.num_codebooks > 1 {
                let projected_hidden = self.project_slow_hidden_to_fast(&current_hidden)?;
                let Some(fast_runner) = self.fast_runner.as_mut() else {
                    return Err(Error::GenerateFailed);
                };

                fast_runner.reset();
                fast_runner.advance_with_override_embedding(&projected_hidden, &mut sampling)?;

                let fast_vocab_limit = self.short_logits_size.min(residual_token_upper_bound);
                if fast_vocab_limit == 0 {
                    return Err(Error::UnableToLoadConfig);
                }

                let mut fast_token = u64::from(first_code);
                for codebook_index in 1..self.num_codebooks {
                    fast_token = fast_runner.decode_next_token_with_vocab_limit(
                        &[fast_token],
                        Some(fast_vocab_limit),
                        &mut sampling,
                    )?;
                    let clamped =
                        u32::try_from((fast_token as usize).min(residual_token_upper_bound.saturating_sub(1)))
                            .map_err(|_| Error::GenerateFailed)?;
                    by_codebook[codebook_index].push(clamped);
                    current_codes.push(clamped);
                    fast_token = u64::from(clamped);
                }
            }

            if let Some(callback) = on_frame.as_mut() {
                callback(&current_codes)?;
            }

            let injection = self.build_slow_embedding_injection(&current_codes)?;
            sampling.begin_step();
            let (next_semantic_token, next_hidden) =
                self.slow_runner.decode_next_token_with_hidden(&[current_semantic_token], injection, &mut sampling)?;
            current_semantic_token = next_semantic_token;
            current_hidden = next_hidden;
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

    fn build_slow_embedding_injection(
        &self,
        codebooks: &[u32],
    ) -> Result<EmbeddingInjection, Error> {
        if codebooks.len() != self.num_codebooks {
            return Err(Error::GenerateFailed);
        }
        let mut sum = vec![0.0_f32; self.slow_model_dim];

        for (codebook_index, &token) in codebooks.iter().enumerate() {
            let token = usize::try_from(token).map_err(|_| Error::GenerateFailed)?;
            if token >= self.codebook_size {
                return Err(Error::GenerateFailed);
            }
            let row = codebook_index
                .checked_mul(self.codebook_size)
                .and_then(|offset| offset.checked_add(token))
                .ok_or(Error::GenerateFailed)?;
            let embedding_row = self.codebook_embeddings.row(row).ok_or(Error::GenerateFailed)?;
            for (dst, &src) in sum.iter_mut().zip(embedding_row.iter()) {
                *dst += src;
            }
        }

        let post_scale = if self.scale_codebook_embeddings {
            Some(1.0 / ((self.num_codebooks + 1) as f32).sqrt())
        } else {
            None
        };

        Ok(EmbeddingInjection::Add {
            values: sum,
            post_scale,
        })
    }

    fn project_slow_hidden_to_fast(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Error> {
        if hidden.len() != self.slow_model_dim {
            return Err(Error::GenerateFailed);
        }

        if let Some(projection) = &self.fast_model_projection {
            if projection.rows != self.fast_model_dim || projection.cols != self.slow_model_dim {
                return Err(Error::UnableToLoadConfig);
            }
            return projection.matmul(hidden).ok_or(Error::GenerateFailed);
        }

        if self.slow_model_dim != self.fast_model_dim {
            return Err(Error::UnableToLoadConfig);
        }

        Ok(hidden.to_vec())
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

impl TokenDecoderRunner {
    fn new(
        model_path: &Path,
        decoder_config: Rc<crate::config::DecoderConfig>,
        transformer_subtree: &str,
        embedding_subtree: &str,
        readout_subtree: &str,
    ) -> Result<Self, Error> {
        let context = MetalContext::new().map_err(|_| Error::UnableToCreateBackendContext)?;
        let command_buffer =
            Rc::new(RefCell::new(context.create_command_buffer().expect("Failed to create command buffer")));

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let max_prefix_length = decoder_config.context_length;
        let max_suffix_length = decoder_config.context_length.max(1);

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
            kv_cache_update,
            next_position: 0,
        })
    }

    fn reset(&mut self) {
        self.cache_layers.borrow_mut().clear();
        self.next_position = 0;
    }

    fn decode_next_token_with_vocab_limit(
        &mut self,
        token_ids: &[u64],
        vocab_limit: Option<usize>,
        sampling: &mut FishAudioSamplingState,
    ) -> Result<u64, Error> {
        let step = self.decode_next_step(token_ids, EmbeddingInjection::None, vocab_limit, false, sampling)?;
        Ok(step.token)
    }

    fn decode_next_token_with_hidden(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        sampling: &mut FishAudioSamplingState,
    ) -> Result<(u64, Vec<f32>), Error> {
        let step = self.decode_next_step(token_ids, embedding_injection, None, true, sampling)?;
        let hidden = step.last_hidden.ok_or(Error::GenerateFailed)?;
        Ok((step.token, hidden))
    }

    fn advance_with_override_embedding(
        &mut self,
        embedding: &[f32],
        sampling: &mut FishAudioSamplingState,
    ) -> Result<(), Error> {
        self.decode_next_step(&[0], EmbeddingInjection::Override(embedding.to_vec()), None, false, sampling)?;
        Ok(())
    }

    fn decode_next_step(
        &mut self,
        token_ids: &[u64],
        embedding_injection: EmbeddingInjection,
        vocab_limit: Option<usize>,
        capture_hidden: bool,
        sampling: &mut FishAudioSamplingState,
    ) -> Result<DecodeStep, Error> {
        if token_ids.is_empty() {
            return Err(Error::GenerateFailed);
        }

        let positions: Vec<usize> = (self.next_position..self.next_position + token_ids.len()).collect();
        let token_seeds = vec![0_u64; token_ids.len()];

        let mut state = ForwardPassState::new_llm(
            self.context.clone(),
            &self.decoder_config,
            &self.model_shape,
            &self.scratch_buffers,
            self.cache_layers.clone(),
            self.shared_buffers.clone(),
            token_ids,
            &positions,
            None,
            &token_seeds,
            token_ids.len(),
            0,
            token_ids.len(),
            false,
            None,
            false,
            true,
            None,
            None,
        );

        self.command_buffer =
            Rc::new(RefCell::new(self.context.create_command_buffer().expect("Failed to create command buffer")));
        let encoding_parameters = EncodingParameters::new(false, false, false);
        self.executables.embed.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
        self.command_buffer.borrow().submit();
        self.command_buffer.borrow().wait_until_completed();

        apply_embedding_injection(&mut state, &embedding_injection, token_ids.len(), self.decoder_config.model_dim)?;

        self.command_buffer =
            Rc::new(RefCell::new(self.context.create_command_buffer().expect("Failed to create command buffer")));
        for layer in self.executables.layers.iter() {
            layer.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
        }
        self.command_buffer.borrow().submit();
        self.command_buffer.borrow().wait_until_completed();

        let last_hidden = if capture_hidden {
            Some(read_last_hidden_from_main(&state, token_ids.len(), self.decoder_config.model_dim)?)
        } else {
            None
        };

        self.command_buffer =
            Rc::new(RefCell::new(self.context.create_command_buffer().expect("Failed to create command buffer")));
        self.executables.norm.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
        self.executables.readout.encode(&mut state, &encoding_parameters, self.command_buffer.borrow().deref());
        self.command_buffer.borrow().submit();
        self.command_buffer.borrow().wait_until_completed();

        let token = read_sampled_token(&state, self.decoder_config.vocab_size, vocab_limit, sampling)?;
        self.accept_suffix(token_ids.len(), &positions);
        self.next_position = self.next_position.saturating_add(token_ids.len());
        Ok(DecodeStep {
            token,
            last_hidden,
        })
    }

    fn accept_suffix(
        &mut self,
        suffix_len: usize,
        positions: &[usize],
    ) {
        let accepted_suffix_indices: Vec<usize> = (0..suffix_len).collect();
        let command_buffer = self.context.create_command_buffer().expect("Failed to create command buffer");
        self.cache_layers.borrow_mut().update_after_acceptance(
            &accepted_suffix_indices,
            None,
            &command_buffer,
            &self.kv_cache_update,
        );
        command_buffer.submit();
        command_buffer.wait_until_completed();

        self.cache_layers.borrow_mut().register_accepted_tokens(positions);
    }
}

fn argmax_index(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

fn read_sampled_token(
    state: &ForwardPassState<Metal>,
    vocab_size: usize,
    vocab_limit: Option<usize>,
    sampling: &mut FishAudioSamplingState,
) -> Result<u64, Error> {
    let logits = state.arrays(&[ArrayId::Logits])[0].clone();
    let logits = logits.borrow();
    if logits.shape().len() != 2 || logits.shape()[1] != vocab_size || logits.shape()[0] == 0 {
        return Err(Error::GenerateFailed);
    }
    let row_count = logits.shape()[0];
    let row_start = (row_count - 1).checked_mul(vocab_size).ok_or(Error::GenerateFailed)?;
    let row_end = row_start.checked_add(vocab_size).ok_or(Error::GenerateFailed)?;

    let limit = vocab_limit.unwrap_or(vocab_size).min(vocab_size);
    if limit == 0 {
        return Err(Error::GenerateFailed);
    }

    let row_values = match logits.data_type() {
        DataType::F32 => logits.as_slice::<f32>()[row_start..row_end][..limit].to_vec(),
        DataType::F16 => logits.as_slice::<f16>()[row_start..row_end][..limit]
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>(),
        DataType::BF16 => logits.as_slice::<bf16>()[row_start..row_end][..limit]
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>(),
        _ => return Err(Error::GenerateFailed),
    };

    let sampled_index = sampling.sample_index(&row_values)?;

    u64::try_from(sampled_index).map_err(|_| Error::GenerateFailed)
}

fn apply_embedding_injection(
    state: &mut ForwardPassState<Metal>,
    injection: &EmbeddingInjection,
    suffix_len: usize,
    model_dim: usize,
) -> Result<(), Error> {
    if matches!(injection, EmbeddingInjection::None) {
        return Ok(());
    }

    let main = state.arrays(&[ArrayId::Main])[0].clone();
    let mut main = main.borrow_mut();
    if main.shape().len() != 2 || main.shape()[0] != suffix_len || main.shape()[1] != model_dim {
        return Err(Error::GenerateFailed);
    }
    let values_len = suffix_len.checked_mul(model_dim).ok_or(Error::GenerateFailed)?;

    match (injection, main.data_type()) {
        (EmbeddingInjection::Override(values), DataType::F32) => {
            if values.len() != values_len {
                return Err(Error::GenerateFailed);
            }
            main.as_slice_mut::<f32>().copy_from_slice(values);
            Ok(())
        },
        (EmbeddingInjection::Override(values), DataType::F16) => {
            if values.len() != values_len {
                return Err(Error::GenerateFailed);
            }
            for (dst, &src) in main.as_slice_mut::<f16>().iter_mut().zip(values.iter()) {
                *dst = f16::from_f32(src);
            }
            Ok(())
        },
        (EmbeddingInjection::Override(values), DataType::BF16) => {
            if values.len() != values_len {
                return Err(Error::GenerateFailed);
            }
            for (dst, &src) in main.as_slice_mut::<bf16>().iter_mut().zip(values.iter()) {
                *dst = bf16::from_f32(src);
            }
            Ok(())
        },
        (
            EmbeddingInjection::Add {
                values,
                post_scale,
            },
            DataType::F32,
        ) => {
            if values.len() != model_dim {
                return Err(Error::GenerateFailed);
            }
            let scale = post_scale.unwrap_or(1.0);
            let slice = main.as_slice_mut::<f32>();
            for token_index in 0..suffix_len {
                let row_start = token_index * model_dim;
                for dim in 0..model_dim {
                    let idx = row_start + dim;
                    slice[idx] = (slice[idx] + values[dim]) * scale;
                }
            }
            Ok(())
        },
        (
            EmbeddingInjection::Add {
                values,
                post_scale,
            },
            DataType::F16,
        ) => {
            if values.len() != model_dim {
                return Err(Error::GenerateFailed);
            }
            let scale = post_scale.unwrap_or(1.0);
            let slice = main.as_slice_mut::<f16>();
            for token_index in 0..suffix_len {
                let row_start = token_index * model_dim;
                for dim in 0..model_dim {
                    let idx = row_start + dim;
                    let current = f32::from(slice[idx]);
                    slice[idx] = f16::from_f32((current + values[dim]) * scale);
                }
            }
            Ok(())
        },
        (
            EmbeddingInjection::Add {
                values,
                post_scale,
            },
            DataType::BF16,
        ) => {
            if values.len() != model_dim {
                return Err(Error::GenerateFailed);
            }
            let scale = post_scale.unwrap_or(1.0);
            let slice = main.as_slice_mut::<bf16>();
            for token_index in 0..suffix_len {
                let row_start = token_index * model_dim;
                for dim in 0..model_dim {
                    let idx = row_start + dim;
                    let current = f32::from(slice[idx]);
                    slice[idx] = bf16::from_f32((current + values[dim]) * scale);
                }
            }
            Ok(())
        },
        _ => Err(Error::GenerateFailed),
    }
}

fn read_last_hidden_from_main(
    state: &ForwardPassState<Metal>,
    suffix_len: usize,
    model_dim: usize,
) -> Result<Vec<f32>, Error> {
    let main = state.arrays(&[ArrayId::Main])[0].clone();
    let main = main.borrow();
    if main.shape().len() != 2 || main.shape()[0] != suffix_len || main.shape()[1] != model_dim || suffix_len == 0 {
        return Err(Error::GenerateFailed);
    }

    let start = (suffix_len - 1).checked_mul(model_dim).ok_or(Error::GenerateFailed)?;
    let end = start.checked_add(model_dim).ok_or(Error::GenerateFailed)?;

    match main.data_type() {
        DataType::F32 => Ok(main.as_slice::<f32>()[start..end].to_vec()),
        DataType::F16 => Ok(main.as_slice::<f16>()[start..end].iter().map(|&v| f32::from(v)).collect()),
        DataType::BF16 => Ok(main.as_slice::<bf16>()[start..end].iter().map(|&v| f32::from(v)).collect()),
        _ => Err(Error::GenerateFailed),
    }
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

    fn matmul(
        &self,
        input: &[f32],
    ) -> Option<Vec<f32>> {
        if input.len() != self.cols {
            return None;
        }
        let mut output = vec![0.0_f32; self.rows];
        for (row_index, row) in self.values.chunks_exact(self.cols).enumerate() {
            let mut acc = 0.0_f32;
            for (&w, &x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            output[row_index] = acc;
        }
        Some(output)
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
        DEFAULT_FISHAUDIO_RANDOM_SEED, DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE, DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE,
        DEFAULT_FISHAUDIO_SAMPLING_TOP_P, DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE, DEFAULT_STUB_SEED,
        FishAudioSamplingState, MatrixF32, StreamingTokenAccumulator, default_repeat_window_size,
        default_short_logits_size, generate_stub_tokens, load_stub_seed, semantic_token_to_code,
    };
    use crate::audio::AudioTokenPacking;

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
    fn fishaudio_decoder_defaults_match_lalamo_consts() {
        assert_eq!(default_short_logits_size(), DEFAULT_FISHAUDIO_SHORT_LOGITS_SIZE);
        assert_eq!(default_repeat_window_size(), DEFAULT_FISHAUDIO_REPEAT_WINDOW_SIZE);
        assert_eq!(DEFAULT_FISHAUDIO_RANDOM_SEED, 123);
        assert_eq!(DEFAULT_FISHAUDIO_SAMPLING_TEMPERATURE, 0.8008);
        assert_eq!(DEFAULT_FISHAUDIO_SAMPLING_TOP_P, 0.8008);
    }

    #[test]
    fn fishaudio_sampling_is_seeded() {
        let logits = [2.0_f32, 1.0, 0.5];
        let mut a = FishAudioSamplingState::new(123);
        let mut b = FishAudioSamplingState::new(123);

        let sample_a = (0..8).map(|_| a.sample_index(&logits).expect("sample")).collect::<Vec<_>>();
        let sample_b = (0..8).map(|_| b.sample_index(&logits).expect("sample")).collect::<Vec<_>>();
        assert_eq!(sample_a, sample_b);
    }

    #[test]
    fn fishaudio_sampling_top_p_zero_falls_back_to_argmax() {
        let logits = [0.1_f32, 1.2, -0.4, 0.7];
        let mut sampler = FishAudioSamplingState::with_params(999, 0.8, 0.0);
        for _ in 0..8 {
            assert_eq!(sampler.sample_index(&logits).expect("sample"), 1);
        }
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
        assert_eq!(matrix.matmul(&[1.0, 0.0, 1.0]), Some(vec![4.0, 10.0]));
        assert_eq!(matrix.matmul(&[1.0, 0.0]), None);
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
}
