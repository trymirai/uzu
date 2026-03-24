#![cfg(all(feature = "audio-runtime", metal_backend))]

mod audio_backend;
mod backend_factory;
mod decoder_runtime;
mod decoder_support;
mod fishaudio;
mod generator;

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use backend_factory::load_tts_runtime;
use decoder_runtime::*;
use decoder_support::*;
use minijinja::{Environment, context};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tokenizers::Tokenizer;

use crate::{
    DataType,
    array::ArrayContextExt,
    audio::{
        AudioCodecRuntime, AudioGenerationContext, AudioPcmBatch, AudioTokenGrid,
        nanocodec::{AudioDecodeStepStats, AudioDecodeStreamState},
    },
    backends::common::{
        Backend, Context as BackendContext, Encoder, Kernels,
        kernel::{
            EmbeddingRowsSumKernel, TensorAddScaleKernel, TensorCopyKernel, TokenCopySampledKernel,
            TokenCopyToResultsKernel,
            kv_cache_update::KVCacheUpdate,
            matmul::{MatmulArguments, MatmulKernel, MatmulKernels},
        },
    },
    config::{InnerModelConfig, ModelMetadata, TtsMessageProcessorConfig},
    encodable_block::{Decoder, EncodingParameters, Sampling as GpuSampling},
    forward_pass::{
        cache_layers::CacheLayers,
        model_shape::ModelShape,
        scratch_buffers::ScratchBuffers,
        state::{ArrayId, ForwardPassState, SharedBuffers},
    },
    parameters::ParameterLoader,
    session::{
        config::{TextDecoderRuntimeConfig, TextSamplingConfig, TtsChunkPolicy, TtsRunConfig, TtsSessionOptions},
        parameter::{ConfigResolvableValue, SamplingMethod, SamplingProcessingOrder},
        types::{Error, Input},
    },
};

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
pub(super) struct RunnerInstrumentation {
    pub(super) command_buffers_submitted: usize,
    pub(super) host_waits: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AdaptiveChunkController {
    pub(super) ema_ms_per_frame: Option<f64>,
    pub(super) current_chunk_frames: usize,
}

pub struct TtsSession<B: Backend> {
    tokenizer: Tokenizer,
    audio: AudioGenerationContext<B>,
    audio_decoder: Box<dyn AudioDecoderBackend>,
    message_processor_config: TtsMessageProcessorConfig,
    text_decoder: Box<dyn SemanticDecoderBackend>,
    last_execution_stats: Option<TtsExecutionStats>,
}

trait SemanticDecoderBackend {
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

pub(super) trait PendingAudioChunkBackend {
    fn step_stats(&self) -> Option<AudioDecodeStepStats>;

    fn try_resolve(&mut self) -> Result<Option<AudioPcmBatch>, Error>;

    fn resolve(self: Box<Self>) -> Result<AudioPcmBatch, Error>;

    fn resolve_with_decode_duration(self: Box<Self>) -> Result<(AudioPcmBatch, std::time::Duration), Error> {
        let start = Instant::now();
        let pcm = self.resolve()?;
        Ok((pcm, start.elapsed()))
    }

    fn try_resolve_with_decode_duration(&mut self) -> Result<Option<(AudioPcmBatch, std::time::Duration)>, Error> {
        let start = Instant::now();
        let Some(pcm) = self.try_resolve()? else {
            return Ok(None);
        };
        Ok(Some((pcm, start.elapsed())))
    }
}

struct ImmediatePendingAudioChunk {
    pcm: Option<AudioPcmBatch>,
    step_stats: Option<AudioDecodeStepStats>,
}

impl PendingAudioChunkBackend for ImmediatePendingAudioChunk {
    fn step_stats(&self) -> Option<AudioDecodeStepStats> {
        self.step_stats
    }

    fn try_resolve(&mut self) -> Result<Option<AudioPcmBatch>, Error> {
        Ok(Some(self.pcm.take().ok_or(Error::GenerateFailed)?))
    }

    fn resolve(mut self: Box<Self>) -> Result<AudioPcmBatch, Error> {
        self.pcm.take().ok_or(Error::GenerateFailed)
    }
}

struct NanoCodecPendingAudioChunk<B: Backend> {
    inner: Option<crate::audio::nanocodec::runtime::PendingStreamPcmChunk<B>>,
}

impl<B: Backend> PendingAudioChunkBackend for NanoCodecPendingAudioChunk<B> {
    fn step_stats(&self) -> Option<AudioDecodeStepStats> {
        self.inner.as_ref().map(|pending| pending.step_stats())
    }

    fn try_resolve(&mut self) -> Result<Option<AudioPcmBatch>, Error> {
        let ready = self.inner.as_ref().is_none_or(|pending| pending.is_complete());
        if !ready {
            return Ok(None);
        }

        let pending = self.inner.take().ok_or(Error::GenerateFailed)?;
        pending.resolve().map(Some).map_err(Error::from)
    }

    fn resolve(mut self: Box<Self>) -> Result<AudioPcmBatch, Error> {
        let pending = self.inner.take().ok_or(Error::GenerateFailed)?;
        pending.resolve().map_err(Error::from)
    }
}

pub(super) struct PendingStreamingChunk {
    pub(super) submission_decode_duration: std::time::Duration,
    pub(super) ready_frames: usize,
    pub(super) next_chunk_frames: usize,
    pub(super) chunk: Box<dyn PendingAudioChunkBackend>,
}
