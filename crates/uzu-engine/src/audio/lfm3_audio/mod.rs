//! LFM3-Audio speech sidecar operators:
//! log-mel front-end (CPU STFT/filterbank), conv downsampler, projector, and
//! MaskGIT depth decoder (Metal GPU in `features_metal` / `depth_metal`, with
//! CPU f32 reference implementations in `features` / `depth` for unit tests).
//!
//! The trunk of the speech encoder is a plain LFM2-style stack and runs on the
//! generic uzu `Decoder<Metal>` (see `pipeline.rs`); this module covers
//! everything that is not a decoder layer. Weights come from the bundle's
//! `audio.safetensors` (BF16 raw, MLX layout preserved); see
//! `scripts/lfm3-audio-convert.py` for the layout contract and
//! `scripts/lfm3_audio_ref.py` for the golden reference.

pub(crate) mod features;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod features_metal;
mod depth;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod depth_metal;

pub use depth::{DepthDecoder, DepthWeights};
pub use features::{Downsampler, Projector};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod pipeline;

use anyhow::Result;
use serde::Deserialize;

use crate::config::decoder::DecoderConfig;

#[derive(Deserialize, Clone)]
pub struct DepthConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f64,
    pub rope_theta: f64,
}

#[derive(Deserialize, Clone, Copy, Debug)]
pub struct AudioSamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

/// `audio_config.json` of an LFM3-Audio bundle.
#[derive(Deserialize)]
pub struct Lfm3AudioConfig {
    pub audio_token_id: u64,
    pub audio_out_token_id: u64,
    pub audio_start_token_id: u64,
    pub text_end_token_id: u64,
    pub mixed_start_token_id: u64,
    pub eos_token_ids: Vec<u64>,
    pub vocab_size: u32,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub audio_eos_code: i64,
    pub maskgit_schedule: Vec<usize>,
    pub interleaved_n_text: usize,
    pub interleaved_n_audio: usize,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub num_mel_bins: usize,
    pub log_floor: f32,
    pub norm_offset: f32,
    pub norm_divisor: f32,
    pub encoder_stride: usize,
    pub encoder_decoder_config: DecoderConfig,
    pub encoder_hidden: usize,
    pub conv_channels: usize,
    pub encoder_sliding_window: usize,
    pub depth: DepthConfig,
    pub audio_sampling: AudioSamplingConfig,
    pub text_repetition_window: usize,
}

impl Lfm3AudioConfig {
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("audio_config.json");
        let text = std::fs::read_to_string(&path)?;
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("Parsing {}: {e}", path.display()))
    }

    /// Spectrogram frames of a clip right-padded to a whole hop.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        let padded = if num_samples < self.n_fft {
            self.n_fft
        } else {
            num_samples + (num_samples - self.n_fft) % self.hop_length
        };
        1 + (padded - self.n_fft) / self.hop_length
    }

    /// Encoder tokens per spectrogram frame count (stride 8).
    pub fn num_encoder_tokens(&self, num_frames: usize) -> usize {
        num_frames.div_ceil(self.encoder_stride)
    }

    /// Frames that end a speech segment: every codebook holds the EOS code.
    pub fn is_eos_frame(&self, codes: &[i64]) -> bool {
        codes.len() == self.num_codebooks && codes.iter().all(|&c| c == self.audio_eos_code)
    }
}

