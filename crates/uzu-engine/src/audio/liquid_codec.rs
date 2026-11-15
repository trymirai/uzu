//! Liquid-Codec audio tokenizer (LiquidAI), CPU f32 with bf16 boundary rounding.
//!
//! Port of the HF reference (`modeling_liquid_codec.py` + its embedded
//! `modeling_lfm3.py`): STFT features -> conv frontend -> LFM3 encoder
//! (8 layers, conv/attention hybrid, causal) -> GFSQ quantizer (12 codebooks
//! × 4 dims, levels `[8, 7, 6, 6]`) -> LFM3 decoder (sliding window 30) ->
//! ISTFT. Weights are read straight from the HF snapshot `model.safetensors`
//! (BF16, keys as shipped); every module boundary rounds to bf16 to mirror
//! the reference's bf16 compute, while STFT/ISTFT and the quantizer stay f32
//! as in the reference. Correctness convention per
//! docs/lfm3-vl-phase-b-handoff.md: integer codes must match the reference
//! exactly, floats carry a max-abs-diff bound (see `unit/liquid_codec_test.rs`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use super::fft::{irfft, rfft};
use crate::vision::tower::{LinearWeight, SafetensorLoader};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Deserialize)]
pub struct LiquidCodecConfig {
    #[serde(default = "usize_512")]
    pub encoder_dim: usize,
    #[serde(default = "usize_256")]
    pub encoder_conv_dim: usize,
    #[serde(default = "usize_1280")]
    pub stft_n_fft: usize,
    #[serde(default = "usize_320")]
    pub stft_hop_length: usize,
    #[serde(default = "usize_1280")]
    pub stft_win_length: usize,
    #[serde(default = "usize_6")]
    pub stft_frame_stack: usize,
    #[serde(default = "usize_256")]
    pub quantizer_dim: usize,
    #[serde(default = "default_fsq_levels")]
    pub fsq_levels: Vec<i64>,
    #[serde(default = "usize_12")]
    pub n_q: usize,
    #[serde(default = "usize_30")]
    pub decoder_window: usize,
    #[serde(default = "usize_24000")]
    pub sampling_rate: usize,
}

fn usize_512() -> usize {
    512
}
fn usize_256() -> usize {
    256
}
fn usize_1280() -> usize {
    1280
}
fn usize_320() -> usize {
    320
}
fn usize_6() -> usize {
    6
}
fn usize_12() -> usize {
    12
}
fn usize_30() -> usize {
    30
}
fn usize_24000() -> usize {
    24_000
}
fn default_fsq_levels() -> Vec<i64> {
    vec![8, 7, 6, 6]
}

impl LiquidCodecConfig {
    /// LFM3 backbone geometry embedded in the codec (see the reference
    /// `LiquidCodecConfig::lfm_config`): attention layers at indices 2, 4, 6.
    const LFM_LAYERS: usize = 8;
    const LFM_HEADS: usize = 8;
    const LFM_KV_HEADS: usize = 2;
    const LFM_HEAD_DIM: usize = 64;
    pub(crate) const LFM_INTERMEDIATE: usize = 2304;
    const LFM_CONV_KERNEL: usize = 7;
    const LFM_EPS: f32 = 1e-5;
    const ROPE_THETA: f64 = 1e6;

    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("Reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("Parsing {}", path.display()))
    }

    /// Samples per codec frame (24000 / (320 · 6) = 12.5 frames/s).
    pub fn frame_size(&self) -> usize {
        self.stft_hop_length * self.stft_frame_stack
    }

    /// Product of the FSQ levels (8 · 7 · 6 · 6 = 2016).
    pub fn codebook_size(&self) -> i64 {
        self.fsq_levels.iter().product()
    }

    pub fn bins(&self) -> usize {
        self.stft_n_fft / 2 + 1
    }
}

// ---------------------------------------------------------------------------
// bf16 helpers
// ---------------------------------------------------------------------------

/// Round f32 to bf16 precision (round-to-nearest-even), staying in f32.
#[inline]
pub(crate) fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = (bits + 0x8000 + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[inline]
pub(crate) fn round_bf16(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = bf16_round(*v);
    }
}

/// Lfm3RMSNorm: variance in f32, `x · rsqrt(var + eps)` rounded to bf16, then
/// multiplied by the (bf16) weight and rounded again.
fn rms_norm_bf16(
    x: &mut [f32],
    weight: &[f32],
    eps: f32,
) {
    let dim = weight.len();
    for row in x.chunks_exact_mut(dim) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for (d, v) in row.iter_mut().enumerate() {
            *v = bf16_round(weight[d] * bf16_round(*v * inv));
        }
    }
}

#[inline]
fn silu_bf16(x: f32) -> f32 {
    bf16_round(x / (1.0 + (-x).exp()))
}

#[inline]
fn sigmoid_bf16(x: f32) -> f32 {
    bf16_round(1.0 / (1.0 + (-x).exp()))
}

fn linear_round(
    w: &LinearWeight,
    x: &[f32],
    num_tokens: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; num_tokens * w.out_features];
    w.forward(x, num_tokens, &mut out);
    round_bf16(&mut out);
    out
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

pub(crate) enum Mixer {
    /// `in_proj -> [B|C|x] -> B*x -> causal depthwise conv k=7 -> C*· -> out_proj`.
    Conv {
        in_proj: LinearWeight,
        conv: Vec<f32>, // [dim, kernel]
        out_proj: LinearWeight,
    },
    /// Qwen3-style attention with head-dim RMSNorm, full RoPE and a sigmoid
    /// per-head output gate.
    Attention {
        q_proj: LinearWeight,
        k_proj: LinearWeight,
        v_proj: LinearWeight,
        o_proj: LinearWeight,
        g_proj: LinearWeight,
        q_norm: Vec<f32>, // [head_dim]
        k_norm: Vec<f32>,
        sliding_window: Option<usize>,
    },
}

pub(crate) struct Lfm3Layer {
    pub(crate) mixer: Mixer,
    pub(crate) operator_norm: Vec<f32>,
    pub(crate) post_operator_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) post_ffn_norm: Vec<f32>,
    pub(crate) w1: LinearWeight,
    pub(crate) w2: LinearWeight,
    pub(crate) w3: LinearWeight,
}

pub(crate) struct Lfm3 {
    pub(crate) layers: Vec<Lfm3Layer>,
    pub(crate) embedding_norm: Vec<f32>,
}

/// Streaming decoder state: LFM3 KV cache + conv states + the unfinished
/// overlap-add tail (reference `LiquidCodecStreamingCache`).
pub struct StreamingCache {
    run: Lfm3RunState,
}

struct Lfm3RunState {
    keys: Vec<Vec<f32>>, // per attention layer [total_kv_tokens * kv_dim]
    values: Vec<Vec<f32>>,
    conv_state: Vec<Vec<f32>>, // per conv layer [dim * (kernel - 1)]
    seen_tokens: usize,
    audio_tail: Option<Vec<f32>>,
    envelope_tail: Option<Vec<f32>>,
    started: bool,
    finished: bool,
}

pub struct LiquidCodec {
    pub(crate) config: LiquidCodecConfig,
    // front end
    pub(crate) in_proj: LinearWeight,       // [256, 1923, 1]
    pub(crate) frontend_conv: LinearWeight, // [256, 256, 3] flattened to rows [out, in*k], causal left-pad 2
    pub(crate) frontend_out: LinearWeight,  // [512, 256, 6] flattened to rows [out, in*k], stride 6
    // encoder / decoder LFM3 backbones
    pub(crate) encoder: Lfm3,
    pub(crate) decoder: Lfm3,
    // quantizer projections
    pub(crate) to_quantizer: LinearWeight,          // [256, 512]
    pub(crate) quantizer_project_in: LinearWeight,  // [48, 256]
    pub(crate) quantizer_project_out: LinearWeight, // [256, 48]
    pub(crate) from_quantizer: LinearWeight,        // [512, 256]
    pub(crate) dec_lin: LinearWeight,               // [1282, 512]
    // synthesis
    pub(crate) window: Vec<f32>, // Hann, periodic, [1280]
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

impl LiquidCodec {
    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = LiquidCodecConfig::load(model_dir)?;
        let mut loader = SafetensorLoader::open(model_dir.join("model.safetensors"))
            .with_context(|| format!("Opening codec weights in {}", model_dir.display()))?;

        ensure!(
            config.fsq_levels.len() == 4 && config.n_q == 12,
            "this port assumes the released geometry (4 FSQ dims, 12 codebooks)"
        );
        let features_dim = 3 * config.bins();

        // `base` is the module path, e.g. "to_quantizer"; bias is optional
        // (attention has `attention_bias = false`).
        let read_linear = |loader: &mut SafetensorLoader, base: &str| -> Result<LinearWeight> {
            let (w, shape) = loader.read_tensor_f32(&format!("{base}.weight"))?;
            ensure!(shape.len() == 2, "expected 2D weight for '{base}', got {shape:?}");
            let bias = loader.read_tensor_f32(&format!("{base}.bias")).ok().map(|(b, _)| b);
            Ok(LinearWeight {
                weight: w,
                bias,
                in_features: shape[1],
                out_features: shape[0],
            })
        };

        // Frontend (stored under `_frontend.` or `frontend.` prefix); conv weights
        // are [out, in, k] 3D, flattened to per-out rows of [in*k].
        let frontend_prefix = if loader.read_tensor_f32("_frontend.in_proj.weight").is_ok() {
            "_frontend"
        } else {
            "frontend"
        };
        let (in_proj_w, in_proj_shape) = loader.read_tensor_f32(&format!("{frontend_prefix}.in_proj.weight"))?;
        ensure!(in_proj_shape == [config.encoder_conv_dim, features_dim, 1], "in_proj {in_proj_shape:?}");
        let (in_proj_b, _) = loader.read_tensor_f32(&format!("{frontend_prefix}.in_proj.bias"))?;
        let in_proj = LinearWeight {
            weight: in_proj_w,
            bias: Some(in_proj_b),
            in_features: features_dim,
            out_features: config.encoder_conv_dim,
        };

        let (conv_w, conv_shape) = loader.read_tensor_f32(&format!("{frontend_prefix}.conv.weight"))?;
        ensure!(conv_shape == [config.encoder_conv_dim, config.encoder_conv_dim, 3], "frontend conv {conv_shape:?}");
        let (conv_b, _) = loader.read_tensor_f32(&format!("{frontend_prefix}.conv.bias"))?;
        // [out, in, k] is already row-major per out: in*k contiguous.
        let frontend_conv = LinearWeight {
            weight: conv_w,
            bias: Some(conv_b),
            in_features: config.encoder_conv_dim * 3,
            out_features: config.encoder_conv_dim,
        };

        let (out_w, out_shape) = loader.read_tensor_f32(&format!("{frontend_prefix}.out.weight"))?;
        ensure!(
            out_shape == [config.encoder_dim, config.encoder_conv_dim, config.stft_frame_stack],
            "frontend out {out_shape:?}"
        );
        let (out_b, _) = loader.read_tensor_f32(&format!("{frontend_prefix}.out.bias"))?;
        let frontend_out = LinearWeight {
            weight: out_w,
            bias: Some(out_b),
            in_features: config.encoder_conv_dim * config.stft_frame_stack,
            out_features: config.encoder_dim,
        };

        let lfm3 = |loader: &mut SafetensorLoader, prefix: &str, sliding_window: Option<usize>| -> Result<Lfm3> {
            let mut layers = Vec::with_capacity(LiquidCodecConfig::LFM_LAYERS);
            for li in 0..LiquidCodecConfig::LFM_LAYERS {
                let mixer = if matches!(li, 2 | 4 | 6) {
                    let p = format!("{prefix}.layers.{li}.self_attn");
                    Mixer::Attention {
                        q_proj: read_linear(loader, &format!("{p}.q_proj"))?,
                        k_proj: read_linear(loader, &format!("{p}.k_proj"))?,
                        v_proj: read_linear(loader, &format!("{p}.v_proj"))?,
                        o_proj: read_linear(loader, &format!("{p}.o_proj"))?,
                        g_proj: read_linear(loader, &format!("{p}.gate_proj"))?,
                        q_norm: loader.read_tensor_f32(&format!("{p}.q_norm.weight"))?.0,
                        k_norm: loader.read_tensor_f32(&format!("{p}.k_norm.weight"))?.0,
                        sliding_window,
                    }
                } else {
                    let p = format!("{prefix}.layers.{li}.conv");
                    let (cw, cshape) = loader.read_tensor_f32(&format!("{p}.conv.weight"))?;
                    ensure!(cshape.len() == 3 && cshape[1] == 1, "depthwise conv {cshape:?}");
                    Mixer::Conv {
                        in_proj: read_linear(loader, &format!("{p}.in_proj"))?,
                        conv: cw, // [dim, 1, kernel] -> [dim, kernel] rows as stored
                        out_proj: read_linear(loader, &format!("{p}.out_proj"))?,
                    }
                };
                let p = format!("{prefix}.layers.{li}");
                layers.push(Lfm3Layer {
                    mixer,
                    operator_norm: loader.read_tensor_f32(&format!("{p}.operator_norm.weight"))?.0,
                    post_operator_norm: loader.read_tensor_f32(&format!("{p}.post_operator_norm.weight"))?.0,
                    ffn_norm: loader.read_tensor_f32(&format!("{p}.ffn_norm.weight"))?.0,
                    post_ffn_norm: loader.read_tensor_f32(&format!("{p}.post_ffn_norm.weight"))?.0,
                    w1: read_linear(loader, &format!("{p}.feed_forward.w1"))?,
                    w2: read_linear(loader, &format!("{p}.feed_forward.w2"))?,
                    w3: read_linear(loader, &format!("{p}.feed_forward.w3"))?,
                });
            }
            Ok(Lfm3 {
                layers,
                embedding_norm: loader.read_tensor_f32(&format!("{prefix}.embedding_norm.weight"))?.0,
            })
        };

        let encoder = lfm3(&mut loader, "encoder.lfm", None)?;
        let decoder = lfm3(&mut loader, "decoder.lfm", Some(config.decoder_window))?;

        let quantizer_prefix = if loader.read_tensor_f32("quantizer.fsq.project_in.weight").is_ok() {
            "quantizer.fsq"
        } else {
            "quantizer"
        };

        Ok(Self {
            quantizer_project_in: read_linear(&mut loader, &format!("{quantizer_prefix}.project_in"))?,
            quantizer_project_out: read_linear(&mut loader, &format!("{quantizer_prefix}.project_out"))?,
            to_quantizer: read_linear(&mut loader, "to_quantizer")?,
            from_quantizer: read_linear(&mut loader, "from_quantizer")?,
            dec_lin: read_linear(&mut loader, "decoder.lin")?,
            window: loader.read_tensor_f32("decoder.istft.window")?.0,
            in_proj,
            frontend_conv,
            frontend_out,
            encoder,
            decoder,
            config,
        })
    }
}

// ---------------------------------------------------------------------------
// STFT / ISTFT
// ---------------------------------------------------------------------------

impl LiquidCodec {
    /// Analysis transform: reflect-pad `audio` by `(win - hop) / 2` per side,
    /// Hann-windowed FFT frames, features `[log1p(|spec|) | cos | sin]` as
    /// `[T, 3 * bins]`.
    fn stft_features(
        &self,
        audio: &[f32],
    ) -> Vec<f32> {
        let cfg = &self.config;
        let (n_fft, hop) = (cfg.stft_n_fft, cfg.stft_hop_length);
        let total_pad = cfg.stft_win_length - hop;
        let (left, right) = (total_pad / 2, total_pad - total_pad / 2);
        let n = audio.len();

        let mut padded = vec![0f32; n + left + right];
        padded[left..left + n].copy_from_slice(audio);
        // torch reflect excludes the edge sample
        for j in 0..left {
            padded[j] = audio[left - j];
        }
        for j in 0..right {
            padded[left + n + j] = audio[n - 2 - j];
        }

        let num_frames = (padded.len() - n_fft) / hop + 1;
        let bins = cfg.bins();
        let mut features = vec![0f32; num_frames * 3 * bins];
        let mut re = vec![0f32; n_fft];
        let mut im = vec![0f32; n_fft];
        for t in 0..num_frames {
            let start = t * hop;
            for (j, fr) in re.iter_mut().enumerate() {
                *fr = padded[start + j] * self.window[j];
            }
            im.iter_mut().for_each(|v| *v = 0.0);
            let (bin_re, bin_im) = rfft(&re);
            let row = &mut features[t * 3 * bins..(t + 1) * 3 * bins];
            for (b, (r, i)) in bin_re.iter().zip(&bin_im).enumerate() {
                let mag = (r * r + i * i).sqrt().clamp(1e-5, f32::MAX);
                row[b] = mag.ln_1p();
                row[bins + b] = r / mag;
                row[2 * bins + b] = i / mag;
            }
        }
        features
    }

    /// Synthesis: per-frame `irfft -> window -> overlap-add`, Hann-squared
    /// envelope accumulated alongside. Returns `(audio, envelope)`.
    fn istft(
        &self,
        specs: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.config;
        let n_fft = cfg.stft_n_fft;
        let num_frames = specs.len();
        let output_size = (num_frames - 1) * cfg.stft_hop_length + cfg.stft_win_length;
        let mut audio = vec![0f32; output_size];
        let mut envelope = vec![0f32; output_size];
        let mut frame = vec![0f32; n_fft];
        for (t, spec) in specs.iter().enumerate() {
            irfft(&spec[..cfg.bins()], &spec[cfg.bins()..], &mut frame);
            let base = t * cfg.stft_hop_length;
            for (j, &f) in frame.iter().enumerate() {
                audio[base + j] += f * self.window[j];
                envelope[base + j] += self.window[j] * self.window[j];
            }
        }
        (audio, envelope)
    }
}

// ---------------------------------------------------------------------------
// Conv frontend
// ---------------------------------------------------------------------------

impl LiquidCodec {
    /// `in_proj(1x1) -> SiLU -> causal conv k=3 -> SiLU -> conv k=6 stride 6`.
    /// Input `[T, features]`, output `[T / frame_stack, encoder_dim]`.
    fn frontend(
        &self,
        features: &[f32],
        num_frames: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let dim = cfg.encoder_conv_dim;
        let mut h = linear_round(&self.in_proj, features, num_frames);
        for v in &mut h {
            *v = silu_bf16(*v);
        }

        // causal conv k=3 (pad=2 + [:T] keeps only the left zero pad)
        let mut c = vec![0f32; num_frames * dim];
        for t in 0..num_frames {
            for d in 0..dim {
                let mut acc = self.frontend_conv.bias.as_ref().unwrap()[d];
                let w_row = &self.frontend_conv.weight[d * dim * 3..(d + 1) * dim * 3];
                for ci in 0..dim {
                    for k in 0..3 {
                        let src_t = t as isize + k as isize - 2;
                        if src_t >= 0 {
                            acc += h[src_t as usize * dim + ci] * w_row[ci * 3 + k];
                        }
                    }
                }
                c[t * dim + d] = bf16_round(acc);
            }
        }
        for v in &mut c {
            *v = silu_bf16(*v);
        }

        // downsample k=frame_stack stride=frame_stack
        let out_dim = cfg.encoder_dim;
        let fs = cfg.stft_frame_stack;
        let codec_frames = num_frames / fs;
        let mut out = vec![0f32; codec_frames * out_dim];
        for tau in 0..codec_frames {
            for d in 0..out_dim {
                let mut acc = self.frontend_out.bias.as_ref().unwrap()[d];
                let w_row = &self.frontend_out.weight[d * dim * fs..(d + 1) * dim * fs];
                for ci in 0..dim {
                    for k in 0..fs {
                        acc += c[(tau * fs + k) * dim + ci] * w_row[ci * fs + k];
                    }
                }
                out[tau * out_dim + d] = bf16_round(acc);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// GFSQ quantizer
// ---------------------------------------------------------------------------

impl LiquidCodec {
    /// Latents `[T, 256]` (bf16 grid) -> indices `[n_q][T]`.
    fn quantizer_encode(
        &self,
        latents: &[f32],
        num_frames: usize,
    ) -> Vec<Vec<i64>> {
        let cfg = &self.config;
        let projected = linear_round(&self.quantizer_project_in, latents, num_frames);
        let dims = cfg.fsq_levels.len();
        let eps = 1e-3f32;
        let half_levels: Vec<f32> = cfg.fsq_levels.iter().map(|&l| ((l - 1) as f32) * (1.0 + eps) / 2.0).collect();
        let offsets: Vec<f32> = cfg
            .fsq_levels
            .iter()
            .map(|&l| {
                if l % 2 == 0 {
                    0.5
                } else {
                    0.0
                }
            })
            .collect();
        let half_width: Vec<f32> = cfg.fsq_levels.iter().map(|&l| (l / 2) as f32).collect();
        let mut basis = vec![1f32; dims];
        for d in 1..dims {
            basis[d] = basis[d - 1] * cfg.fsq_levels[d - 1] as f32;
        }

        let mut codes = vec![vec![0i64; num_frames]; cfg.n_q];
        for t in 0..num_frames {
            for cb in 0..cfg.n_q {
                let mut index = 0f32;
                for d in 0..dims {
                    let x = projected[t * cfg.n_q * dims + cb * dims + d];
                    let shift = (offsets[d] / half_levels[d]).atanh();
                    let bounded = (x + shift).tanh() * half_levels[d] - offsets[d];
                    let r = bounded.round_ties_even() / half_width[d];
                    index += (r * half_width[d] + half_width[d]) * basis[d];
                }
                codes[cb][t] = index.round_ties_even() as i64;
            }
        }
        codes
    }

    /// Indices `[n_q][T]` -> latents `[T, quantizer_dim]` (bf16 grid).
    fn quantizer_decode(
        &self,
        codes: &[Vec<i64>],
        num_frames: usize,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let dims = cfg.fsq_levels.len();
        let half_width: Vec<i64> = cfg.fsq_levels.iter().map(|&l| l / 2).collect();
        let mut basis = vec![1i64; dims];
        for d in 1..dims {
            basis[d] = basis[d - 1] * cfg.fsq_levels[d - 1];
        }
        let mut latents = vec![0f32; num_frames * cfg.n_q * dims];
        for t in 0..num_frames {
            for cb in 0..cfg.n_q {
                for (d, &bw) in basis.iter().enumerate() {
                    let idx = codes[cb][t];
                    let value = (idx / bw) % cfg.fsq_levels[d];
                    latents[t * cfg.n_q * dims + cb * dims + d] =
                        bf16_round(((value - half_width[d]) as f32) / half_width[d] as f32);
                }
            }
        }
        linear_round(&self.quantizer_project_out, &latents, num_frames)
    }
}

// ---------------------------------------------------------------------------
// LFM3 backbone
// ---------------------------------------------------------------------------

const LFM_HEAD_DIM: usize = LiquidCodecConfig::LFM_HEAD_DIM;
const LFM_KV_DIM: usize = LiquidCodecConfig::LFM_KV_HEADS * LFM_HEAD_DIM;

/// NeoX half-rotation rope on one head (f32 math, bf16-rounded output).
fn apply_rope(
    head: &mut [f32],
    cos: &[f32],
    sin: &[f32],
) {
    let half = head.len() / 2;
    let vals: Vec<f32> = head.to_vec();
    for d in 0..half {
        let (a, b) = (vals[d], vals[d + half]);
        head[d] = bf16_round(a * cos[d] - b * sin[d]);
        head[d + half] = bf16_round(b * cos[d + half] + a * sin[d + half]);
    }
}

impl LiquidCodec {
    fn rope_inv_freq() -> Vec<f64> {
        (0..LFM_HEAD_DIM / 2)
            .map(|i| LiquidCodecConfig::ROPE_THETA.powf(-(2.0 * i as f64) / LFM_HEAD_DIM as f64))
            .collect()
    }

    /// `(cos, sin)` tables `[T][head_dim]` for absolute positions
    /// `base..base + T`.
    fn rope_tables(
        base: usize,
        num_tokens: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let inv_freq = Self::rope_inv_freq();
        let mut cos = Vec::with_capacity(num_tokens);
        let mut sin = Vec::with_capacity(num_tokens);
        for p in base..base + num_tokens {
            let angles: Vec<f32> = inv_freq.iter().map(|&f| (p as f64 * f) as f32).collect();
            let mut c = Vec::with_capacity(LFM_HEAD_DIM);
            let mut s = Vec::with_capacity(LFM_HEAD_DIM);
            for d in 0..LFM_HEAD_DIM {
                let half = d % (LFM_HEAD_DIM / 2);
                c.push(angles[half].cos());
                s.push(angles[half].sin());
            }
            cos.push(c);
            sin.push(s);
        }
        (cos, sin)
    }

    /// Lfm3Model forward over `[T, dim]` (causal; `state` carries KV/conv state
    /// across streaming chunks).
    fn lfm_forward(
        &self,
        lfm: &Lfm3,
        x: &[f32],
        num_tokens: usize,
        mut state: Option<&mut Lfm3RunState>,
    ) -> Vec<f32> {
        let cfg = &self.config;
        let dim = cfg.encoder_dim;
        let eps = LiquidCodecConfig::LFM_EPS;
        let base_pos = state.as_ref().map(|s| s.seen_tokens).unwrap_or(0);
        let (cos_t, sin_t) = Self::rope_tables(base_pos, num_tokens);

        let mut x = x.to_vec();
        let mut conv_idx = 0usize;
        let mut attn_idx = 0usize;
        for layer in &lfm.layers {
            // --- mixer half (sandwich norm) ---
            let mut h = x.clone();
            rms_norm_bf16(&mut h, &layer.operator_norm, eps);

            let mixer_out = match &layer.mixer {
                Mixer::Conv {
                    in_proj,
                    conv,
                    out_proj,
                } => {
                    let bcx = linear_round(in_proj, &h, num_tokens);
                    let kernel = LiquidCodecConfig::LFM_CONV_KERNEL;
                    let state_len = kernel - 1;
                    let mut bx = vec![0f32; num_tokens * dim]; // B * x
                    let mut cvec = vec![0f32; num_tokens * dim];
                    for t in 0..num_tokens {
                        for d in 0..dim {
                            let b = bcx[t * 3 * dim + d];
                            let xx = bcx[t * 3 * dim + 2 * dim + d];
                            cvec[t * dim + d] = bcx[t * 3 * dim + dim + d];
                            bx[t * dim + d] = bf16_round(b * xx);
                        }
                    }

                    // effective input = [conv state | bx]; causal depthwise conv
                    let mut bx_ext = vec![0f32; (state_len + num_tokens) * dim];
                    if let Some(s) = state.as_ref() {
                        let st = &s.conv_state[conv_idx];
                        for j in 0..state_len {
                            for d in 0..dim {
                                bx_ext[j * dim + d] = st[d * state_len + j];
                            }
                        }
                    }
                    for (j, chunk) in bx.chunks_exact(dim).enumerate() {
                        for dd in 0..dim {
                            bx_ext[(state_len + j) * dim + dd] = chunk[dd];
                        }
                    }
                    let mut y = vec![0f32; num_tokens * dim];
                    for t in 0..num_tokens {
                        for d in 0..dim {
                            let mut acc = 0f32;
                            for k in 0..kernel {
                                acc += bx_ext[(t + k) * dim + d] * conv[d * kernel + k];
                            }
                            y[t * dim + d] = bf16_round(acc);
                        }
                    }
                    // roll state: keep the last `state_len` inputs
                    if let Some(s) = state.as_mut() {
                        let st = &mut s.conv_state[conv_idx];
                        let width = state_len + num_tokens;
                        for d in 0..dim {
                            for j in 0..state_len {
                                st[d * state_len + j] = bx_ext[(width - state_len + j) * dim + d];
                            }
                        }
                    }
                    conv_idx += 1;

                    // C * conv_out -> out_proj
                    for t in 0..num_tokens {
                        for d in 0..dim {
                            y[t * dim + d] = bf16_round(cvec[t * dim + d] * y[t * dim + d]);
                        }
                    }
                    linear_round(out_proj, &y, num_tokens)
                },
                Mixer::Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    g_proj,
                    q_norm,
                    k_norm,
                    sliding_window,
                } => {
                    let q = linear_round(q_proj, &h, num_tokens);
                    let k = linear_round(k_proj, &h, num_tokens);
                    let v = linear_round(v_proj, &h, num_tokens);

                    let mut qh = q.clone();
                    rms_norm_bf16(&mut qh, q_norm, eps);
                    let mut kh = k.clone();
                    rms_norm_bf16(&mut kh, k_norm, eps);

                    for t in 0..num_tokens {
                        for h_idx in 0..LiquidCodecConfig::LFM_HEADS {
                            apply_rope(&mut qh[t * dim + h_idx * LFM_HEAD_DIM..][..LFM_HEAD_DIM], &cos_t[t], &sin_t[t]);
                        }
                        for h_idx in 0..LiquidCodecConfig::LFM_KV_HEADS {
                            apply_rope(
                                &mut kh[t * LFM_KV_DIM + h_idx * LFM_HEAD_DIM..][..LFM_HEAD_DIM],
                                &cos_t[t],
                                &sin_t[t],
                            );
                        }
                    }

                    let total_past = state.as_ref().map(|s| s.keys[attn_idx].len() / LFM_KV_DIM).unwrap_or(0);
                    let mut keys_all = state.as_ref().map(|s| s.keys[attn_idx].clone()).unwrap_or_default();
                    let mut values_all = state.as_ref().map(|s| s.values[attn_idx].clone()).unwrap_or_default();
                    keys_all.extend_from_slice(&kh);
                    values_all.extend_from_slice(&v);
                    if let Some(s) = state.as_mut() {
                        s.keys[attn_idx] = keys_all.clone();
                        s.values[attn_idx] = values_all.clone();
                    }
                    let window = *sliding_window;

                    let mut attended = vec![0f32; num_tokens * dim];
                    let scale = 1.0 / (LFM_HEAD_DIM as f32).sqrt();
                    for t in 0..num_tokens {
                        let abs_pos = total_past + t;
                        let first_key = window.map(|w| abs_pos.saturating_sub(w - 1)).unwrap_or(0);
                        for h_idx in 0..LiquidCodecConfig::LFM_HEADS {
                            let kv_head = h_idx / (LiquidCodecConfig::LFM_HEADS / LiquidCodecConfig::LFM_KV_HEADS);
                            let q_row = &qh[t * dim + h_idx * LFM_HEAD_DIM..][..LFM_HEAD_DIM];
                            let n_keys = abs_pos - first_key + 1;
                            let mut scores = vec![0f32; n_keys];
                            for (j, s) in scores.iter_mut().enumerate() {
                                let kj = first_key + j;
                                let k_row = &keys_all[kj * LFM_KV_DIM + kv_head * LFM_HEAD_DIM..][..LFM_HEAD_DIM];
                                let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                                *s = bf16_round(bf16_round(dot) * scale);
                            }
                            let max_v = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0f32;
                            for s in &mut scores {
                                *s = (*s - max_v).exp();
                                sum += *s;
                            }
                            let inv = 1.0 / sum;
                            for s in &mut scores {
                                *s = bf16_round(*s * inv);
                            }
                            let mut ctx = [0f32; LFM_HEAD_DIM];
                            for (j, &p) in scores.iter().enumerate() {
                                let kj = first_key + j;
                                let v_row = &values_all[kj * LFM_KV_DIM + kv_head * LFM_HEAD_DIM..][..LFM_HEAD_DIM];
                                for (c, &vv) in ctx.iter_mut().zip(v_row) {
                                    *c += p * vv;
                                }
                            }
                            for c in &mut ctx {
                                *c = bf16_round(*c);
                            }
                            let dst = &mut attended[t * dim + h_idx * LFM_HEAD_DIM..][..LFM_HEAD_DIM];
                            dst.copy_from_slice(&ctx);
                        }
                    }

                    let gate = linear_round(g_proj, &h, num_tokens);
                    for t in 0..num_tokens {
                        for h_idx in 0..LiquidCodecConfig::LFM_HEADS {
                            let g = sigmoid_bf16(gate[t * LiquidCodecConfig::LFM_HEADS + h_idx]);
                            for d in 0..LFM_HEAD_DIM {
                                let idx = t * dim + h_idx * LFM_HEAD_DIM + d;
                                attended[idx] = bf16_round(attended[idx] * g);
                            }
                        }
                    }
                    attn_idx += 1;
                    linear_round(o_proj, &attended, num_tokens)
                },
            };

            let mut mixer_out = mixer_out;
            rms_norm_bf16(&mut mixer_out, &layer.post_operator_norm, eps);
            for (xi, mi) in x.iter_mut().zip(mixer_out) {
                *xi = bf16_round(*xi + mi);
            }

            // --- MLP half ---
            let mut h = x.clone();
            rms_norm_bf16(&mut h, &layer.ffn_norm, eps);
            let gate = linear_round(&layer.w1, &h, num_tokens);
            let up = linear_round(&layer.w3, &h, num_tokens);
            let mut m = vec![0f32; num_tokens * LiquidCodecConfig::LFM_INTERMEDIATE];
            for (m_v, (g, u)) in m.iter_mut().zip(gate.iter().zip(&up)) {
                *m_v = bf16_round(silu_bf16(*g) * u);
            }
            let mut down = linear_round(&layer.w2, &m, num_tokens);
            rms_norm_bf16(&mut down, &layer.post_ffn_norm, eps);
            for (xi, di) in x.iter_mut().zip(down) {
                *xi = bf16_round(*xi + di);
            }
        }

        rms_norm_bf16(&mut x, &lfm.embedding_norm, eps);
        x
    }
}

// ---------------------------------------------------------------------------
// Codec API
// ---------------------------------------------------------------------------

impl LiquidCodec {
    /// Audio (24 kHz, any length, tail zero-padded to whole frames) -> codes
    /// `[n_q][T]`.
    pub fn encode(
        &self,
        audio: &[f32],
    ) -> Result<Vec<Vec<i64>>> {
        ensure!(!audio.is_empty(), "audio must be non-empty");
        let frame_size = self.config.frame_size();
        let padded_len = audio.len().div_ceil(frame_size) * frame_size;
        let mut padded = vec![0f32; padded_len];
        padded[..audio.len()].copy_from_slice(audio);

        let features = self.stft_features(&padded);
        let num_frames = features.len() / (3 * self.config.bins());
        let hidden = self.frontend(&features, num_frames);
        let codec_frames = hidden.len() / self.config.encoder_dim;
        let encoded = self.lfm_forward(&self.encoder, &hidden, codec_frames, None);
        let latents = linear_round(&self.to_quantizer, &encoded, codec_frames);
        Ok(self.quantizer_encode(&latents, codec_frames))
    }

    /// Codes `[n_q][T]` -> audio of `T * frame_size` samples.
    pub fn decode(
        &self,
        codes: &[Vec<i64>],
    ) -> Result<Vec<f32>> {
        self.decode_impl(codes, None, true)
    }

    pub fn new_streaming_cache(&self) -> StreamingCache {
        let layers = LiquidCodecConfig::LFM_LAYERS;
        let conv_layers = layers - 3; // attention at 2, 4, 6
        let state_len = self.config.encoder_dim * (LiquidCodecConfig::LFM_CONV_KERNEL - 1);
        StreamingCache {
            run: Lfm3RunState {
                keys: vec![Vec::new(); 3],
                values: vec![Vec::new(); 3],
                conv_state: vec![vec![0f32; state_len]; conv_layers],
                seen_tokens: 0,
                audio_tail: None,
                envelope_tail: None,
                started: false,
                finished: false,
            },
        }
    }

    /// Decode one chunk. `codes` is `[n_q][T]` for this chunk's frames (may be
    /// empty when finalizing); until `finalize`, the trailing samples later
    /// frames still contribute to are held back in the cache.
    pub fn decode_streaming(
        &self,
        codes: &[Vec<i64>],
        cache: &mut StreamingCache,
        finalize: bool,
    ) -> Result<Vec<f32>> {
        ensure!(!cache.run.finished, "this stream is finalized; start with a new cache");
        self.decode_impl(codes, Some(&mut cache.run), finalize)
    }

    fn decode_impl(
        &self,
        codes: &[Vec<i64>],
        state: Option<&mut Lfm3RunState>,
        finalize: bool,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let trim = (cfg.stft_win_length - cfg.stft_hop_length) / 2;
        let tail = 2 * trim;
        let num_frames = codes.first().map(|c| c.len()).unwrap_or(0);
        for c in codes {
            ensure!(c.len() == num_frames, "ragged code chunks");
            for &code in c {
                ensure!((0..cfg.codebook_size()).contains(&code), "codes must be in [0, {})", cfg.codebook_size());
            }
        }

        let mut state = state;
        let started = state.as_ref().map(|s| s.started).unwrap_or(false);
        if num_frames == 0 && !(finalize && started) {
            if let Some(s) = state.as_mut()
                && finalize
            {
                s.finished = true;
            }
            return Ok(Vec::new());
        }

        let (mut audio, mut envelope) = if num_frames > 0 {
            let latents = self.quantizer_decode(codes, num_frames);
            let embeddings = linear_round(&self.from_quantizer, &latents, num_frames);
            // nearest-exact upsample: each codec frame repeats frame_stack times
            let up = cfg.stft_frame_stack;
            let mut seq = Vec::with_capacity(num_frames * up * cfg.encoder_dim);
            for row in embeddings.chunks_exact(cfg.encoder_dim) {
                for _ in 0..up {
                    seq.extend_from_slice(row);
                }
            }
            let seq_len = num_frames * up;
            let hidden = match state.as_mut() {
                Some(s) => {
                    // positions must start at the chunk's first unseen token, so
                    // advance the counter only after the forward pass
                    let out = self.lfm_forward(&self.decoder, &seq, seq_len, Some(s));
                    s.seen_tokens += seq_len;
                    out
                },
                None => self.lfm_forward(&self.decoder, &seq, seq_len, None),
            };
            let projected = linear_round(&self.dec_lin, &hidden, seq_len);
            let bins = cfg.bins();
            let mut specs = Vec::with_capacity(seq_len);
            let mut spec = vec![0f32; 2 * bins];
            for t in 0..seq_len {
                let row = &projected[t * 2 * bins..(t + 1) * 2 * bins];
                for b in 0..bins {
                    let (log_mag, angle) = (row[b], row[bins + b]);
                    let mag = log_mag.exp();
                    spec[b] = mag * angle.cos();
                    spec[bins + b] = mag * angle.sin();
                }
                specs.push(spec.clone());
            }
            self.istft(&specs)
        } else {
            (vec![0f32; tail], vec![0f32; tail])
        };

        match state.as_mut() {
            Some(s) => {
                // overlap-add with the held-back tail from the previous chunk
                if let (Some(prev_audio), Some(prev_env)) = (&s.audio_tail, &s.envelope_tail) {
                    for (i, (a, p)) in audio.iter_mut().zip(prev_audio.iter()).enumerate() {
                        if i < tail {
                            *a += *p;
                        }
                    }
                    for (i, (e, p)) in envelope.iter_mut().zip(prev_env.iter()).enumerate() {
                        if i < tail {
                            *e += *p;
                        }
                    }
                }
                if !s.started {
                    audio.drain(..trim);
                    envelope.drain(..trim);
                    s.started = true;
                }
                if finalize {
                    audio.truncate(audio.len() - trim);
                    envelope.truncate(envelope.len() - trim);
                    s.audio_tail = None;
                    s.envelope_tail = None;
                    s.finished = true;
                } else {
                    s.audio_tail = Some(audio[audio.len() - tail..].to_vec());
                    s.envelope_tail = Some(envelope[envelope.len() - tail..].to_vec());
                    audio.truncate(audio.len() - tail);
                    envelope.truncate(envelope.len() - tail);
                }
            },
            None => {
                audio.drain(..trim);
                audio.truncate(audio.len() - trim);
                envelope.drain(..trim);
                envelope.truncate(envelope.len() - trim);
            },
        }

        ensure!(audio.len() == envelope.len(), "audio/envelope length mismatch");
        Ok(audio.iter().zip(&envelope).map(|(a, e)| a / e).collect())
    }
}

impl StreamingCache {
    pub fn is_finished(&self) -> bool {
        self.run.finished
    }
}
