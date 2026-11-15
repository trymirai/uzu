//! Liquid-Codec on Metal.
//!
//! Mirrors the CPU reference in `liquid_codec.rs`; weights are parsed once by
//! the CPU path and uploaded as bf16 (`from_cpu`), like the vision-tower
//! precedent. All dense matmuls run on the generic Metal matmul kernel
//! (bf16 weights, f32 accumulate, bias epilogue); the codec-specific work is
//! done by the `kernel/liquid_codec` kernels:
//!
//! ```text
//! audio ──► CodecStft (window + FFT + features fused)
//!       ──► GEMM in_proj ──► CodecSilu ──► CodecPackWindow3 ──► GEMM k3 ──► CodecSilu
//!       ──► CodecPackFrames ──► GEMM k6 ──► LFM3 encoder ──► GEMM to_quantizer
//!       ──► CodecFsqEncode
//! codes ──► CodecFsqDecode ──► GEMMs ──► CodecPackRepeat ──► LFM3 decoder
//!       ──► GEMM dec_lin ──► CodecIstft (polar + irfft + window fused)
//!       ──► CodecOverlapAdd (host-side envelope normalisation)
//! ```
//!
//! LFM3 layers use per-layer fused tails: `CodecRmsNorm` (pre-norm),
//! `CodecQkvNormRope` (qkv norm + RoPE written straight into the KV cache),
//! `CodecAttention` (window-masked attention + per-head sigmoid gate),
//! `CodecConvMixer` (state carried across chunks via `CodecConvMixerState`),
//! `CodecSiluMul` (SwiGLU) and `CodecRmsNormResidual` (post-norm + residual
//! in one pass).

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use futures::executor::block_on;

use super::liquid_codec::{Lfm3, LiquidCodec, LiquidCodecConfig, Mixer};
use crate::{
    backends::{
        Backend,
        common::{
            BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, CpuBuffer, GlobalBufferCpuGpu, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
        },
        metal::{
            Metal,
            context::MetalContext,
            kernel::{
                CodecAttentionMetalKernel, CodecConvMixerMetalKernel, CodecConvMixerStateMetalKernel,
                CodecFsqDecodeMetalKernel, CodecFsqEncodeMetalKernel, CodecIstftMetalKernel,
                CodecOverlapAddMetalKernel, CodecPackFramesMetalKernel, CodecPackRepeatMetalKernel,
                CodecPackWindow3MetalKernel, CodecQkvNormRopeMetalKernel, CodecRmsNormMetalKernel,
                CodecRmsNormResidualMetalKernel, CodecSiluMetalKernel, CodecSiluMulMetalKernel, CodecStftMetalKernel,
            },
        },
    },
    data_type::{DataType, DataTypeRepr},
    vision::tower::LinearWeight,
};

const HEAD_DIM: u32 = 64;
const N_HEADS: u32 = 8;
const KV_HEADS: u32 = 2;
const N_FFT: u32 = 1280;
const HOP: u32 = 320;
const BINS: u32 = 641;
const TRIM: u32 = 480;
const TAIL: u32 = 960;
const FRAME_STACK: u32 = 6;
const DIM: u32 = 512;
/// KV cache capacity in 75 Hz tokens (six per codec frame).
const KV_CAPACITY: u32 = 640;

type GpuBuffer = <Metal as Backend>::GlobalGpuBuffer;
type Scratch = <Metal as Backend>::ScratchBuffer;
type Encoding = <<Metal as Backend>::CommandBuffer as CommandBuffer>::Encoding;
type MatmulK = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct MetalLinear {
    weights: GpuBuffer,
    bias: Option<GpuBuffer>,
    out_features: u32,
    in_features: u32,
}

enum MetalMixer {
    Conv {
        in_proj: MetalLinear,
        conv_weight: GpuBuffer,
        out_proj: MetalLinear,
    },
    Attention {
        /// fused [q | k | v | gate] projection, rows [q_heads*64 + 2*kv_heads*64 + q_heads, dim]
        qkv_gate: MetalLinear,
        o_proj: MetalLinear,
        q_norm: GpuBuffer,
        k_norm: GpuBuffer,
        sliding_window: u32,
    },
}

struct MetalLayer {
    mixer: MetalMixer,
    operator_norm: GpuBuffer,
    post_operator_norm: GpuBuffer,
    ffn_norm: GpuBuffer,
    post_ffn_norm: GpuBuffer,
    w1: MetalLinear,
    w2: MetalLinear,
    w3: MetalLinear,
}

struct MetalLfm3 {
    layers: Vec<MetalLayer>,
    embedding_norm: GpuBuffer,
}

struct Weights {
    config: LiquidCodecConfig,
    in_proj: MetalLinear,
    frontend_conv: MetalLinear,
    frontend_out: MetalLinear,
    encoder: MetalLfm3,
    decoder: MetalLfm3,
    to_quantizer: MetalLinear,
    quantizer_project_in: MetalLinear,
    quantizer_project_out: MetalLinear,
    from_quantizer: MetalLinear,
    dec_lin: MetalLinear,
    window: GpuBuffer,
    /// host copy of the Hann window (envelope computation)
    window_host: Vec<f32>,
    twiddle: GpuBuffer,
    rope_inv_freq: GpuBuffer,
    fsq_encode_params: GpuBuffer,
    fsq_decode_params: GpuBuffer,
    zero_conv_state: GpuBuffer,
    /// fold(hann²), host side: the overlap-add kernel stores the unnormalised
    /// sum so the streaming path can accumulate audio and envelope separately
    /// (the reference divides once, after summing).
    #[allow(dead_code)]
    envelope: Vec<f32>,
    envelope_recip: Vec<f32>,
}

struct CodecKernels {
    stft: CodecStftMetalKernel,
    istft: CodecIstftMetalKernel,
    overlap_add: CodecOverlapAddMetalKernel,
    pack_frames: CodecPackFramesMetalKernel,
    pack_window3: CodecPackWindow3MetalKernel,
    pack_repeat: CodecPackRepeatMetalKernel,
    fsq_encode: CodecFsqEncodeMetalKernel,
    fsq_decode: CodecFsqDecodeMetalKernel,
    rms_norm: CodecRmsNormMetalKernel,
    rms_norm_residual: CodecRmsNormResidualMetalKernel,
    silu: CodecSiluMetalKernel,
    silu_mul: CodecSiluMulMetalKernel,
    qkv_norm_rope: CodecQkvNormRopeMetalKernel,
    conv_mixer_state: CodecConvMixerStateMetalKernel,
    attention: CodecAttentionMetalKernel,
    conv_mixer: CodecConvMixerMetalKernel,
}

pub struct MetalStreamingState {
    keys: Vec<GpuBuffer>,
    values: Vec<GpuBuffer>,
    conv_states: Vec<GpuBuffer>,
    kv_len: u32,
    started: bool,
    finished: bool,
    audio_tail: Option<Vec<f32>>,
    envelope_tail: Option<Vec<f32>>,
}

pub struct MetalLiquidCodec {
    weights: Weights,
    context: std::sync::Arc<MetalContext>,
    matmul: MatmulK,
    kernels: CodecKernels,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

impl MetalLiquidCodec {
    pub fn is_available() -> bool {
        block_on(<Metal as Backend>::Context::new()).is_ok()
    }

    pub fn load_from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let cpu = LiquidCodec::load_from_model_dir(model_dir)?;
        Self::from_cpu(&cpu)
    }

    /// Uploads the CPU-parsed weights (bf16, lossless from the HF checkpoint).
    pub fn from_cpu(cpu: &LiquidCodec) -> Result<Self> {
        let context: std::sync::Arc<MetalContext> = block_on(<Metal as Backend>::Context::new())?;
        let bf16 = DataType::BF16;

        let linear = |linear: &LinearWeight| -> Result<MetalLinear> {
            Ok(MetalLinear {
                weights: upload_bf16(&context, &linear.weight)?,
                bias: match &linear.bias {
                    Some(b) => Some(upload_bf16(&context, b)?),
                    None => None,
                },
                out_features: linear.out_features as u32,
                in_features: linear.in_features as u32,
            })
        };

        let lfm3 = |lfm: &Lfm3| -> Result<MetalLfm3> {
            let layers = lfm
                .layers
                .iter()
                .map(|layer| -> Result<MetalLayer> {
                    let mixer = match &layer.mixer {
                        Mixer::Conv {
                            in_proj,
                            conv,
                            out_proj,
                        } => MetalMixer::Conv {
                            in_proj: linear(in_proj)?,
                            conv_weight: upload_f32(&context, conv)?,
                            out_proj: linear(out_proj)?,
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
                        } => MetalMixer::Attention {
                            qkv_gate: concat_rows(&context, &[g_proj, q_proj, k_proj, v_proj])?,
                            o_proj: linear(o_proj)?,
                            q_norm: upload_bf16(&context, q_norm)?,
                            k_norm: upload_bf16(&context, k_norm)?,
                            sliding_window: sliding_window.unwrap_or(0) as u32,
                        },
                    };
                    Ok(MetalLayer {
                        mixer,
                        operator_norm: upload_bf16(&context, &layer.operator_norm)?,
                        post_operator_norm: upload_bf16(&context, &layer.post_operator_norm)?,
                        ffn_norm: upload_bf16(&context, &layer.ffn_norm)?,
                        post_ffn_norm: upload_bf16(&context, &layer.post_ffn_norm)?,
                        w1: linear(&layer.w1)?,
                        w2: linear(&layer.w2)?,
                        w3: linear(&layer.w3)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(MetalLfm3 {
                layers,
                embedding_norm: upload_bf16(&context, &lfm.embedding_norm)?,
            })
        };

        let twiddle: Vec<f32> = (0..N_FFT)
            .flat_map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        let rope_inv_freq: Vec<f32> =
            (0..HEAD_DIM / 2).map(|i| 1e6f64.powf(-(2.0 * i as f64) / HEAD_DIM as f64) as f32).collect();
        let envelope = fold_envelope(&cpu.window);
        let envelope_recip = envelope
            .iter()
            .map(|&e| {
                if e > 0.0 {
                    1.0 / e
                } else {
                    0.0
                }
            })
            .collect();

        let config = cpu.config.clone();
        let weights = Weights {
            in_proj: linear(&cpu.in_proj)?,
            frontend_conv: linear(&cpu.frontend_conv)?,
            frontend_out: linear(&cpu.frontend_out)?,
            to_quantizer: linear(&cpu.to_quantizer)?,
            quantizer_project_in: linear(&cpu.quantizer_project_in)?,
            quantizer_project_out: linear(&cpu.quantizer_project_out)?,
            from_quantizer: linear(&cpu.from_quantizer)?,
            dec_lin: linear(&cpu.dec_lin)?,
            encoder: lfm3(&cpu.encoder)?,
            decoder: lfm3(&cpu.decoder)?,
            window_host: cpu.window.clone(),
            window: upload_f32(&context, &cpu.window)?,
            twiddle: upload_f32(&context, &twiddle)?,
            rope_inv_freq: upload_f32(&context, &rope_inv_freq)?,
            fsq_encode_params: upload_f32(&context, &fsq_encode_params(&config))?,
            fsq_decode_params: upload_i32(&context, &fsq_decode_params(&config))?,
            zero_conv_state: upload_f32(&context, &vec![0f32; DIM as usize * 6])?,
            envelope,
            envelope_recip,
            config,
        };

        let kernels = CodecKernels {
            stft: CodecStftMetalKernel::new(&context)?,
            istft: CodecIstftMetalKernel::new(&context)?,
            overlap_add: CodecOverlapAddMetalKernel::new(&context)?,
            pack_frames: CodecPackFramesMetalKernel::new(&context, bf16)?,
            pack_window3: CodecPackWindow3MetalKernel::new(&context, bf16)?,
            pack_repeat: CodecPackRepeatMetalKernel::new(&context, bf16)?,
            fsq_encode: CodecFsqEncodeMetalKernel::new(&context, bf16)?,
            fsq_decode: CodecFsqDecodeMetalKernel::new(&context)?,
            rms_norm: CodecRmsNormMetalKernel::new(&context)?,
            rms_norm_residual: CodecRmsNormResidualMetalKernel::new(&context)?,
            silu: CodecSiluMetalKernel::new(&context)?,
            silu_mul: CodecSiluMulMetalKernel::new(&context)?,
            qkv_norm_rope: CodecQkvNormRopeMetalKernel::new(&context)?,
            conv_mixer_state: CodecConvMixerStateMetalKernel::new(&context)?,
            attention: CodecAttentionMetalKernel::new(&context)?,
            conv_mixer: CodecConvMixerMetalKernel::new(&context)?,
        };

        Ok(Self {
            weights,
            matmul: MatmulK::new(&context, bf16, bf16, bf16)?,
            context,
            kernels,
        })
    }

    /// Matmul kernel handle for the throughput test's micro-benchmark.
    #[allow(dead_code)]
    pub(crate) fn matmul_bench(&mut self) -> &mut MatmulK {
        &mut self.matmul
    }

    /// Context handle for the throughput test's overhead measurement.
    pub fn context_for_bench(&self) -> &std::sync::Arc<MetalContext> {
        &self.context
    }

    /// Upload a small i32 buffer (CPU->GPU path cost).
    pub fn upload_bench(
        &self,
        data: &[i32],
    ) -> GpuBuffer {
        upload_i32(&self.context, data).expect("upload")
    }

    pub fn new_streaming_state(&self) -> Result<MetalStreamingState> {
        let bytes = KV_CAPACITY as usize * KV_HEADS as usize * HEAD_DIM as usize * 2;
        let kv = || -> Result<GpuBuffer> { upload::<u8>(&self.context, &vec![0u8; bytes]) };
        let conv = || -> Result<GpuBuffer> { upload_f32(&self.context, &vec![0f32; DIM as usize * 6]) };
        Ok(MetalStreamingState {
            keys: vec![kv()?, kv()?, kv()?],
            values: vec![kv()?, kv()?, kv()?],
            conv_states: vec![conv()?, conv()?, conv()?, conv()?, conv()?],
            kv_len: 0,
            started: false,
            finished: false,
            audio_tail: None,
            envelope_tail: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

/// Concatenate linear weights row-wise (matching `in_features`) into one buffer,
/// so a fused projection is a single GEMM.
fn concat_rows(
    context: &MetalContext,
    parts: &[&LinearWeight],
) -> Result<MetalLinear> {
    let out_features: usize = parts.iter().map(|p| p.out_features).sum();
    let in_features = parts[0].in_features;
    let mut weights = Vec::with_capacity(out_features * in_features);
    let mut bias = Vec::with_capacity(out_features);
    for p in parts {
        ensure!(p.in_features == in_features, "concat requires matching in_features");
        weights.extend_from_slice(&p.weight);
        // parts without a bias contribute zeros (attention has attention_bias=false)
        bias.extend_from_slice(p.bias.as_deref().unwrap_or(&vec![0f32; p.out_features]));
    }
    Ok(MetalLinear {
        weights: upload_bf16(context, &weights)?,
        bias: Some(upload_bf16(context, &bias)?),
        out_features: out_features as u32,
        in_features: in_features as u32,
    })
}

fn upload_bf16(
    context: &MetalContext,
    data: &[f32],
) -> Result<GpuBuffer> {
    let values: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
    upload::<half::bf16>(context, &values)
}

fn upload_f32(
    context: &MetalContext,
    data: &[f32],
) -> Result<GpuBuffer> {
    upload::<f32>(context, data)
}

fn upload_i32(
    context: &MetalContext,
    data: &[i32],
) -> Result<GpuBuffer> {
    upload::<i32>(context, data)
}

fn upload<T: DataTypeRepr>(
    context: &MetalContext,
    data: &[T],
) -> Result<GpuBuffer> {
    let mut buffer =
        context.create_buffer_cpu_gpu(std::mem::size_of_val(data)).context("allocating GPU buffer")?;
    buffer.copyin(data);
    buffer.handoff().map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn fsq_encode_params(config: &LiquidCodecConfig) -> Vec<f32> {
    let eps = 1e-3f32;
    let mut out = Vec::with_capacity(16);
    for &level in &config.fsq_levels {
        let half_levels = ((level - 1) as f32) * (1.0 + eps) / 2.0;
        let offset = if level % 2 == 0 {
            0.5
        } else {
            0.0
        };
        let half_width = (level / 2) as f32;
        out.extend_from_slice(&[half_levels, offset, half_width, (offset / half_levels).atanh()]);
    }
    out
}

fn fsq_decode_params(config: &LiquidCodecConfig) -> Vec<i32> {
    let mut out = Vec::with_capacity(16);
    let mut basis = 1i32;
    for &level in &config.fsq_levels {
        out.extend_from_slice(&[level as i32, basis, (level / 2) as i32, 0]);
        basis *= level as i32;
    }
    out
}

fn fold_envelope(window: &[f32]) -> Vec<f32> {
    let output_size = 4096 * HOP as usize;
    let mut envelope = vec![0f32; output_size];
    let frames = (output_size - N_FFT as usize) / HOP as usize + 1;
    for t in 0..frames {
        for j in 0..N_FFT as usize {
            envelope[t * HOP as usize + j] += window[j] * window[j];
        }
    }
    envelope
}

/// `fold(window^2)` over `seq_len` frames: the chunk-local envelope, which tapers
/// at the chunk edges (the tails of adjacent chunks sum to the global envelope).
fn chunk_envelope(
    window: &[f32],
    seq_len: u32,
) -> Vec<f32> {
    let output_size = (seq_len as usize - 1) * HOP as usize + N_FFT as usize;
    let mut envelope = vec![0f32; output_size];
    for t in 0..seq_len as usize {
        for j in 0..N_FFT as usize {
            envelope[t * HOP as usize + j] += window[j] * window[j];
        }
    }
    envelope
}

fn scratch(
    cb: &mut Encoding,
    shape: &[u32],
    data_type: DataType,
) -> Result<Scratch> {
    Ok(cb.allocate_scratch_for_shape(shape, data_type)?)
}

fn matmul_into(
    matmul: &mut MatmulK,
    linear: &MetalLinear,
    input: impl BufferRef<Backend = Metal>,
    m: u32,
    output: impl BufferMut<Backend = Metal>,
    cb: &mut Encoding,
) -> Result<()> {
    matmul.encode(
        MatmulArguments {
            a: MatmulA::FullPrecision {
                values: input,
                offset: 0,
            },
            b: MatmulB::FullPrecision {
                b: &linear.weights,
            },
            b_leading_dimension: None,
            b_transpose: true,
            output: MatmulOutput::new(
                output,
                MatmulDOps {
                    bias: linear.bias.as_ref(),
                    ..MatmulDOps::none()
                },
            ),
            gather_indices: None::<&GpuBuffer>,
            m,
            n: linear.out_features,
            k: linear.in_features,
        },
        cb,
    )?;
    Ok(())
}

fn matmul_scratch(
    matmul: &mut MatmulK,
    linear: &MetalLinear,
    input: impl BufferRef<Backend = Metal>,
    m: u32,
    cb: &mut Encoding,
) -> Result<Scratch> {
    let mut out = scratch(cb, &[m, linear.out_features], DataType::BF16)?;
    matmul_into(matmul, linear, input, m, &mut out, cb)?;
    Ok(out)
}

fn matmul_silu(
    matmul: &mut MatmulK,
    silu: &CodecSiluMetalKernel,
    linear: &MetalLinear,
    input: impl BufferRef<Backend = Metal>,
    m: u32,
    cb: &mut Encoding,
) -> Result<Scratch> {
    let out = matmul_scratch(matmul, linear, input, m, cb)?;
    let mut activated = scratch(cb, &[m, linear.out_features], DataType::BF16)?;
    silu.encode(&out, &mut activated, m * linear.out_features, cb);
    Ok(activated)
}

// ---------------------------------------------------------------------------
// LFM3 backbone
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn lfm3_forward(
    matmul: &mut MatmulK,
    kernels: &CodecKernels,
    weights: &Weights,
    lfm: &MetalLfm3,
    mut x: Scratch,
    tokens: u32,
    mut state: Option<&mut MetalStreamingState>,
    cb: &mut Encoding,
) -> Result<Scratch> {
    let pos_base = state.as_ref().map(|s| s.kv_len).unwrap_or(0);
    let mut attn_idx = 0usize;
    let mut conv_idx = 0usize;

    for layer in &lfm.layers {
        let mut h = scratch(cb, &[tokens, DIM], DataType::BF16)?;
        kernels.rms_norm.encode(&x, &layer.operator_norm, &mut h, tokens, DIM, 1e-5, cb);

        let mixer_out = match &layer.mixer {
            MetalMixer::Conv {
                in_proj,
                conv_weight,
                out_proj,
            } => {
                let bcx = matmul_scratch(matmul, in_proj, &h, tokens, cb)?;
                let mut gated = scratch(cb, &[tokens, DIM], DataType::BF16)?;
                match state.as_deref_mut() {
                    Some(s) => {
                        // single dispatch: conv output + rolled state (one thread per
                        // channel, so the state read/write cannot race)
                        kernels.conv_mixer_state.encode(
                            &bcx,
                            conv_weight,
                            &mut gated,
                            &mut s.conv_states[conv_idx],
                            tokens,
                            DIM,
                            6,
                            cb,
                        );
                    },
                    None => {
                        kernels.conv_mixer.encode(
                            &bcx,
                            conv_weight,
                            &mut gated,
                            &weights.zero_conv_state,
                            tokens,
                            DIM,
                            6,
                            cb,
                        );
                    },
                }
                conv_idx += 1;
                matmul_scratch(matmul, out_proj, &gated, tokens, cb)?
            },
            MetalMixer::Attention {
                qkv_gate,
                o_proj,
                q_norm,
                k_norm,
                sliding_window,
            } => {
                // one GEMM for q, k, v and the per-head gate
                let fused = matmul_scratch(matmul, qkv_gate, &h, tokens, cb)?;
                let mut q = scratch(cb, &[tokens, N_HEADS * HEAD_DIM], DataType::BF16)?;
                let mut aggregated = scratch(cb, &[tokens, N_HEADS * HEAD_DIM], DataType::BF16)?;
                // the gate heads live at the start of every fused row; the attention
                // kernel reads them with an explicit row stride
                let gate = &fused;

                match state.as_deref_mut() {
                    Some(s) => {
                        let kv_pos = s.kv_len;
                        kernels.qkv_norm_rope.encode(
                            &fused,
                            q_norm,
                            k_norm,
                            &weights.rope_inv_freq,
                            &mut q,
                            &mut s.keys[attn_idx],
                            &mut s.values[attn_idx],
                            tokens,
                            N_HEADS,
                            KV_HEADS,
                            qkv_gate.out_features,
                            N_HEADS,
                            kv_pos,
                            kv_pos,
                            1e-5,
                            pos_base,
                            cb,
                        );
                        kernels.attention.encode(
                            &q,
                            &s.keys[attn_idx],
                            &s.values[attn_idx],
                            gate,
                            &mut aggregated,
                            tokens,
                            N_HEADS,
                            KV_HEADS,
                            *sliding_window,
                            pos_base,
                            (HEAD_DIM as f32).sqrt().recip(),
                            qkv_gate.out_features,
                            cb,
                        );
                    },
                    None => {
                        let mut k = scratch(cb, &[tokens, KV_HEADS * HEAD_DIM], DataType::BF16)?;
                        let mut v = scratch(cb, &[tokens, KV_HEADS * HEAD_DIM], DataType::BF16)?;
                        kernels.qkv_norm_rope.encode(
                            &fused,
                            q_norm,
                            k_norm,
                            &weights.rope_inv_freq,
                            &mut q,
                            &mut k,
                            &mut v,
                            tokens,
                            N_HEADS,
                            KV_HEADS,
                            qkv_gate.out_features,
                            N_HEADS,
                            0,
                            0,
                            1e-5,
                            pos_base,
                            cb,
                        );
                        kernels.attention.encode(
                            &q,
                            &k,
                            &v,
                            gate,
                            &mut aggregated,
                            tokens,
                            N_HEADS,
                            KV_HEADS,
                            *sliding_window,
                            pos_base,
                            (HEAD_DIM as f32).sqrt().recip(),
                            qkv_gate.out_features,
                            cb,
                        );
                    },
                }
                attn_idx += 1;
                matmul_scratch(matmul, o_proj, &aggregated, tokens, cb)?
            },
        };

        let mut x_new = scratch(cb, &[tokens, DIM], DataType::BF16)?;
        kernels.rms_norm_residual.encode(&x, &mixer_out, &layer.post_operator_norm, &mut x_new, tokens, DIM, 1e-5, cb);

        let mut h = scratch(cb, &[tokens, DIM], DataType::BF16)?;
        kernels.rms_norm.encode(&x_new, &layer.ffn_norm, &mut h, tokens, DIM, 1e-5, cb);
        let gate = matmul_scratch(matmul, &layer.w1, &h, tokens, cb)?;
        let up = matmul_scratch(matmul, &layer.w3, &h, tokens, cb)?;
        let mut activated = scratch(cb, &[tokens, 2304], DataType::BF16)?;
        kernels.silu_mul.encode(&gate, &up, &mut activated, tokens * 2304, cb);
        let down = matmul_scratch(matmul, &layer.w2, &activated, tokens, cb)?;
        let mut x_final = scratch(cb, &[tokens, DIM], DataType::BF16)?;
        kernels.rms_norm_residual.encode(&x_new, &down, &layer.post_ffn_norm, &mut x_final, tokens, DIM, 1e-5, cb);
        x = x_final;
    }

    let mut out = scratch(cb, &[tokens, DIM], DataType::BF16)?;
    kernels.rms_norm.encode(&x, &lfm.embedding_norm, &mut out, tokens, DIM, 1e-5, cb);
    if let Some(s) = state.as_mut() {
        s.kv_len += tokens;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

impl MetalLiquidCodec {
    pub fn encode(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<Vec<i64>>> {
        let w = &self.weights;
        let frame_size = w.config.frame_size();
        let padded_len = audio.len().div_ceil(frame_size) * frame_size;
        let mut padded = vec![0f32; padded_len];
        padded[..audio.len()].copy_from_slice(audio);

        // reflect padding (torch semantics: edges excluded)
        let left = TRIM as usize;
        let right = (N_FFT - HOP) as usize - left;
        let n = padded.len();
        let mut framed = vec![0f32; n + left + right];
        framed[left..left + n].copy_from_slice(&padded);
        for j in 0..left {
            framed[j] = padded[left - j];
        }
        for j in 0..right {
            framed[left + n + j] = padded[n - 2 - j];
        }
        let stft_frames = ((framed.len() as u32 - N_FFT) / HOP + 1) as u32;
        let codec_frames = stft_frames / FRAME_STACK;
        ensure!(stft_frames == codec_frames * FRAME_STACK, "ragged frame stack");

        let mut cb = self.context.create_command_buffer(None, None)?;
        let audio_buf = upload_f32(&self.context, &framed)?;

        let mut features = scratch(&mut cb, &[stft_frames, 3 * BINS], DataType::BF16)?;
        self.kernels.stft.encode(&audio_buf, &w.window, &w.twiddle, &mut features, stft_frames, &mut cb);

        let a1 = matmul_silu(&mut self.matmul, &self.kernels.silu, &w.in_proj, &features, stft_frames, &mut cb)?;
        let mut packed3 = scratch(&mut cb, &[stft_frames, 3 * 256], DataType::BF16)?;
        self.kernels.pack_window3.encode(&a1, &mut packed3, stft_frames, 256, &mut cb);
        let a2 = matmul_silu(&mut self.matmul, &self.kernels.silu, &w.frontend_conv, &packed3, stft_frames, &mut cb)?;

        let mut packed6 = scratch(&mut cb, &[codec_frames, 6 * 256], DataType::BF16)?;
        self.kernels.pack_frames.encode(&a2, &mut packed6, stft_frames, 256, &mut cb);
        let hidden = matmul_scratch(&mut self.matmul, &w.frontend_out, &packed6, codec_frames, &mut cb)?;

        let encoded =
            lfm3_forward(&mut self.matmul, &self.kernels, w, &w.encoder, hidden, codec_frames, None, &mut cb)?;
        let latents = matmul_scratch(&mut self.matmul, &w.to_quantizer, &encoded, codec_frames, &mut cb)?;

        let projected = matmul_scratch(&mut self.matmul, &w.quantizer_project_in, &latents, codec_frames, &mut cb)?;
        let mut indices = scratch(&mut cb, &[12, codec_frames], DataType::I32)?;
        self.kernels.fsq_encode.encode(&projected, &w.fsq_encode_params, &mut indices, codec_frames, 12, &mut cb);

        let mut out = self.context.create_buffer_cpu_gpu(12 * codec_frames as usize * 4)?;
        cb.encode_copy(&indices, &mut out);
        block_on(cb.end_encoding().submit().wait_until_completed())?;
        let flat: Vec<i32> = block_on(self.context.copyout(&out))?;
        let frames = codec_frames as usize;
        Ok((0..12).map(|cb| flat[cb * frames..(cb + 1) * frames].iter().map(|&v| v as i64).collect()).collect())
    }

    pub fn config(&self) -> &LiquidCodecConfig {
        &self.weights.config
    }

    pub fn decode(
        &mut self,
        codes: &[Vec<i64>],
    ) -> Result<Vec<f32>> {
        let (audio, envelope) = self.decode_chunk(codes, None)?;
        let trim = TRIM as usize;
        Ok(audio[trim..audio.len() - trim]
            .iter()
            .zip(&envelope[trim..])
            .map(|(&a, &e)| {
                if e > 0.0 {
                    a / e
                } else {
                    0.0
                }
            })
            .collect())
    }

    pub fn decode_streaming(
        &mut self,
        codes: &[Vec<i64>],
        state: &mut MetalStreamingState,
        finalize: bool,
    ) -> Result<Vec<f32>> {
        ensure!(!state.finished, "this stream is finalized; start with a new state");
        let frames = codes.first().map(|c| c.len()).unwrap_or(0) as u32;
        // Global sample offset of this chunk's first sample; chunk-local
        // envelope slices are aligned to it.
        let base_offset = state.kv_len as usize * HOP as usize;
        let started = state.started;
        if frames == 0 && !(finalize && started) {
            if finalize {
                state.finished = true;
            }
            return Ok(Vec::new());
        }

        let (mut audio, mut envelope) = if frames > 0 {
            self.decode_chunk(codes, Some(state))?
        } else {
            (vec![0f32; TAIL as usize], vec![0f32; TAIL as usize])
        };

        let tail = TAIL as usize;
        let trim = TRIM as usize;
        if let (Some(prev), Some(prev_env)) = (&state.audio_tail, &state.envelope_tail) {
            for (i, p) in prev.iter().enumerate() {
                audio[i] += p;
            }
            for (i, p) in prev_env.iter().enumerate() {
                envelope[i] += p;
            }
        }
        let front_trim = if started {
            0
        } else {
            trim
        };

        if !started {
            audio.drain(..trim);
            envelope.drain(..trim);
            state.started = true;
        }

        let normalize = |audio: &[f32], envelope: &[f32], global_offset: usize| -> Vec<f32> {
            audio
                .iter()
                .zip(envelope)
                .enumerate()
                .map(|(i, (&a, &e))| {
                    if e > 0.0 {
                        a / e
                    } else {
                        a * self.weights.envelope_recip[global_offset + i]
                    }
                })
                .collect()
        };

        if finalize {
            audio.truncate(audio.len() - trim);
            envelope.truncate(envelope.len() - trim);
            state.audio_tail = None;
            state.envelope_tail = None;
            state.finished = true;
            Ok(normalize(&audio, &envelope, base_offset + front_trim))
        } else {
            state.audio_tail = Some(audio[audio.len() - tail..].to_vec());
            state.envelope_tail = Some(envelope[envelope.len() - tail..].to_vec());
            audio.truncate(audio.len() - tail);
            envelope.truncate(envelope.len() - tail);
            Ok(normalize(&audio, &envelope, base_offset + front_trim))
        }
    }

    /// One decode chunk: returns the un-trimmed overlap-added audio and its
    /// envelope, both in chunk-local sample coordinates.
    fn decode_chunk(
        &mut self,
        codes: &[Vec<i64>],
        state: Option<&mut MetalStreamingState>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(codes.len() == 12, "expected 12 codebooks");
        let frames = codes.first().map(|c| c.len()).unwrap_or(0) as u32;
        ensure!(frames > 0, "decode needs at least one frame");
        for row in codes {
            ensure!(row.len() == frames as usize, "ragged codes");
        }

        let w = &self.weights;
        let flat_codes: Vec<i32> = (0..12).flat_map(|cb| codes[cb].iter().map(|&v| v as i32)).collect();
        let codes_buf = upload_i32(&self.context, &flat_codes)?;

        let mut cb = self.context.create_command_buffer(None, None)?;
        let mut latents48 = scratch(&mut cb, &[frames, 48], DataType::BF16)?;
        self.kernels.fsq_decode.encode(&codes_buf, &w.fsq_decode_params, &mut latents48, frames, 12, &mut cb);
        let latents = matmul_scratch(&mut self.matmul, &w.quantizer_project_out, &latents48, frames, &mut cb).unwrap();
        let embeddings = matmul_scratch(&mut self.matmul, &w.from_quantizer, &latents, frames, &mut cb).unwrap();

        let seq_len = frames * FRAME_STACK;
        let mut seq = scratch(&mut cb, &[seq_len, DIM], DataType::BF16)?;
        self.kernels.pack_repeat.encode(&embeddings, &mut seq, frames, DIM, &mut cb);

        let hidden =
            lfm3_forward(&mut self.matmul, &self.kernels, w, &w.decoder, seq, seq_len, state, &mut cb)?;
        let projected = matmul_scratch(&mut self.matmul, &w.dec_lin, &hidden, seq_len, &mut cb).unwrap();

        let mut frames_out = scratch(&mut cb, &[seq_len, N_FFT], DataType::F32)?;
        self.kernels.istft.encode(&projected, &w.window, &w.twiddle, &mut frames_out, seq_len, 2 * BINS, &mut cb);
        let output_size = (seq_len - 1) * HOP + N_FFT;
        let mut audio = scratch(&mut cb, &[output_size], DataType::F32)?;
        self.kernels.overlap_add.encode(&frames_out, &mut audio, seq_len, &mut cb);

        let mut out = self.context.create_buffer_cpu_gpu(output_size as usize * 4)?;
        cb.encode_copy(&audio, &mut out);
        block_on(cb.end_encoding().submit().wait_until_completed())?;
        let audio: Vec<f32> = block_on(self.context.copyout(&out))?;

        // per-chunk envelope: fold(window^2) over this chunk's own frames only
        let envelope = chunk_envelope(&w.window_host, seq_len);
        Ok((audio, envelope))
    }
}
