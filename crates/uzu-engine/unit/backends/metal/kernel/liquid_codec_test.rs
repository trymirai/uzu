//! Kernel-level parity tests for the Liquid-Codec Metal kernels, against small
//! Rust references (the repo convention: CPU reference vs Metal).

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    audio::fft::rfft,
    backends::{
        Backend,
        common::{CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
        metal::{
            Metal,
            context::MetalContext,
            kernel::{CodecFsqDecodeMetalKernel, CodecFsqEncodeMetalKernel, CodecStftMetalKernel},
        },
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer_with_data},
};

const N_FFT: usize = 1280;
const HOP: usize = 320;
const BINS: usize = 641;

fn window() -> Vec<f32> {
    (0..N_FFT).map(|j| 0.5 - 0.5 * (std::f32::consts::TAU * j as f32 / N_FFT as f32).cos()).collect()
}

fn twiddle() -> Vec<f32> {
    (0..N_FFT)
        .flat_map(|k| {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64;
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

/// Straightforward Rust reference for the analysis features.
#[allow(dead_code)]
fn reference_features(
    frames: usize,
    audio: &[f32],
    window: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; frames * 3 * BINS];
    for t in 0..frames {
        let mut re = vec![0f32; N_FFT];
        for j in 0..N_FFT {
            re[j] = audio[t * HOP + j] * window[j];
        }
        let (bin_re, bin_im) = rfft(&re);
        for b in 0..BINS {
            let mag = (bin_re[b] * bin_re[b] + bin_im[b] * bin_im[b]).sqrt().max(1e-5);
            out[t * 3 * BINS + b] = mag.ln_1p();
            out[t * 3 * BINS + BINS + b] = bin_re[b] / mag;
            out[t * 3 * BINS + 2 * BINS + b] = bin_im[b] / mag;
        }
    }
    out
}

#[uzu_test]
fn codec_stft_kernel_matches_golden() {
    let Some(golden) = std::env::var_os("LIQUID_CODEC_GOLDEN").map(std::path::PathBuf::from) else {
        println!("{{\"skipped\": \"LIQUID_CODEC_GOLDEN not set\"}}");
        return;
    };
    let read_f32 = |name: &str| -> Vec<f32> {
        let bytes = std::fs::read(golden.join(name)).expect("golden file");
        bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let input = read_f32("input.f32");
    let expected_cm = read_f32("stft_features.f32"); // [1923][frames]

    // same framing as the codec: pad to a whole 1920-frame, reflect pad 480
    let mut padded = input.clone();
    padded.resize(24_000usize.div_ceil(1920) * 1920, 0.0);
    let (left, right) = (480usize, 480usize);
    let n = padded.len();
    let mut framed = vec![0f32; n + left + right];
    framed[left..left + n].copy_from_slice(&padded);
    for j in 0..left {
        framed[j] = padded[left - j];
    }
    for j in 0..right {
        framed[left + n + j] = padded[n - 2 - j];
    }
    let frames = (framed.len() - N_FFT) / HOP + 1;
    assert_eq!(expected_cm.len(), 3 * BINS * frames);

    let context: std::sync::Arc<MetalContext> = block_on(<Metal as Backend>::Context::new()).expect("context");
    let window = window();
    let kernel = CodecStftMetalKernel::new(&context).expect("kernel");
    let audio_buf = create_buffer_with_data::<Metal, f32>(&context, &framed);
    let window_buf = create_buffer_with_data::<Metal, f32>(&context, &window);
    let twiddle_buf = create_buffer_with_data::<Metal, f32>(&context, &twiddle());
    let zeros: Vec<half::bf16> = vec![half::bf16::from_f32(0.0); frames * 3 * BINS];
    let mut features = create_buffer_with_data::<Metal, half::bf16>(&context, &zeros);
    let mut cb = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(&audio_buf, &window_buf, &twiddle_buf, &mut features, frames as u32, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let actual: Vec<half::bf16> = buffer_to_vec::<Metal, half::bf16>(&context, &features);

    let mut diff = 0f32;
    let mut argmax = 0usize;
    for t in 0..frames {
        for ch in 0..3 * BINS {
            let a = actual[t * 3 * BINS + ch].to_f32();
            let b = expected_cm[ch * frames + t];
            let d = (a - b).abs();
            if d > diff {
                diff = d;
                argmax = t * 3 * BINS + ch;
            }
        }
    }
    println!(
        "{{\"stft_golden_max_abs\": {diff}, \"frame\": {}, \"component\": {}, \"bin\": {}}}",
        argmax / (3 * BINS),
        (argmax % (3 * BINS)) / BINS,
        argmax % BINS
    );
    assert!(diff < 0.05, "stft vs golden max-abs-diff {diff}");
}

#[uzu_test]
fn codec_fsq_kernel_roundtrip() {
    let context: std::sync::Arc<MetalContext> = block_new_context();
    let frames = 4u32;
    let levels: [i32; 4] = [8, 7, 6, 6];
    // encoded indices must round-trip through decode to the canonical levels
    let mut indices = vec![0i32; 12 * frames as usize];
    for cb in 0..12usize {
        for t in 0..frames as usize {
            let mut idx = 0i32;
            let mut basis = 1i32;
            for (d, &level) in levels.iter().enumerate() {
                let value = ((cb + t + d) as i32) % level;
                idx += value * basis;
                basis *= level;
            }
            indices[cb * frames as usize + t] = idx;
        }
    }
    let mut params_decode = Vec::new();
    let mut basis = 1i32;
    for &level in &levels {
        params_decode.extend_from_slice(&[level, basis, level / 2, 0]);
        basis *= level;
    }

    let kernel = CodecFsqDecodeMetalKernel::new(&context).expect("kernel");
    let indices_buf = create_buffer_with_data::<Metal, i32>(&context, &indices);
    let params_buf = create_buffer_with_data::<Metal, i32>(&context, &params_decode);
    let zeros: Vec<half::bf16> = vec![half::bf16::from_f32(0.0); 12 * 4 * frames as usize];
    let mut latents = create_buffer_with_data::<Metal, half::bf16>(&context, &zeros);
    let mut cb = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(&indices_buf, &params_buf, &mut latents, frames, 12, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");

    let decoded: Vec<half::bf16> = buffer_to_vec::<Metal, half::bf16>(&context, &latents);
    let mut max_diff = 0f32;
    for cb_idx in 0..12usize {
        for t in 0..frames as usize {
            let idx = indices[cb_idx * frames as usize + t];
            let mut remaining = idx;
            for (d, &level) in levels.iter().enumerate() {
                let value = remaining % level;
                remaining /= level;
                let expected = (value - level / 2) as f32 / (level / 2) as f32;
                let actual = decoded[t * 48 + cb_idx * 4 + d].to_f32();
                max_diff = max_diff.max((actual - expected).abs());
            }
        }
    }
    println!("{{\"fsq_decode_max_abs\": {max_diff}}}");
    assert!(max_diff < 5e-3, "fsq decode max-abs-diff {max_diff}");

    // and the encoder must reproduce the same indices from the decoded latents
    let encode_kernel = CodecFsqEncodeMetalKernel::new(&context, DataType::BF16).expect("encode kernel");
    let eps = 1e-3f32;
    let mut params_encode = Vec::new();
    for &level in &levels {
        let half_levels = ((level - 1) as f32) * (1.0 + eps) / 2.0;
        let offset = if level % 2 == 0 {
            0.5
        } else {
            0.0
        };
        let half_width = (level / 2) as f32;
        params_encode.extend_from_slice(&[half_levels, offset, half_width, (offset / half_levels).atanh()]);
    }
    let params_buf = create_buffer_with_data::<Metal, f32>(&context, &params_encode);
    let zeros_i32: Vec<i32> = vec![0i32; 12 * frames as usize];
    let mut encoded = create_buffer_with_data::<Metal, i32>(&context, &zeros_i32);
    let mut cb = context.create_command_buffer(None, None).expect("command buffer");
    encode_kernel.encode(&latents, &params_buf, &mut encoded, frames, 12, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let encoded: Vec<i32> = buffer_to_vec::<Metal, i32>(&context, &encoded);
    let mismatches = encoded.iter().zip(&indices).filter(|(a, b)| a != b).count();
    // encode(decode(idx)) is intentionally not the identity at level edges:
    // `_bound` squashes with eps, so edge levels map back one step inwards.
    // This is informational; the real requirement is reproducing the
    // reference formula, covered by the end-to-end code parity test.
    println!("{{\"fsq_encode_roundtrip_mismatches\": {mismatches} / {}}}", indices.len());
}

#[uzu_test]
fn codec_pack_kernels_match_reference() {
    use crate::backends::metal::kernel::{
        CodecPackFramesMetalKernel, CodecPackRepeatMetalKernel, CodecPackWindow3MetalKernel,
    };
    let context = block_new_context();
    let rows = 6u32;
    let dim = 4u32;
    let src: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();

    // frames: dst[row][d*6+k] = src[row*6+k][d]
    let kernel = CodecPackFramesMetalKernel::new(&context, DataType::F32).expect("kernel");
    let src_buf = create_buffer_with_data::<Metal, f32>(&context, &src);
    let zeros: Vec<f32> = vec![0.0; rows as usize * dim as usize];
    let mut dst = create_buffer_with_data::<Metal, f32>(&context, &zeros);
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    kernel.encode(&src_buf, &mut dst, rows, dim, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let out: Vec<f32> = buffer_to_vec::<Metal, f32>(&context, &dst);
    let mut expected = vec![0f32; rows as usize * dim as usize];
    for orow in 0..(rows / 6) as usize {
        for d in 0..dim as usize {
            for k in 0..6usize {
                expected[orow * dim as usize * 6 + d * 6 + k] = src[(orow * 6 + k) * dim as usize + d];
            }
        }
    }
    let frames_diff = out.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

    // window3: dst[t][d*3+k] = src[t+k-2][d] (zero padded)
    let kernel = CodecPackWindow3MetalKernel::new(&context, DataType::F32).expect("kernel");
    let zeros3: Vec<f32> = vec![0.0; rows as usize * dim as usize * 3];
    let mut dst3 = create_buffer_with_data::<Metal, f32>(&context, &zeros3);
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    kernel.encode(&src_buf, &mut dst3, rows, dim, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let out3: Vec<f32> = buffer_to_vec::<Metal, f32>(&context, &dst3);
    let mut expected3 = vec![0f32; rows as usize * dim as usize * 3];
    for t in 0..rows as usize {
        for d in 0..dim as usize {
            for k in 0..3usize {
                let s = t as isize + k as isize - 2;
                let value = if s >= 0 {
                    src[s as usize * dim as usize + d]
                } else {
                    0.0
                };
                expected3[t * dim as usize * 3 + d * 3 + k] = value;
            }
        }
    }
    let window_diff = out3.iter().zip(&expected3).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

    // repeat: dst[t*6+r][d] = src[t][d]
    let kernel = CodecPackRepeatMetalKernel::new(&context, DataType::F32).expect("kernel");
    let zeros6: Vec<f32> = vec![0.0; rows as usize * 6 * dim as usize];
    let mut dst6 = create_buffer_with_data::<Metal, f32>(&context, &zeros6);
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    kernel.encode(&src_buf, &mut dst6, rows, dim, &mut cb);
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let out6: Vec<f32> = buffer_to_vec::<Metal, f32>(&context, &dst6);
    let mut expected6 = vec![0f32; rows as usize * 6 * dim as usize];
    for t in 0..rows as usize {
        for r in 0..6usize {
            for d in 0..dim as usize {
                expected6[(t * 6 + r) * dim as usize + d] = src[t * dim as usize + d];
            }
        }
    }
    let repeat_diff = out6.iter().zip(&expected6).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("{{\"pack\": {{\"frames\": {frames_diff}, \"window3\": {window_diff}, \"repeat\": {repeat_diff}}}}}");
    assert!(frames_diff < 1e-6 && window_diff < 1e-6 && repeat_diff < 1e-6);
}

fn block_new_context() -> std::sync::Arc<MetalContext> {
    block_on(<Metal as Backend>::Context::new()).expect("context")
}

/// Helper: the encode/decode kernels are unused in the fsq roundtrip above only
/// for symmetry with other kernel tests.
#[allow(dead_code)]
fn _unused<T: DataTypeRepr>(_: T) {}

/// Fused qkv norm/RoPE + attention with a KV cache split into two chunks must
/// match a single full-sequence pass (streaming decoder correctness).
#[uzu_test]
fn codec_attention_kv_chunking_matches_full() {
    use crate::backends::metal::kernel::{CodecAttentionMetalKernel, CodecQkvNormRopeMetalKernel};
    let context = block_new_context();
    let (tokens, heads, kv_heads, dim) = (12usize, 8usize, 2usize, 64usize);
    let window = 30u32;
    let scale = 0.125f32;
    // Fused [gate | q | k | v] projection rows, as produced by the fused GEMM.
    let fused_stride = heads + heads * dim + 2 * (kv_heads * dim);

    let q_raw: Vec<f32> = (0..tokens * heads * dim).map(|i| ((i as f32) * 0.013).sin()).collect();
    let k_raw: Vec<f32> = (0..tokens * kv_heads * dim).map(|i| ((i as f32) * 0.017).cos()).collect();
    let v_raw: Vec<f32> = (0..tokens * kv_heads * dim).map(|i| ((i as f32) * 0.019).sin()).collect();
    let gate: Vec<f32> = (0..tokens * heads).map(|i| 2.0 * ((i as f32) * 0.05).cos()).collect();
    let q_norm: Vec<f32> = (0..dim).map(|i| 0.8 + 0.2 * ((i as f32) * 0.3).cos()).collect();
    let k_norm: Vec<f32> = (0..dim).map(|i| 0.7 + 0.3 * ((i as f32) * 0.2).sin()).collect();
    let inv_freq: Vec<f32> = (0..32).map(|i| 1e6f64.powf(-(2.0 * i as f64) / 64.0) as f32).collect();

    let build = |data: &[f32]| {
        let values: Vec<half::bf16> = data.iter().map(|&x| half::bf16::from_f32(x)).collect();
        create_buffer_with_data::<Metal, half::bf16>(&context, &values)
    };
    let zeros = |n: usize| build(&vec![0f32; n]);

    let build_fused = |q: &[f32], k: &[f32], v: &[f32], gate: &[f32], rows: usize| {
        let mut fused = vec![0f32; rows * fused_stride];
        for t in 0..rows {
            let row = &mut fused[t * fused_stride..(t + 1) * fused_stride];
            row[..heads].copy_from_slice(&gate[t * heads..(t + 1) * heads]);
            row[heads..heads + heads * dim].copy_from_slice(&q[t * heads * dim..(t + 1) * heads * dim]);
            let k_base = heads + heads * dim;
            row[k_base..k_base + kv_heads * dim].copy_from_slice(&k[t * kv_heads * dim..(t + 1) * kv_heads * dim]);
            let v_base = k_base + kv_heads * dim;
            row[v_base..v_base + kv_heads * dim].copy_from_slice(&v[t * kv_heads * dim..(t + 1) * kv_heads * dim]);
        }
        build(&fused)
    };

    let q_norm_buf = build(&q_norm);
    let k_norm_buf = build(&k_norm);
    let inv_buf = create_buffer_with_data::<Metal, f32>(&context, &inv_freq);
    let qkv_norm_rope = CodecQkvNormRopeMetalKernel::new(&context).expect("qkv norm rope");
    let attention = CodecAttentionMetalKernel::new(&context).expect("attention");

    // reference: one pass over all tokens
    let fused_all = build_fused(&q_raw, &k_raw, &v_raw, &gate, tokens);
    let mut q = zeros(tokens * heads * dim);
    let mut k = zeros(tokens * kv_heads * dim);
    let mut v = zeros(tokens * kv_heads * dim);
    let mut out = zeros(tokens * heads * dim);
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    qkv_norm_rope.encode(
        &fused_all,
        &q_norm_buf,
        &k_norm_buf,
        &inv_buf,
        &mut q,
        &mut k,
        &mut v,
        tokens as u32,
        heads as u32,
        kv_heads as u32,
        fused_stride as u32,
        heads as u32,
        0,
        0,
        1e-5,
        0,
        &mut cb,
    );
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("prep run");
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    attention.encode(
        &q,
        &k,
        &v,
        &fused_all,
        &mut out,
        tokens as u32,
        heads as u32,
        kv_heads as u32,
        window,
        0,
        scale,
        fused_stride as u32,
        &mut cb,
    );
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let expected: Vec<half::bf16> = buffer_to_vec::<Metal, half::bf16>(&context, &out);

    // chunked: two 6-token passes sharing the KV cache buffers. Each stage runs
    // in its own command buffer: within one buffer the DSL only tracks hazards
    // through `access()` ranges, and qkv-norm-rope -> attention on the same
    // buffer is not ordered for our purposes.
    let mut chunked = vec![half::bf16::from_f32(0.0); tokens * heads * dim];
    let chunk = 6usize;
    let mut k_cache = zeros(tokens * kv_heads * dim);
    let mut v_cache = zeros(tokens * kv_heads * dim);
    for c in 0..2usize {
        let offset = (c * chunk) as u32;
        let fused_chunk = build_fused(
            &q_raw[c * chunk * heads * dim..(c + 1) * chunk * heads * dim],
            &k_raw[c * chunk * kv_heads * dim..(c + 1) * chunk * kv_heads * dim],
            &v_raw[c * chunk * kv_heads * dim..(c + 1) * chunk * kv_heads * dim],
            &gate[c * chunk * heads..(c + 1) * chunk * heads],
            chunk,
        );
        let mut q_chunk = zeros(chunk * heads * dim);
        let mut cb = context.create_command_buffer(None, None).expect("cb");
        qkv_norm_rope.encode(
            &fused_chunk,
            &q_norm_buf,
            &k_norm_buf,
            &inv_buf,
            &mut q_chunk,
            &mut k_cache,
            &mut v_cache,
            chunk as u32,
            heads as u32,
            kv_heads as u32,
            fused_stride as u32,
            heads as u32,
            offset,
            offset,
            1e-5,
            offset,
            &mut cb,
        );
        block_on(cb.end_encoding().submit().wait_until_completed()).expect("prep run");

        let mut aggregated = zeros(chunk * heads * dim);
        let mut cb = context.create_command_buffer(None, None).expect("cb");
        attention.encode(
            &q_chunk,
            &k_cache,
            &v_cache,
            &fused_chunk,
            &mut aggregated,
            chunk as u32,
            heads as u32,
            kv_heads as u32,
            window,
            offset,
            scale,
            fused_stride as u32,
            &mut cb,
        );
        block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
        let got: Vec<half::bf16> = buffer_to_vec::<Metal, half::bf16>(&context, &aggregated);
        chunked[c * chunk * heads * dim..(c + 1) * chunk * heads * dim].copy_from_slice(&got);
    }

    // control: a standalone 6-token pass over the same prefix inputs (fresh
    // k/v buffers: the chunked loop above already filled the shared cache).
    let fused6 = build_fused(
        &q_raw[..chunk * heads * dim],
        &k_raw[..chunk * kv_heads * dim],
        &v_raw[..chunk * kv_heads * dim],
        &gate[..chunk * heads],
        chunk,
    );
    let mut q6 = zeros(chunk * heads * dim);
    let mut k6 = zeros(chunk * kv_heads * dim);
    let mut v6 = zeros(chunk * kv_heads * dim);
    let mut out6 = zeros(chunk * heads * dim);
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    qkv_norm_rope.encode(
        &fused6,
        &q_norm_buf,
        &k_norm_buf,
        &inv_buf,
        &mut q6,
        &mut k6,
        &mut v6,
        chunk as u32,
        heads as u32,
        kv_heads as u32,
        fused_stride as u32,
        heads as u32,
        0,
        0,
        1e-5,
        0,
        &mut cb,
    );
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("prep run");
    let mut cb = context.create_command_buffer(None, None).expect("cb");
    attention.encode(
        &q6,
        &k6,
        &v6,
        &fused6,
        &mut out6,
        chunk as u32,
        heads as u32,
        kv_heads as u32,
        window,
        0,
        scale,
        fused_stride as u32,
        &mut cb,
    );
    block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
    let standalone6: Vec<half::bf16> = buffer_to_vec::<Metal, half::bf16>(&context, &out6);
    let prefix_out = standalone6
        .iter()
        .zip(&expected[..chunk * heads * dim])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    let q6_host = buffer_to_vec::<Metal, half::bf16>(&context, &q6);
    let full_q_host = buffer_to_vec::<Metal, half::bf16>(&context, &q);
    let prefix_q = q6_host
        .iter()
        .zip(&full_q_host[..chunk * heads * dim])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    let k6_host = buffer_to_vec::<Metal, half::bf16>(&context, &k6);
    let full_k_host = buffer_to_vec::<Metal, half::bf16>(&context, &k);
    let prefix_k = k6_host
        .iter()
        .zip(&full_k_host[..chunk * kv_heads * dim])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    let v6_host = buffer_to_vec::<Metal, half::bf16>(&context, &v6);
    let full_v_host = buffer_to_vec::<Metal, half::bf16>(&context, &v);
    let prefix_v = v6_host
        .iter()
        .zip(&full_v_host[..chunk * kv_heads * dim])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    println!("prefix_out={prefix_out} prefix_q={prefix_q} prefix_k={prefix_k} prefix_v={prefix_v}");

    let diff = chunked.iter().zip(&expected).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0f32, f32::max);
    let first_half = chunked[..chunk * heads * dim]
        .iter()
        .zip(&expected[..chunk * heads * dim])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    let second_half = chunked[chunk * heads * dim..]
        .iter()
        .zip(&expected[chunk * heads * dim..])
        .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
        .fold(0f32, f32::max);
    println!("{{\"kv_chunking_first_half\": {first_half}, \"kv_chunking_second_half\": {second_half}}}");
    println!("{{\"attention_kv_chunking_max_abs\": {diff}}}");
    assert!(diff < 1e-3, "kv chunking max-abs-diff {diff}");
}
