//! Liquid-Codec contract tests against the HF PyTorch reference.
//!
//! Env:
//! - `LIQUID_CODEC_MODEL`: HF snapshot dir with `config.json` +
//!   `model.safetensors`. Unset -> tests print `{"skipped": ...}`.
//! - `LIQUID_CODEC_GOLDEN`: dir with `input.f32`, `audio.f32`, `streamed.f32`
//!   dumped from the HF reference implementation run over the committed
//!   `fixtures/liquid_codec/codes.i64` clip. No generator script is bundled;
//!   the dumps are regenerable from the reference env, which is why they are
//!   kept outside the repo. Unset -> the length/consistency checks still run,
//!   float parity is skipped.
//!
//! Golden geometry: 1 s of seeded noise (seed 670) -> 13 codec frames. Codes
//! must match within the reference's own bf16 sensitivity (bf16 vs f32 weights
//! flip 4/156 codes); the audio bounds are calibrated the same way — decoding
//! the same codes in f32 vs the bf16 reference differs by rel_rms 4.6% /
//! max-abs 0.108 (ISTFT phase is knife-edge sensitive).

use std::path::Path;

use uzu_engine_macros::uzu_test;

use crate::audio::liquid_codec::LiquidCodec;

const CODES: &[u8] = include_bytes!("fixtures/liquid_codec/codes.i64");

/// Frame count of the 1 s golden clip: ceil(24000 / 1920) codec frames.
const CODEC_FRAMES: usize = 13;

fn read_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let bytes = std::fs::read(path.as_ref()).unwrap_or_else(|e| panic!("reading {}: {e}", path.as_ref().display()));
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i64(bytes: &[u8]) -> Vec<i64> {
    bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect()
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt()
}

/// Signal-relative error: rms(a - b) / rms(b). Robust to bf16 output noise.
fn rel_rms(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    rms(&diff) / rms(b)
}

fn max_abs_diff(
    a: &[f32],
    b: &[f32],
) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn load_codec() -> Option<LiquidCodec> {
    let Some(model_dir) = std::env::var_os("LIQUID_CODEC_MODEL") else {
        println!("{{\"skipped\": \"LIQUID_CODEC_MODEL not set\"}}");
        return None;
    };
    Some(LiquidCodec::load_from_model_dir(&model_dir).expect("loading Liquid-Codec"))
}

fn golden_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("LIQUID_CODEC_GOLDEN").map(std::path::PathBuf::from);
    if dir.is_none() {
        println!("{{\"skipped\": \"LIQUID_CODEC_GOLDEN not set (float parity needs the reference dumps)\"}}");
    }
    dir
}

#[uzu_test]
fn liquid_codec_matches_reference() {
    let Some(codec) = load_codec() else {
        return;
    };
    let Some(golden) = golden_dir() else {
        return;
    };

    let input = read_f32(golden.join("input.f32"));
    assert_eq!(input.len(), 24_000, "golden clip is one second");
    let expected_codes = read_i64(CODES);
    let ref_codes: Vec<Vec<i64>> =
        (0..12).map(|cb| expected_codes[cb * CODEC_FRAMES..(cb + 1) * CODEC_FRAMES].to_vec()).collect();

    // --- encode ---
    let codes = codec.encode(&input).expect("encoding");
    assert_eq!(codes.len(), 12, "codebook count");
    assert_eq!(codes[0].len(), CODEC_FRAMES, "frame count");
    let mut code_mismatches = Vec::new();
    for (cb, row) in codes.iter().enumerate() {
        for (t, &value) in row.iter().enumerate() {
            let expected = expected_codes[cb * CODEC_FRAMES + t];
            if value != expected {
                code_mismatches.push((cb, t, value, expected));
            }
        }
    }

    // --- decode (reference codes: isolates the decoder numerics) ---
    let audio = codec.decode(&ref_codes).expect("decoding");
    let expected_audio = read_f32(golden.join("audio.f32"));
    assert_eq!(audio.len(), expected_audio.len(), "decoded length");
    let audio_diff = max_abs_diff(&audio, &expected_audio);
    let audio_rel = rel_rms(&audio, &expected_audio);

    // --- streaming decode ---
    let mut cache = codec.new_streaming_cache();
    let mut streamed = Vec::new();
    for t in 0..CODEC_FRAMES {
        let frame: Vec<Vec<i64>> = ref_codes.iter().map(|row| vec![row[t]]).collect();
        let chunk = codec.decode_streaming(&frame, &mut cache, false).expect("streaming chunk");
        assert_eq!(
            chunk.len(),
            if t == 0 {
                1440
            } else {
                1920
            },
            "chunk {t} length"
        );
        streamed.extend_from_slice(&chunk);
    }
    let empty: Vec<Vec<i64>> = ref_codes.iter().map(|_| Vec::new()).collect();
    let final_chunk = codec.decode_streaming(&empty, &mut cache, true).expect("finalize");
    assert_eq!(final_chunk.len(), 480, "final flush length");
    streamed.extend_from_slice(&final_chunk);
    assert!(cache.is_finished());
    let expected_streamed = read_f32(golden.join("streamed.f32"));
    assert_eq!(streamed.len(), expected_streamed.len(), "streamed length");
    let stream_diff = max_abs_diff(&streamed, &expected_streamed);
    let stream_rel = rel_rms(&streamed, &expected_streamed);

    println!(
        "{{\"codes\": \"{} mismatches / {}\", \"audio\": {{\"max_abs\": {audio_diff}, \"rel_rms\": {audio_rel}}}, \
         \"streaming\": {{\"max_abs\": {stream_diff}, \"rel_rms\": {stream_rel}}}}}",
        code_mismatches.len(),
        codes.len() * CODEC_FRAMES,
    );

    // FSQ rounding boundaries are knife-edge under bf16 noise; the measured
    // floor is the reference itself (bf16 vs f32 weights flip 4/156 codes).
    assert!(
        code_mismatches.len() <= 4,
        "code mismatches above the reference's own bf16 noise floor: {code_mismatches:?}"
    );
    assert!(audio_rel < 0.06, "decoded audio relative RMS {audio_rel} above the bf16 noise floor");
    assert!(audio_diff < 0.2, "decoded audio max-abs-diff {audio_diff} above the bf16 noise floor");
    assert!(stream_rel < 0.06, "streamed audio relative RMS {stream_rel} above the bf16 noise floor");
    assert!(stream_diff < 0.2, "streamed audio vs reference chunks {stream_diff} above the bf16 noise floor");
}

/// CPU throughput (same shape as the Metal one) for the three-way comparison.
#[uzu_test]
fn liquid_codec_cpu_throughput() {
    let Some(codec) = load_codec() else { return };
    let mut state = 0x1234_5678u32;
    let audio: Vec<f32> = (0..72_000)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect();
    let mut best_encode = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let codes = codec.encode(&audio).expect("encode");
        best_encode = best_encode.min(t.elapsed().as_secs_f64());
        assert_eq!(codes[0].len(), 38);
    }
    let codes = codec.encode(&audio).expect("encode");
    let mut best_decode = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let _ = codec.decode(&codes).expect("decode");
        best_decode = best_decode.min(t.elapsed().as_secs_f64());
    }
    let mut cache = codec.new_streaming_cache();
    let mut best_stream = f64::MAX;
    for _ in 0..3 {
        let mut fresh = codec.new_streaming_cache();
        let t = std::time::Instant::now();
        for f in 0..38 {
            let frame: Vec<Vec<i64>> = codes.iter().map(|row| vec![row[f]]).collect();
            let _ = codec.decode_streaming(&frame, &mut fresh, false).expect("chunk");
        }
        let empty: Vec<Vec<i64>> = codes.iter().map(|_| Vec::new()).collect();
        let _ = codec.decode_streaming(&empty, &mut fresh, true).expect("finalize");
        best_stream = best_stream.min(t.elapsed().as_secs_f64());
        let _ = &mut cache;
    }
    println!(
        "{{\"cpu_throughput\": {{\"encode_s\": {best_encode}, \"decode_s\": {best_decode}, \"stream_s\": {best_stream},          \"realtime_factor_encode\": {}}}}}",
        3.0f64 / best_encode
    );
}

/// Length semantics: 3 s of deterministic synthetic audio -> 38 codec frames,
/// 72960 samples on both the batch and the streaming decode paths.
#[uzu_test]
fn liquid_codec_length_semantics() {
    let Some(codec) = load_codec() else {
        return;
    };
    // deterministic LCG noise in [-1, 1)
    let mut state = 0x1234_5678u32;
    let audio: Vec<f32> = (0..72_000)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect();

    let codes = codec.encode(&audio).expect("encode");
    assert_eq!(codes.len(), 12);
    assert_eq!(codes[0].len(), 38, "24 kHz / 1920 = 12.5 frames per second");

    let decoded = codec.decode(&codes).expect("decode");
    assert_eq!(decoded.len(), 38 * 1920);

    let mut cache = codec.new_streaming_cache();
    let mut streamed = 0usize;
    for t in 0..38 {
        let frame: Vec<Vec<i64>> = codes.iter().map(|row| vec![row[t]]).collect();
        streamed += codec.decode_streaming(&frame, &mut cache, false).expect("chunk").len();
    }
    let empty: Vec<Vec<i64>> = codes.iter().map(|_| Vec::new()).collect();
    streamed += codec.decode_streaming(&empty, &mut cache, true).expect("finalize").len();
    assert_eq!(streamed, 38 * 1920, "streaming length must match the batch decode");
    assert!(cache.is_finished());

    // No frames, no started stream: nothing to emit, overlap-add state untouched.
    assert!(codec.decode(&empty).expect("empty decode").is_empty());
    let mut fresh = codec.new_streaming_cache();
    assert!(codec.decode_streaming(&empty, &mut fresh, true).expect("empty stream").is_empty());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use crate::audio::liquid_codec_metal::MetalLiquidCodec;

    #[uzu_test]
    fn liquid_codec_metal_matches_cpu() {
        let Some(model_dir) = std::env::var_os("LIQUID_CODEC_MODEL") else {
            println!("{{\"skipped\": \"LIQUID_CODEC_MODEL not set\"}}");
            return;
        };
        let Some(golden) = std::env::var_os("LIQUID_CODEC_GOLDEN").map(std::path::PathBuf::from) else {
            println!("{{\"skipped\": \"LIQUID_CODEC_GOLDEN not set\"}}");
            return;
        };
        let cpu = LiquidCodec::load_from_model_dir(&model_dir).expect("cpu codec");
        let mut gpu = MetalLiquidCodec::from_cpu(&cpu).expect("metal codec");

        let input = read_f32(golden.join("input.f32"));
        let expected_codes = read_i64(CODES);
        let ref_codes: Vec<Vec<i64>> =
            (0..12).map(|cb| expected_codes[cb * CODEC_FRAMES..(cb + 1) * CODEC_FRAMES].to_vec()).collect();

        // encode: metal vs CPU (and vs the reference codes)
        let cpu_codes = cpu.encode(&input).expect("cpu encode");
        let gpu_codes = gpu.encode(&input).expect("metal encode");
        let vs_cpu = cpu_codes
            .iter()
            .zip(&gpu_codes)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| usize::from(x != y)))
            .sum::<usize>();
        let mut vs_ref = 0usize;
        for (cb, row) in gpu_codes.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                vs_ref += usize::from(v != expected_codes[cb * CODEC_FRAMES + t]);
            }
        }

        // decode: metal vs CPU on the reference codes
        let cpu_audio = cpu.decode(&ref_codes).expect("cpu decode");
        let gpu_audio = gpu.decode(&ref_codes).expect("metal decode");
        assert_eq!(cpu_audio.len(), gpu_audio.len(), "decoded length");
        let audio_rel = rel_rms(&gpu_audio, &cpu_audio);
        let audio_max = max_abs_diff(&gpu_audio, &cpu_audio);

        // streaming decode
        let mut cache = cpu.new_streaming_cache();
        let mut state = gpu.new_streaming_state().expect("streaming state");
        let mut cpu_streamed = Vec::new();
        let mut gpu_streamed = Vec::new();
        let mut per_chunk_rel = Vec::new();
        let _ = &per_chunk_rel;
        for t in 0..CODEC_FRAMES as u32 {
            let frame: Vec<Vec<i64>> = ref_codes.iter().map(|row| vec![row[t as usize]]).collect();
            let cpu_chunk = cpu.decode_streaming(&frame, &mut cache, false).expect("cpu chunk");
            let gpu_chunk = gpu.decode_streaming(&frame, &mut state, false).expect("metal chunk");
            per_chunk_rel.push(rel_rms(&gpu_chunk, &cpu_chunk));
            cpu_streamed.extend_from_slice(&cpu_chunk);
            gpu_streamed.extend_from_slice(&gpu_chunk);
        }
        let empty: Vec<Vec<i64>> = ref_codes.iter().map(|_| Vec::new()).collect();
        cpu_streamed.extend_from_slice(&cpu.decode_streaming(&empty, &mut cache, true).expect("cpu finalize"));
        gpu_streamed.extend_from_slice(&gpu.decode_streaming(&empty, &mut state, true).expect("metal finalize"));
        assert_eq!(cpu_streamed.len(), gpu_streamed.len(), "stream length");
        let stream_rel = rel_rms(&gpu_streamed, &cpu_streamed);
        let stream_max = max_abs_diff(&gpu_streamed, &cpu_streamed);

        println!(
            "{{\"metal\": {{\"codes_vs_cpu\": {vs_cpu}, \"codes_vs_ref\": {vs_ref}, \
             \"audio\": {{\"rel_rms\": {audio_rel}, \"max_abs\": {audio_max}}}, \
             \"streaming\": {{\"rel_rms\": {stream_rel}, \"max_abs\": {stream_max}}}, \
             \"per_chunk_rel\": {:?}}}",
            per_chunk_rel
        );

        assert!(vs_cpu <= 4, "metal codes differ from CPU by {vs_cpu}/156");
        assert!(vs_ref <= 4, "metal codes differ from the reference by {vs_ref}/156");
        assert!(audio_rel < 0.06, "metal audio rel_rms {audio_rel} above the bf16 noise floor");
        assert!(stream_rel < 0.06, "metal streaming rel_rms {stream_rel} above the bf16 noise floor");
    }
}

/// Throughput / bandwidth check (env-gated like the parity tests). Prints the
/// measured wall time per call, the implied weight-bandwidth floor (147 MB of
/// bf16 weights per pass) and the achieved fraction of it.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[uzu_test]
fn liquid_codec_metal_throughput() {
    let Some(model_dir) = std::env::var_os("LIQUID_CODEC_MODEL") else {
        println!("{{\"skipped\": \"LIQUID_CODEC_MODEL not set\"}}");
        return;
    };
    if !MetalLiquidCodec::is_available() {
        println!("{{\"skipped\": \"Metal unavailable\"}}");
        return;
    }
    use crate::audio::liquid_codec_metal::MetalLiquidCodec;
    let cpu = LiquidCodec::load_from_model_dir(&model_dir).expect("cpu codec");
    let mut gpu = MetalLiquidCodec::from_cpu(&cpu).expect("metal codec");

    let mut state = 0x1234_5678u32;
    let audio: Vec<f32> = (0..24_000 * 3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect();

    let mut best_encode = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let codes = gpu.encode(&audio).expect("encode");
        best_encode = best_encode.min(t.elapsed().as_secs_f64());
        assert_eq!(codes[0].len(), 38);
    }
    let codes = gpu.encode(&audio).expect("encode");

    // Scaling: 1 / 6 / 38 frames separates fixed overhead from marginal cost.
    let mut decode_scaling = Vec::new();
    for frames in [1usize, 6, 38] {
        let sub: Vec<Vec<i64>> = codes.iter().map(|row| row[..frames].to_vec()).collect();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let audio = gpu.decode(&sub).expect("decode");
            best = best.min(t.elapsed().as_secs_f64());
            assert_eq!(audio.len(), frames * 1920);
        }
        decode_scaling.push((frames, best));
    }
    let mut encode_scaling = Vec::new();
    for seconds in [1usize, 3] {
        let sub = &audio[..24_000 * seconds];
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let _ = gpu.encode(sub).expect("encode");
            best = best.min(t.elapsed().as_secs_f64());
        }
        encode_scaling.push((seconds, best));
    }
    let mut best_decode = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let audio = gpu.decode(&codes).expect("decode");
        best_decode = best_decode.min(t.elapsed().as_secs_f64());
        assert_eq!(audio.len(), 38 * 1920);
    }

    // streaming: 38 single-frame chunks
    let mut stream_state = gpu.new_streaming_state().expect("state");
    let mut best_stream = f64::MAX;
    for _ in 0..3 {
        let mut fresh = gpu.new_streaming_state().expect("state");
        let t = std::time::Instant::now();
        for f in 0..38 {
            let frame: Vec<Vec<i64>> = codes.iter().map(|row| vec![row[f]]).collect();
            let _ = gpu.decode_streaming(&frame, &mut fresh, false).expect("chunk");
        }
        let empty: Vec<Vec<i64>> = codes.iter().map(|_| Vec::new()).collect();
        let _ = gpu.decode_streaming(&empty, &mut fresh, true).expect("finalize");
        best_stream = best_stream.min(t.elapsed().as_secs_f64());
        let _ = &mut stream_state;
    }

    // Micro-benchmark: one decoder-shaped GEMM (M=6, K=512, N=2304), 100 reps in
    // one command buffer vs 100 separate command buffers -> per-dispatch overhead
    // vs kernel time.
    {
        use crate::{
            backends::common::{
                CpuBuffer, GlobalBufferCpuGpu,
                kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput},
            },
            data_type::DataType,
        };
        type GpuBuf = <crate::backends::metal::Metal as crate::backends::Backend>::GlobalGpuBuffer;
        let ctx = gpu.context_for_bench().clone();
        let m = 6u32;
        let (k, n) = (512u32, 2304u32);
        let weights: Vec<half::bf16> = vec![half::bf16::from_f32(0.01); (n * k) as usize];
        let mut w_buf = ctx.create_buffer_gpu((n * k * 2) as usize).expect("w");
        w_buf.copyin(&weights);
        let w_buf = w_buf.handoff().expect("w handoff");
        let a: Vec<half::bf16> = vec![half::bf16::from_f32(0.1); (m * k) as usize];
        let mut a_buf = ctx.create_buffer_gpu((m * k * 2) as usize).expect("a");
        a_buf.copyin(&a);
        let a_buf = a_buf.handoff().expect("a handoff");
        let _out = ctx.create_buffer_gpu((m * n * 2) as usize).expect("o");

        let mut one_buffer = f64::MAX;
        for _ in 0..5 {
            let mut cb = ctx.create_command_buffer(None, None).expect("cb");
            let mut scratch_out = cb.allocate_scratch_for_shape(&[m, n], DataType::BF16).expect("s");
            for _ in 0..100 {
                gpu.matmul_bench()
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &a_buf,
                                offset: 0,
                            },
                            b: MatmulB::FullPrecision {
                                b: &w_buf,
                            },
                            b_leading_dimension: None,
                            b_transpose: true,
                            output: MatmulOutput::new(&mut scratch_out, MatmulDOps::none()),
                            gather_indices: None::<&GpuBuf>,
                            m,
                            n,
                            k,
                        },
                        &mut cb,
                    )
                    .expect("encode");
            }
            let t = std::time::Instant::now();
            block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
            one_buffer = one_buffer.min(t.elapsed().as_secs_f64());
        }
        let mut many_buffers = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..100 {
                let mut cb = ctx.create_command_buffer(None, None).expect("cb");
                let mut scratch_out = cb.allocate_scratch_for_shape(&[m, n], DataType::BF16).expect("s");
                gpu.matmul_bench()
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &a_buf,
                                offset: 0,
                            },
                            b: MatmulB::FullPrecision {
                                b: &w_buf,
                            },
                            b_leading_dimension: None,
                            b_transpose: true,
                            output: MatmulOutput::new(&mut scratch_out, MatmulDOps::none()),
                            gather_indices: None::<&GpuBuf>,
                            m,
                            n,
                            k,
                        },
                        &mut cb,
                    )
                    .expect("encode");
                block_on(cb.end_encoding().submit().wait_until_completed()).expect("run");
            }
            many_buffers = many_buffers.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "GEMM 100x one buffer: {one_buffer:.4}s | 100x separate buffers: {many_buffers:.4}s | weights read once: {:.1} MB",
            (n * k * 2) as f64 / 1e6
        );
    }

    // Overhead breakdown: empty command buffer round trip vs CPU->GPU upload.
    use futures::executor::block_on;

    use crate::backends::common::{
        CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
    };
    let context = gpu.context_for_bench();
    let mut best_empty = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let cb = context.create_command_buffer(None, None).expect("cb");
        block_on(cb.end_encoding().submit().wait_until_completed()).expect("submit");
        best_empty = best_empty.min(t.elapsed().as_secs_f64());
    }
    let mut best_upload = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let buf = gpu.upload_bench(&codes.iter().map(|r| r[0] as i32).collect::<Vec<_>>());
        best_upload = best_upload.min(t.elapsed().as_secs_f64());
        let _ = buf;
    }

    // 147 MB of bf16 weights: the decoder owns the bulk (decoder.lfm + dec_lin).
    const WEIGHT_MB: f64 = 147.0;
    let audio_seconds = 3.0f64;
    let encode_bw = WEIGHT_MB / best_encode;
    let decode_bw = WEIGHT_MB / best_decode;
    println!(
        "{{\"throughput\": {{\"encode_s\": {best_encode}, \"decode_s\": {best_decode}, \
         \"stream_s\": {best_stream}, \"weight_mb_per_s_encode\": {encode_bw}, \
         \"weight_mb_per_s_decode\": {decode_bw}, \"audio_seconds\": {audio_seconds}, \
         \"realtime_factor_encode\": {}, \"decode_scaling\": {:?}, \"encode_scaling\": {:?}, \
         \"empty_submit_s\": {best_empty}, \"upload_s\": {best_upload}}}",
        audio_seconds / best_encode,
        decode_scaling,
        encode_scaling
    );
}
