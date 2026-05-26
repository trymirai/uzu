#![cfg(metal_backend)]

//! GPU FRAME-CAPTURE harness — dispatches ONLY PSO A (awq-lut256) and PSO B
//! (nf4-lut256-graft) back-to-back inside an MTLCaptureManager scope so the
//! produced `.gputrace` contains a clean A→B signal with no other PSOs.
//!
//! Both PSOs share the `QuantizedMatmulQmvFast` skeleton, gs=64 bf16 4-bit:
//!
//!   PSO A — awq-lut256:
//!     use_zp=true, use_mlx=false, use_hadamard=false, use_lut=true, use_nf4=false
//!   PSO B — nf4-lut256-graft:
//!     use_zp=false, use_mlx=false, use_hadamard=false, use_lut=true, use_nf4=true
//!
//! Shape: Llama-MLPup K=4096 N=14336 M=4 — the deciding cell where
//! awq-lut256 = -13.1% vs scalar and nf4-lut256-graft = +91.1% vs scalar.
//!
//! Run via:
//!   cargo test --release -p backend-uzu --features metal --test performance \
//!     -- matmul::qmv_lut_trace::qmv_lut_trace_capture --ignored --nocapture
//!
//! Open the printed .gputrace path in Xcode → "Compute Pipeline State" view →
//! select PSO A or PSO B → "Source" pane shows qmv_fast.metal per-line cost
//! (enabled by `-O2 -gline-tables-only -frecord-sources` from the build script).

use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use backend_uzu::{
    DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::QuantizedMatmulQmvFastKernel},
        metal::Metal,
    },
};
use half::bf16;

use crate::common::helpers::alloc_buffer_with_data;

type Ctx = <Metal as Backend>::Context;
type B = Metal;

const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

// Llama-MLPup decoding cell: awq-lut256 -13.1% vs scalar, nf4-lut256-graft +91.1%.
const K: usize = 4096;
const N: usize = 14336;
const M: usize = 4;

// One command buffer, N_DISPATCH_PER_PSO of PSO A then N_DISPATCH_PER_PSO of PSO B.
const N_DISPATCH_PER_PSO: usize = 200;
const N_WARMUP: usize = 8;

fn bf16_buf(
    ctx: &Ctx,
    values: &[f32],
) -> <B as Backend>::DenseBuffer {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    alloc_buffer_with_data::<B, bf16>(ctx, &data)
}

fn pack_weights_u32(values: &[u8]) -> Vec<u32> {
    values
        .chunks(8)
        .map(|chunk| {
            let mut word = 0u32;
            for (i, &v) in chunk.iter().enumerate() {
                word |= ((v & 0xF) as u32) << (i * 4);
            }
            word
        })
        .collect()
}

#[test]
#[ignore]
fn qmv_lut_trace_capture() {
    // 1) Enable Metal capture BEFORE creating the context (sets METAL_CAPTURE_ENABLED=1).
    <Metal as Backend>::Context::enable_capture();

    // 2) Create the context.
    let ctx_rc = Ctx::new().expect("Metal context required");
    let ctx: &Ctx = &ctx_rc;
    let device_name = {
        use metal::MTLDeviceExt;
        ctx_rc.device.name()
    };
    println!("[QMV_LUT_TRACE] device={}", device_name);
    println!("[QMV_LUT_TRACE] shape K={} N={} M={}  n_dispatch_per_pso={}", K, N, M, N_DISPATCH_PER_PSO);

    // 3) Build shared inputs. Values are irrelevant; the GPU just needs valid buffers.
    let num_groups = K / GROUP_SIZE;
    let weights_raw: Vec<u8> = (0..(N * K)).map(|i| ((i * 7 + 1) % 16) as u8).collect();
    let w_u32_data = pack_weights_u32(&weights_raw);
    let weights_u8: Vec<u8> = (0..(N * (K * BITS / 8))).map(|i| (i % 251) as u8).collect();
    let scales_f32: Vec<f32> = (0..(N * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
    let zp_stride = (num_groups + 1) / 2;
    let zp_packed: Vec<u8> = (0..(N * zp_stride)).map(|i| (i % 251) as u8).collect();
    let x_f32: Vec<f32> = (0..(M * K)).map(|i| ((i % 257) as f32) / 257.0).collect();

    let w_u32 = alloc_buffer_with_data::<B, u32>(ctx, &w_u32_data);
    let w_u8 = alloc_buffer_with_data::<B, u8>(ctx, &weights_u8);
    let s_bf16 = bf16_buf(ctx, &scales_f32);
    let zp = alloc_buffer_with_data::<B, u8>(ctx, &zp_packed);
    let x_buf = bf16_buf(ctx, &x_f32);
    let y_a = ctx.create_buffer(M * N * DataType::BF16.size_in_bytes()).expect("y_a buffer");
    let y_b = ctx.create_buffer(M * N * DataType::BF16.size_in_bytes()).expect("y_b buffer");

    // 4) Build PSO A — awq-lut256.
    let pso_a = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        true,  // use_zp
        false, // use_mlx
        false, // use_hadamard
        true,  // use_lut
        false, // use_nf4
    )
    .expect("PSO A (awq-lut256) build");

    // 4b) Build PSO B — nf4-lut256-graft.
    let pso_b = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zp
        false, // use_mlx
        false, // use_hadamard
        true,  // use_lut
        true,  // use_nf4
    )
    .expect("PSO B (nf4-lut256-graft) build");

    let mut y_a_mut = y_a.clone();
    let mut y_b_mut = y_b.clone();

    // 5) Warmup OUTSIDE the capture scope (caches, PSO load, allocator pools).
    println!("[QMV_LUT_TRACE] warmup ({} dispatches each, outside capture)", N_WARMUP);
    {
        let mut enc = Encoder::new(ctx).unwrap();
        for _ in 0..N_WARMUP {
            pso_a.encode(
                &w_u32,
                &s_bf16,
                Some(&zp),
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_a_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        for _ in 0..N_WARMUP {
            pso_b.encode(
                &w_u8,
                &s_bf16,
                None::<&<B as Backend>::DenseBuffer>,
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_b_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        enc.end_encoding().submit().wait_until_completed().expect("warmup submit");
    }

    // 6) Build the trace path and START the capture.
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let trace_path = PathBuf::from(format!("/tmp/qmv_lut_{}.gputrace", timestamp));
    println!("[QMV_LUT_TRACE] starting capture -> {}", trace_path.display());
    ctx.start_capture(&trace_path).expect("start_capture (METAL_CAPTURE_ENABLED must be set)");

    // 7) Submit ONE command buffer: N_DISPATCH_PER_PSO of A then N_DISPATCH_PER_PSO of B.
    {
        let mut enc = Encoder::new(ctx).unwrap();
        for _ in 0..N_DISPATCH_PER_PSO {
            pso_a.encode(
                &w_u32,
                &s_bf16,
                Some(&zp),
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_a_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        for _ in 0..N_DISPATCH_PER_PSO {
            pso_b.encode(
                &w_u8,
                &s_bf16,
                None::<&<B as Backend>::DenseBuffer>,
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_b_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        let completed = enc.end_encoding().submit().wait_until_completed().expect("captured submit");
        let total_ms = completed.gpu_execution_time().as_secs_f64() * 1000.0;
        let per_dispatch_avg_ms = total_ms / (2 * N_DISPATCH_PER_PSO) as f64;
        println!(
            "[QMV_LUT_TRACE] captured CB gpu_total={:.3} ms  (~{:.4} ms / dispatch, averaged across A+B)",
            total_ms, per_dispatch_avg_ms
        );
    }

    // 8) Stop the capture.
    ctx.stop_capture().expect("stop_capture");
    println!("[QMV_LUT_TRACE] capture stopped");

    // 9) Verify the trace file exists and report its size.
    match std::fs::metadata(&trace_path) {
        Ok(meta) => {
            let bytes = meta.len();
            println!(
                "[QMV_LUT_TRACE] trace written: {}  ({} bytes ≈ {:.2} MiB)",
                trace_path.display(),
                bytes,
                bytes as f64 / (1024.0 * 1024.0)
            );
            assert!(bytes > 0, "trace file is empty");
        },
        Err(err) => {
            // .gputrace is a bundle (directory) on disk; if std::fs::metadata fails on
            // the bundle root, try the inner descriptor.
            if trace_path.is_dir() {
                println!("[QMV_LUT_TRACE] trace written (bundle): {}", trace_path.display());
            } else {
                panic!("[QMV_LUT_TRACE] trace file missing at {}: {}", trace_path.display(), err);
            }
        },
    }

    println!("[QMV_LUT_TRACE] DONE. Open in Xcode:\n    open {}", trace_path.display());
}

/// Split-capture variant: emits TWO `.gputrace` files, one per PSO, so each
/// trace contains a single PSO's dispatches for cleaner per-PSO inspection.
///
///   PSO A — awq-lut256       → /tmp/qmv_lut_awq.gputrace
///   PSO B — nf4-lut256-graft → /tmp/qmv_lut_nf4.gputrace
///
/// Run via:
///   cargo test --release -p backend-uzu --features metal --test performance \
///     -- matmul::qmv_lut_trace::qmv_lut_trace_capture_split --ignored --nocapture
///
/// If multi-capture in a single process is blocked by Apple, set
/// `UZU_TRACE_SELECT=awq` or `UZU_TRACE_SELECT=nf4` and run twice; the test
/// will skip the other PSO's capture stage when the env var is set.
#[test]
#[ignore]
fn qmv_lut_trace_capture_split() {
    let select = std::env::var("UZU_TRACE_SELECT").ok();
    let do_awq = select.as_deref().map(|s| s == "awq").unwrap_or(true);
    let do_nf4 = select.as_deref().map(|s| s == "nf4").unwrap_or(true);

    // 1) Enable Metal capture BEFORE creating the context.
    <Metal as Backend>::Context::enable_capture();

    // 2) Create the context.
    let ctx_rc = Ctx::new().expect("Metal context required");
    let ctx: &Ctx = &ctx_rc;
    let device_name = {
        use metal::MTLDeviceExt;
        ctx_rc.device.name()
    };
    println!("[QMV_LUT_TRACE_SPLIT] device={}", device_name);
    println!(
        "[QMV_LUT_TRACE_SPLIT] shape K={} N={} M={}  n_dispatch_per_pso={}  select={:?}",
        K, N, M, N_DISPATCH_PER_PSO, select
    );

    // 3) Build shared inputs.
    let num_groups = K / GROUP_SIZE;
    let weights_raw: Vec<u8> = (0..(N * K)).map(|i| ((i * 7 + 1) % 16) as u8).collect();
    let w_u32_data = pack_weights_u32(&weights_raw);
    let weights_u8: Vec<u8> = (0..(N * (K * BITS / 8))).map(|i| (i % 251) as u8).collect();
    let scales_f32: Vec<f32> = (0..(N * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
    let zp_stride = (num_groups + 1) / 2;
    let zp_packed: Vec<u8> = (0..(N * zp_stride)).map(|i| (i % 251) as u8).collect();
    let x_f32: Vec<f32> = (0..(M * K)).map(|i| ((i % 257) as f32) / 257.0).collect();

    let w_u32 = alloc_buffer_with_data::<B, u32>(ctx, &w_u32_data);
    let w_u8 = alloc_buffer_with_data::<B, u8>(ctx, &weights_u8);
    let s_bf16 = bf16_buf(ctx, &scales_f32);
    let zp = alloc_buffer_with_data::<B, u8>(ctx, &zp_packed);
    let x_buf = bf16_buf(ctx, &x_f32);
    let y_a = ctx.create_buffer(M * N * DataType::BF16.size_in_bytes()).expect("y_a buffer");
    let y_b = ctx.create_buffer(M * N * DataType::BF16.size_in_bytes()).expect("y_b buffer");

    // 4) Build both PSOs (cheap; we may skip dispatch below).
    let pso_a = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        true,  // use_zp
        false, // use_mlx
        false, // use_hadamard
        true,  // use_lut
        false, // use_nf4
    )
    .expect("PSO A (awq-lut256) build");
    let pso_b = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zp
        false, // use_mlx
        false, // use_hadamard
        true,  // use_lut
        true,  // use_nf4
    )
    .expect("PSO B (nf4-lut256-graft) build");

    let mut y_a_mut = y_a.clone();
    let mut y_b_mut = y_b.clone();

    // 5) Warmup OUTSIDE the capture scopes.
    println!("[QMV_LUT_TRACE_SPLIT] warmup ({} dispatches each, outside capture)", N_WARMUP);
    {
        let mut enc = Encoder::new(ctx).unwrap();
        for _ in 0..N_WARMUP {
            pso_a.encode(
                &w_u32,
                &s_bf16,
                Some(&zp),
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_a_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        for _ in 0..N_WARMUP {
            pso_b.encode(
                &w_u8,
                &s_bf16,
                None::<&<B as Backend>::DenseBuffer>,
                None::<&<B as Backend>::DenseBuffer>,
                &x_buf,
                &mut y_b_mut,
                None::<&<B as Backend>::DenseBuffer>,
                K as u32,
                N as u32,
                M as u32,
                &mut enc,
            );
        }
        enc.end_encoding().submit().wait_until_completed().expect("warmup submit");
    }

    let trace_awq = PathBuf::from("/tmp/qmv_lut_awq.gputrace");
    let trace_nf4 = PathBuf::from("/tmp/qmv_lut_nf4.gputrace");

    // Helper to remove a pre-existing bundle so the capture API can write fresh.
    fn ensure_no_existing(path: &std::path::Path) {
        if path.exists() {
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(path);
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    // 6a) Capture PSO A.
    let mut awq_ok = false;
    if do_awq {
        ensure_no_existing(&trace_awq);
        println!("[QMV_LUT_TRACE_SPLIT] starting capture A -> {}", trace_awq.display());
        match ctx.start_capture(&trace_awq) {
            Ok(()) => {
                let mut enc = Encoder::new(ctx).unwrap();
                for _ in 0..N_DISPATCH_PER_PSO {
                    pso_a.encode(
                        &w_u32,
                        &s_bf16,
                        Some(&zp),
                        None::<&<B as Backend>::DenseBuffer>,
                        &x_buf,
                        &mut y_a_mut,
                        None::<&<B as Backend>::DenseBuffer>,
                        K as u32,
                        N as u32,
                        M as u32,
                        &mut enc,
                    );
                }
                let completed = enc.end_encoding().submit().wait_until_completed().expect("captured submit A");
                let total_ms = completed.gpu_execution_time().as_secs_f64() * 1000.0;
                println!(
                    "[QMV_LUT_TRACE_SPLIT] A captured CB gpu_total={:.3} ms  (~{:.4} ms / dispatch)",
                    total_ms,
                    total_ms / N_DISPATCH_PER_PSO as f64
                );
                ctx.stop_capture().expect("stop_capture A");
                awq_ok = true;
            },
            Err(err) => {
                println!("[QMV_LUT_TRACE_SPLIT] start_capture A FAILED: {:?}", err);
            },
        }
    } else {
        println!("[QMV_LUT_TRACE_SPLIT] skipping PSO A (UZU_TRACE_SELECT != awq)");
    }

    // 6b) Capture PSO B (separate session, new encoder, new .gputrace path).
    let mut nf4_ok = false;
    if do_nf4 {
        ensure_no_existing(&trace_nf4);
        println!("[QMV_LUT_TRACE_SPLIT] starting capture B -> {}", trace_nf4.display());
        match ctx.start_capture(&trace_nf4) {
            Ok(()) => {
                let mut enc = Encoder::new(ctx).unwrap();
                for _ in 0..N_DISPATCH_PER_PSO {
                    pso_b.encode(
                        &w_u8,
                        &s_bf16,
                        None::<&<B as Backend>::DenseBuffer>,
                        None::<&<B as Backend>::DenseBuffer>,
                        &x_buf,
                        &mut y_b_mut,
                        None::<&<B as Backend>::DenseBuffer>,
                        K as u32,
                        N as u32,
                        M as u32,
                        &mut enc,
                    );
                }
                let completed = enc.end_encoding().submit().wait_until_completed().expect("captured submit B");
                let total_ms = completed.gpu_execution_time().as_secs_f64() * 1000.0;
                println!(
                    "[QMV_LUT_TRACE_SPLIT] B captured CB gpu_total={:.3} ms  (~{:.4} ms / dispatch)",
                    total_ms,
                    total_ms / N_DISPATCH_PER_PSO as f64
                );
                ctx.stop_capture().expect("stop_capture B");
                nf4_ok = true;
            },
            Err(err) => {
                println!(
                    "[QMV_LUT_TRACE_SPLIT] start_capture B FAILED: {:?} (likely Apple's \
                     one-capture-per-process limit — rerun with UZU_TRACE_SELECT=nf4)",
                    err
                );
            },
        }
    } else {
        println!("[QMV_LUT_TRACE_SPLIT] skipping PSO B (UZU_TRACE_SELECT != nf4)");
    }

    // 7) Report on each produced bundle.
    fn report(
        label: &str,
        path: &std::path::Path,
        wrote: bool,
    ) {
        if !wrote {
            return;
        }
        if path.is_dir() {
            // .gputrace is a bundle — sum the bytes inside.
            let mut total: u64 = 0;
            if let Ok(read) = std::fs::read_dir(path) {
                for entry in read.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        total += meta.len();
                    }
                }
            }
            println!(
                "[QMV_LUT_TRACE_SPLIT] {} bundle: {}  ({} bytes ≈ {:.2} MiB)",
                label,
                path.display(),
                total,
                total as f64 / (1024.0 * 1024.0)
            );
        } else if let Ok(meta) = std::fs::metadata(path) {
            let bytes = meta.len();
            println!(
                "[QMV_LUT_TRACE_SPLIT] {} file: {}  ({} bytes ≈ {:.2} MiB)",
                label,
                path.display(),
                bytes,
                bytes as f64 / (1024.0 * 1024.0)
            );
        } else {
            println!("[QMV_LUT_TRACE_SPLIT] {} MISSING at {}", label, path.display());
        }
    }
    report("AWQ", &trace_awq, awq_ok);
    report("NF4", &trace_nf4, nf4_ok);

    println!("[QMV_LUT_TRACE_SPLIT] DONE.\n    open {}\n    open {}", trace_awq.display(), trace_nf4.display());
}
