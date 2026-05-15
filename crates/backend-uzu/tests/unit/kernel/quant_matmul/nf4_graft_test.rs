//! Correctness gate for the NF4-codebook dequant GRAFTED into the production
//! `QuantizedMatmulQmvFast` skeleton (`use_nf4=true`, `use_zero_points=false`,
//! `use_mlx_quant=false`, `use_lut=false`).
//!
//! It must be numerically equivalent to the existing `Nf4QmvConstant` for
//! IDENTICAL NF4 inputs: same 16-entry codebook, same packed 4-bit weight
//! bytes, same bf16 per-group scales. Only the kernel SKELETON differs (the
//! tuned QmvFast tiling/occupancy vs the standalone Nf4QmvConstant kernel);
//! the per-weight dequant math (`scale * codebook[nibble]`, fp32 accum,
//! `dot` of 4-lane vectors) is byte-for-byte the same. Both cast the weight
//! buffer to `uint8_t*` and walk `w[2*i]`, `w[2*i+1]` with the same 4-nibble
//! (low->high) stride, so feeding the SAME u8 buffer to both is exact.
//!
//! A different kernel skeleton can reorder the K-block partial sums, so we
//! allow a tiny tolerance (`worst_rel < 1e-3`) rather than bit-exactness.

#![cfg(metal_backend)]

use backend_uzu::{
    DataType,
    backends::{
        common::{Backend, Context, DenseBuffer, Encoder, Kernels, kernel::QuantizedMatmulQmvFastKernel},
        metal::{
            Metal,
            kernel::quant_matmul_nf4_bench::{Nf4QmvBench, Nf4Variant},
        },
    },
};
use half::bf16;

use crate::{common::helpers::alloc_buffer_with_data, uzu_test};

type Ctx = <Metal as Backend>::Context;
type B = Metal;

const GROUP_SIZE: usize = 64;
const BITS: usize = 4;

fn bf16_buf(
    ctx: &Ctx,
    values: &[f32],
) -> <B as Backend>::DenseBuffer {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    alloc_buffer_with_data::<B, bf16>(ctx, &data)
}

struct Problem {
    m: usize,
    k: usize,
    n: usize,
    /// Raw 4-bit weight nibbles packed two-per-byte, row-major [N, K].
    /// This is the EXACT layout `Nf4QmvConstant` consumes; the grafted
    /// QmvFast kernel reads the very same bytes via `uint8_t*`.
    w_u8: Vec<u8>,
    scales: Vec<f32>,
    x: Vec<f32>,
}

fn make_problem(
    m: usize,
    k: usize,
    n: usize,
) -> Problem {
    let num_groups = k / GROUP_SIZE;

    // One byte per 2 K-elements per output row (2 nibbles/byte).
    let packed_bytes_per_row = (k * BITS) / 8;
    let mut w_u8: Vec<u8> = Vec::with_capacity(n * packed_bytes_per_row);
    for j in 0..n {
        for b in 0..packed_bytes_per_row {
            w_u8.push(((j * 13 + b * 7 + 1) % 256) as u8);
        }
    }

    let mut scales: Vec<f32> = Vec::with_capacity(n * num_groups);
    for j in 0..n {
        for g in 0..num_groups {
            scales.push(0.02 + 0.005 * ((j + g) % 9) as f32);
        }
    }

    let mut x: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for l in 0..k {
            x.push(0.05 * f32::sin((i * k + l) as f32 * 0.037) + 0.25);
        }
    }

    Problem {
        m,
        k,
        n,
        w_u8,
        scales,
        x,
    }
}

/// Reference: `Nf4QmvConstant` (16-entry constant codebook, bf16 scale).
fn run_nf4_constant(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.w_u8);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode(
        Nf4Variant::Constant,
        &w_buf,
        &s_buf,
        &x_buf,
        &mut y_buf,
        p.k as u32,
        p.n as u32,
        p.m as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const bf16;
    let raw = unsafe { std::slice::from_raw_parts(y_ptr, p.m * p.n) };
    raw.iter().map(|v| v.to_f32()).collect()
}

/// NF4 codebook dequant GRAFTED into the QmvFast skeleton. SAME weight bytes,
/// SAME bf16 scales, NO zero-points (scale-only NF4).
fn run_nf4_grafted(
    ctx: &Ctx,
    p: &Problem,
) -> Vec<f32> {
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        false, // use_zero_points: NF4 graft is scale-only
        false, // use_mlx_quant
        false, // use_hadamard
        false, // use_lut
        true,  // use_nf4: 16-entry NF4 codebook dequant
    )
    .expect("grafted-NF4 QmvFast kernel build");

    // Same u8 byte buffer the reference consumes; QmvFast reads it as
    // uint8_t* with an identical 4-nibble (low->high) stride.
    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.w_u8);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    kernel.encode(
        &w_buf,
        &s_buf,
        None::<&<B as Backend>::DenseBuffer>,
        None::<&<B as Backend>::DenseBuffer>,
        &x_buf,
        &mut y_buf,
        None::<&<B as Backend>::DenseBuffer>,
        p.k as u32,
        p.n as u32,
        p.m as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const bf16;
    let raw = unsafe { std::slice::from_raw_parts(y_ptr, p.m * p.n) };
    raw.iter().map(|v| v.to_f32()).collect()
}

/// Worst relative error among significant outputs (|expected| > 0.1) and
/// worst absolute error overall.
fn worst_errors(
    expected: &[f32],
    actual: &[f32],
) -> (f64, f64) {
    assert_eq!(expected.len(), actual.len());
    let mut worst_rel = 0.0f64;
    let mut worst_abs = 0.0f64;
    for (&e, &a) in expected.iter().zip(actual.iter()) {
        let diff = (e - a).abs() as f64;
        worst_abs = worst_abs.max(diff);
        if e.abs() as f64 > 0.1 {
            worst_rel = worst_rel.max(diff / e.abs() as f64);
        }
    }
    (worst_rel, worst_abs)
}

#[uzu_test]
fn nf4_grafted_qmv_equivalent_to_constant() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    eprintln!("[nf4_graft_eq] device={}", device);
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    // QMV: K multiple of 512 (512-element K block/iter), N multiple of 32.
    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (2, 1024, 128), (4, 2048, 256)] {
        let p = make_problem(m, k, n);
        let cst = run_nf4_constant(&ctx, &bench, &p);
        let graft = run_nf4_grafted(&ctx, &p);
        let (worst_rel, worst_abs) = worst_errors(&cst, &graft);
        eprintln!(
            "[nf4_graft_eq] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        assert!(
            worst_rel < 1e-3,
            "grafted-NF4-in-QmvFast not equivalent to Nf4QmvConstant (M={} K={} N={}): \
             worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}
