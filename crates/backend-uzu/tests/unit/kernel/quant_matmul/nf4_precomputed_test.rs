//! Correctness gate for the `QmvFastNf4Precomputed` kernel.
//!
//! Same math as `Nf4QmvConstant` / `nf4-lut-grft` / `Nf4QmvByte256`:
//!   out = scale * Σ codebook[nibble] * x
//! but the 256-entry bfloat2 LUT is precomputed CPU-side and bound as a
//! device buffer (vs being initialized from `constant nf4_codebook[16]` at
//! kernel start). The kernel intentionally does NOT include `nf4_common.h`.
//!
//! Different kernel skeleton can reorder K-block partial sums, so we allow
//! `worst_rel < 2.5e-2` to match `nf4_lut256_graft_test.rs`'s tolerance.

#![cfg(metal_backend)]

use backend_uzu::{
    DataType,
    backends::{
        common::{Backend, Context, DenseBuffer, Encoder},
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

/// CPU-side mirror of the NF4 16-entry codebook used by `nf4_common.h`.
/// MUST stay byte-identical to the `nf4_codebook[]` half literals (modulo
/// the half→bf16 round during LUT precompute).
const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.5250730,
    -0.39491748,
    -0.28444138,
    -0.18477343,
    -0.09105003,
    0.0,
    0.07958029,
    0.16093750,
    0.24611230,
    0.33791524,
    0.44070983,
    0.56261432,
    0.72295684,
    1.0,
];

/// Byte → (bfloat2(low_nibble, high_nibble)) expansion = 256 entries × 2 bf16
/// = 512 bf16 values. Flat `Vec<bf16>` laid out [b0.lo, b0.hi, b1.lo, b1.hi,
/// ...] so the kernel's `const device bfloat2*` load reads exactly the right
/// pair.
fn precompute_nf4_byte_lut() -> Vec<bf16> {
    let mut out = Vec::with_capacity(512);
    for b in 0u32..256 {
        out.push(bf16::from_f32(NF4_CODEBOOK[(b & 0xF) as usize]));
        out.push(bf16::from_f32(NF4_CODEBOOK[((b >> 4) & 0xF) as usize]));
    }
    out
}

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

fn run_nf4_precomputed(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    lut_buf: &<B as Backend>::DenseBuffer,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.w_u8);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode_nf4_precomputed(
        &w_buf,
        &s_buf,
        lut_buf,
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
fn nf4_precomputed_equivalent_to_constant() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    eprintln!("[nf4_precomputed] device={}", device);
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    // Precompute the 256-entry bfloat2 LUT once (shared across all problems).
    let lut_data = precompute_nf4_byte_lut();
    let lut_buf = alloc_buffer_with_data::<B, bf16>(&ctx, &lut_data);

    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (2, 1024, 128), (4, 2048, 256)] {
        let p = make_problem(m, k, n);
        let cst = run_nf4_constant(&ctx, &bench, &p);
        let pre = run_nf4_precomputed(&ctx, &bench, &lut_buf, &p);
        let (worst_rel, worst_abs) = worst_errors(&cst, &pre);
        eprintln!(
            "[nf4_precomputed] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        assert!(
            worst_rel < 2.5e-2,
            "nf4-precomputed not equivalent to Nf4QmvConstant (M={} K={} N={}): \
             worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}
