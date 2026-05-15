//! Correctness tests for the NF4-ZP (asymmetric 4-bit-LUT zero-point) Metal
//! kernels `Nf4QmvZp` / `Nf4QmmZp` against a self-contained CPU reference.
//!
//! These NF4 kernels are bench-only (non-PUBLIC in the DSL sense: no CPU
//! impl), so unlike the PUBLIC qmv/qmm tests we cannot route through the
//! CPU backend. Instead we compute the reference here, in
//! `nf4_zp_reference`, mirroring the exact `nf4_zp_lut` + NF4 codebook the
//! Metal side uses.
//!
//! Dequant:  out[i,j] = Σ_l x[i,l] · scale[j,g] · (codebook[w[j,l]] +
//!                                                  zp_lut[zp_idx[j,g]])
//! where g = l / group_size and zp indices are 4-bit, packed two-per-byte,
//! row-major (AWQ 4-bit zero-point packing convention).

#![cfg(metal_backend)]

use backend_uzu::{
    DataType,
    backends::{
        common::{Backend, Context, DenseBuffer, Encoder},
        metal::{
            Metal,
            kernel::quant_matmul_nf4_bench::{Nf4QmmBench, Nf4QmmTile, Nf4QmvBench},
        },
    },
};
use half::bf16;

use crate::{common::helpers::alloc_buffer_with_data, uzu_test};

type Ctx = <Metal as Backend>::Context;
type B = Metal;

const GROUP_SIZE: usize = 64;

/// Max relative error tolerance. The NF4-ZP dequant is exact integer/LUT
/// arithmetic; the only error source is bf16 (8-bit mantissa) rounding of
/// the x/scale operands and fp32 accumulation over K. ~3% covers the worst
/// near-zero output at K up to ~1024.
const TOL: f64 = 0.03;
/// Absolute tolerance floor (outputs are O(1); near-zero outputs accumulate
/// bf16 cancellation noise that is meaningless in relative terms).
const ABS_TOL: f64 = 0.01;

/// NF4 16-entry codebook (must match `nf4_codebook` in nf4_common.h).
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

/// NF4 zero-point LUT: evenly spaced symmetric offsets,
/// `nf4_zp_lut[i] = i / 15.0 - 0.5` for i in 0..16 (matches nf4_common.h).
/// The Metal table is stored as `half`, so we round to f16 here to mirror it.
fn nf4_zp_lut(i: usize) -> f32 {
    let v = (i as f32) / 15.0 - 0.5;
    half::f16::from_f32(v).to_f32()
}

fn bf16_buf(
    ctx: &Ctx,
    values: &[f32],
) -> <B as Backend>::DenseBuffer {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    alloc_buffer_with_data::<B, bf16>(ctx, &data)
}

/// Pack 4-bit weights [N, K] row-major into u32 words (8 nibbles/word),
/// matching the codebase's `pack_weights_u32` for 4-bit.
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

/// Pack one row of 4-bit zero-point indices two-per-byte (low nibble = even
/// group, high nibble = odd group). Matches `pack_zero_points` (bits == 4).
fn pack_zp_row(row: &[u8]) -> Vec<u8> {
    row.chunks(2)
        .map(|chunk| {
            let lo = chunk[0] & 0x0F;
            let hi = if chunk.len() > 1 {
                chunk[1] & 0x0F
            } else {
                0
            };
            lo | (hi << 4)
        })
        .collect()
}

struct Problem {
    m: usize,
    k: usize,
    n: usize,
    w_packed: Vec<u32>,
    scales: Vec<f32>,
    zp_idx: Vec<u8>,    // [N, num_groups] raw indices
    zp_packed: Vec<u8>, // [N, zp_stride] packed
    x: Vec<f32>,
}

fn make_problem(
    m: usize,
    k: usize,
    n: usize,
) -> Problem {
    let num_groups = k / GROUP_SIZE;
    let zp_stride = (num_groups + 1) / 2;

    // Weights [N, K], 4-bit, deterministic pseudo-random.
    let mut w_raw: Vec<u8> = Vec::with_capacity(n * k);
    for j in 0..n {
        for l in 0..k {
            w_raw.push(((j * 7 + l * 13 + 3) % 16) as u8);
        }
    }
    let w_packed = pack_weights_u32(&w_raw);

    // Scales [N, num_groups].
    let mut scales: Vec<f32> = Vec::with_capacity(n * num_groups);
    for j in 0..n {
        for g in 0..num_groups {
            scales.push(0.02 + 0.005 * ((j + g) % 9) as f32);
        }
    }

    // Zero-point indices [N, num_groups], 4-bit, deterministic.
    let mut zp_idx: Vec<u8> = Vec::with_capacity(n * num_groups);
    for j in 0..n {
        for g in 0..num_groups {
            zp_idx.push(((j * 5 + g * 3 + 1) % 16) as u8);
        }
    }
    // Pack per row, pad each packed row to zp_stride.
    let mut zp_packed: Vec<u8> = Vec::with_capacity(n * zp_stride);
    for j in 0..n {
        let mut packed = pack_zp_row(&zp_idx[j * num_groups..(j + 1) * num_groups]);
        packed.resize(zp_stride, 0);
        zp_packed.extend_from_slice(&packed);
    }

    // X [M, K].
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
        w_packed,
        scales,
        zp_idx,
        zp_packed,
        x,
    }
}

/// CPU reference: out[i,j] = Σ_l x[i,l] · scale[j,g] ·
///                              (codebook[w[j,l]] + zp_lut[zp_idx[j,g]])
fn nf4_zp_reference(p: &Problem) -> Vec<f32> {
    let num_groups = p.k / GROUP_SIZE;
    // Round x and scales through bf16 to mirror the bf16 GPU buffers.
    let xb: Vec<f32> = p.x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let sb: Vec<f32> = p.scales.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();

    let mut out = vec![0.0f32; p.m * p.n];
    for i in 0..p.m {
        for j in 0..p.n {
            let mut acc = 0.0f32;
            for l in 0..p.k {
                let widx = j * p.k + l;
                let word = p.w_packed[widx / 8];
                let nib = ((word >> ((widx % 8) * 4)) & 0xF) as usize;
                let g = l / GROUP_SIZE;
                let scale = sb[j * num_groups + g];
                let zp = nf4_zp_lut(p.zp_idx[j * num_groups + g] as usize);
                let w_deq = scale * (NF4_CODEBOOK[nib] + zp);
                acc += xb[i * p.k + l] * w_deq;
            }
            out[i * p.n + j] = acc;
        }
    }
    out
}

fn run_qmv_gpu(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_packed);
    let s_buf = bf16_buf(ctx, &p.scales);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.zp_packed);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode_zp(
        &w_buf,
        &s_buf,
        &zp_buf,
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

fn run_qmm_gpu(
    ctx: &Ctx,
    bench: &Nf4QmmBench,
    tile: Nf4QmmTile,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_packed);
    let s_buf = bf16_buf(ctx, &p.scales);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.zp_packed);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode_zp(
        tile,
        &w_buf,
        &s_buf,
        &zp_buf,
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

/// Combined abs+rel tolerance check (mirrors the codebase's
/// `check_tolerance`): an element is a mismatch only if the absolute diff
/// exceeds BOTH `ABS_TOL` and `|expected| * TOL`. Returns (num_mismatches,
/// worst observed rel error among significant outputs).
fn count_mismatches(
    expected: &[f32],
    actual: &[f32],
) -> (usize, f64) {
    assert_eq!(expected.len(), actual.len());
    let mut mism = 0usize;
    let mut worst_rel = 0.0f64;
    for (&e, &a) in expected.iter().zip(actual.iter()) {
        let diff = (e - a).abs() as f64;
        let tol = ABS_TOL.max(e.abs() as f64 * TOL);
        if diff > tol {
            mism += 1;
        }
        // Track rel error only where the output is not near zero (the abs
        // floor dominates there and a huge ratio is meaningless).
        if e.abs() as f64 > 10.0 * ABS_TOL {
            worst_rel = worst_rel.max(diff / e.abs() as f64);
        }
    }
    (mism, worst_rel)
}

#[uzu_test]
fn nf4_zp_qmv_vs_cpu_reference() {
    let ctx = Ctx::new().expect("Metal context required");
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    // QMV regime: small M. The QMV kernel processes a 512-element K block
    // per outer iteration (16 vals/thread × 32 lanes), so K must be a
    // multiple of 512; N a multiple of 32.
    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256)] {
        let p = make_problem(m, k, n);
        let expected = nf4_zp_reference(&p);
        let got = run_qmv_gpu(&ctx, &bench, &p);
        let (mism, worst_rel) = count_mismatches(&expected, &got);
        eprintln!(
            "[nf4_zp_qmv] M={} K={} N={} mismatches={}/{} worst_rel(significant)={:.5}",
            m,
            k,
            n,
            mism,
            expected.len(),
            worst_rel
        );
        // Arithmetic is exact; residual is bf16 (8-bit mantissa) rounding of
        // operands + accumulation over K.
        assert_eq!(mism, 0, "NF4-ZP QMV mismatch (M={} K={} N={}): {} elements out of tolerance", m, k, n, mism);
    }
}

#[uzu_test]
fn nf4_zp_qmm_vs_cpu_reference() {
    let ctx = Ctx::new().expect("Metal context required");
    let bench = Nf4QmmBench::new(&ctx).expect("Nf4QmmBench build");

    // Small tile (BM=8) for small M; big tile (BM=64) for large M.
    let cases: &[(usize, usize, usize, Nf4QmmTile)] = &[
        (2, 256, 128, Nf4QmmTile::Small),
        (8, 256, 128, Nf4QmmTile::Small),
        (64, 512, 256, Nf4QmmTile::Big),
    ];
    for &(m, k, n, tile) in cases {
        let p = make_problem(m, k, n);
        let expected = nf4_zp_reference(&p);
        let got = run_qmm_gpu(&ctx, &bench, tile, &p);
        let (mism, worst_rel) = count_mismatches(&expected, &got);
        eprintln!(
            "[nf4_zp_qmm] M={} K={} N={} tile={:?} mismatches={}/{} worst_rel(significant)={:.5}",
            m,
            k,
            n,
            tile,
            mism,
            expected.len(),
            worst_rel
        );
        assert_eq!(mism, 0, "NF4-ZP QMM mismatch (M={} K={} N={}): {} elements out of tolerance", m, k, n, mism);
    }
}
