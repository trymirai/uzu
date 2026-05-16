//! Correctness gate for `Nf4QmvShuffle` (zero-memory register *shuffle*
//! codebook, S ∈ {8,16,32}).
//!
//!  - S=16 must be numerically equivalent to `Nf4QmvConstant` for identical
//!    NF4 inputs (same 16 codebook values, same packed weights, same bf16
//!    scales): same dequant math, only the codebook *access* differs
//!    (constant-space gather vs `simd_shuffle` of a register-held entry).
//!    Expect bit-exact / tiny rel err.
//!  - S=8 and S=32 use SYNTHETIC monotonic tables; their values are timing
//!    probes, so we check them against a CPU reference using the SAME table
//!    (proves the shuffle indexing is correct, not garbage).

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

fn bf16_buf(
    ctx: &Ctx,
    values: &[f32],
) -> <B as Backend>::DenseBuffer {
    let data: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
    alloc_buffer_with_data::<B, bf16>(ctx, &data)
}

/// Pack 4-bit weights [N, K] row-major into u32 words (8 nibbles/word).
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

/// The exact synthetic / real shuffle codebooks the kernel uses (mirrors
/// `nf4_common.h::nf4_my_shuffle_entry_s{8,16,32}`). bf16-rounded so the CPU
/// reference matches the half literals after fp16 storage in registers.
fn shuffle_table(s: usize) -> Vec<f32> {
    match s {
        8 => vec![
            -1.0, -0.71428573, -0.42857143, -0.14285715, 0.14285713, 0.42857146, 0.71428573, 1.0,
        ],
        16 => vec![
            -1.0, -0.6961928, -0.5250730, -0.39491748, -0.28444138, -0.18477343, -0.09105003, 0.0,
            0.07958029, 0.16093750, 0.24611230, 0.33791524, 0.44070983, 0.56261432, 0.72295684, 1.0,
        ],
        32 => (0..32).map(|i| -1.0 + 2.0 * (i as f32) / 31.0).collect(),
        _ => unreachable!(),
    }
}

struct Problem {
    m: usize,
    k: usize,
    n: usize,
    w_raw: Vec<u8>,
    w_packed: Vec<u32>,
    scales: Vec<f32>,
    x: Vec<f32>,
}

fn make_problem(
    m: usize,
    k: usize,
    n: usize,
) -> Problem {
    let num_groups = k / GROUP_SIZE;

    let mut w_raw: Vec<u8> = Vec::with_capacity(n * k);
    for j in 0..n {
        for l in 0..k {
            w_raw.push(((j * 7 + l * 13 + 3) % 16) as u8);
        }
    }
    let w_packed = pack_weights_u32(&w_raw);

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
        w_raw,
        w_packed,
        scales,
        x,
    }
}

fn run_qmv(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    variant: Nf4Variant,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_packed);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode(
        variant,
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

/// CPU reference for the shuffle kernel with codebook size `s`. Mirrors the
/// kernel: nibble masked to 3 bits for S=8 (else 4 bits), table lookup, fp32
/// accumulate, per-group bf16 scale, bf16 output. Inputs are bf16-rounded the
/// same way the kernel reads them.
fn cpu_reference(
    p: &Problem,
    s: usize,
) -> Vec<f32> {
    let table = shuffle_table(s);
    let num_groups = p.k / GROUP_SIZE;
    let nibble_mask: u8 = if s == 8 {
        0x07
    } else {
        0x0F
    };
    let mut out = vec![0.0f32; p.m * p.n];
    for i in 0..p.m {
        for j in 0..p.n {
            let mut acc = 0.0f32;
            for g in 0..num_groups {
                let scale = bf16::from_f32(p.scales[j * num_groups + g]).to_f32();
                let mut group_acc = 0.0f32;
                for t in 0..GROUP_SIZE {
                    let l = g * GROUP_SIZE + t;
                    let nib = p.w_raw[j * p.k + l] & nibble_mask;
                    let w = table[nib as usize];
                    let xv = bf16::from_f32(p.x[i * p.k + l]).to_f32();
                    group_acc += xv * w;
                }
                acc += scale * group_acc;
            }
            out[i * p.n + j] = bf16::from_f32(acc).to_f32();
        }
    }
    out
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
fn nf4_shuffle16_qmv_equivalent_to_constant() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    eprintln!("[nf4_shuffle16_eq] device={}", device);
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (2, 1024, 128)] {
        let p = make_problem(m, k, n);
        let cst = run_qmv(&ctx, &bench, Nf4Variant::Constant, &p);
        let shuf = run_qmv(&ctx, &bench, Nf4Variant::Shuffle16, &p);
        let (worst_rel, worst_abs) = worst_errors(&cst, &shuf);
        eprintln!(
            "[nf4_shuffle16_eq] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        // Same dequant math + same fp32 accumulation order; only the codebook
        // access (constant gather vs register simd_shuffle) differs. Must be
        // bit-exact / within a single bf16 ULP.
        assert!(
            worst_rel < 1e-3,
            "Nf4QmvShuffle S=16 not equivalent to Nf4QmvConstant (M={} K={} N={}): \
             worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}

#[uzu_test]
fn nf4_shuffle8_32_qmv_match_cpu_reference() {
    let ctx = Ctx::new().expect("Metal context required");
    let device = {
        use metal::MTLDeviceExt;
        ctx.device.name()
    };
    eprintln!("[nf4_shuffle8_32_cpu] device={}", device);
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    for &(s, variant) in &[(8usize, Nf4Variant::Shuffle8), (32usize, Nf4Variant::Shuffle32)] {
        for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 1024, 256)] {
            let p = make_problem(m, k, n);
            let gpu = run_qmv(&ctx, &bench, variant, &p);
            let cpu = cpu_reference(&p, s);
            let (worst_rel, worst_abs) = worst_errors(&cpu, &gpu);
            eprintln!(
                "[nf4_shuffle8_32_cpu] S={} M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
                s, m, k, n, worst_rel, worst_abs
            );
            // fp16 register codebook + fp32 dot + bf16 round: small numeric
            // drift vs an fp32 CPU table is expected; a correct shuffle index
            // keeps this tight (well under 2%). Garbage indexing would blow up.
            assert!(
                worst_rel < 2e-2,
                "Nf4QmvShuffle S={} disagrees with CPU reference (M={} K={} N={}): \
                 worst_rel={:.3e} worst_abs={:.3e}",
                s,
                m,
                k,
                n,
                worst_rel,
                worst_abs
            );
        }
    }
}
