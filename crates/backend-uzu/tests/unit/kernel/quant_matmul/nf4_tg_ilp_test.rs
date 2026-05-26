//! Correctness gate for `Nf4QmvTgIlp` (multi-accumulator TG-codebook variant
//! of `Nf4QmvTg`). Must be numerically equivalent to `Nf4QmvConstant` within
//! bf16 ULP slack — the two-accumulator sum order differs slightly from the
//! single-accumulator path, but accuracy should hold.

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

struct Problem {
    m: usize,
    k: usize,
    n: usize,
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
fn nf4_tg_ilp_qmv_equivalent_to_constant() {
    let ctx = Ctx::new().expect("Metal context required");
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (2, 1024, 128)] {
        let p = make_problem(m, k, n);
        let cst = run_qmv(&ctx, &bench, Nf4Variant::Constant, &p);
        let ilp = run_qmv(&ctx, &bench, Nf4Variant::TgIlp, &p);
        let (worst_rel, worst_abs) = worst_errors(&cst, &ilp);
        eprintln!(
            "[nf4_tg_ilp_eq] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        // bf16 ULP slack: two-accumulator sum order differs slightly from
        // single-accumulator path, so allow ~1e-2 rel tolerance.
        assert!(
            worst_rel < 1e-2,
            "Nf4QmvTgIlp not equivalent to Nf4QmvConstant (M={} K={} N={}): worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}
