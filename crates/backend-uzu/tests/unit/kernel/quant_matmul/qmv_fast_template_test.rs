//! Correctness gate for the function-constant-isolation template kernels
//! `QmvFastTemplateAwqLut` and `QmvFastTemplateNf4Lut`.
//!
//! Each is meant to be a hardcoded-flag mirror of a SPECIALIZE-based QmvFast
//! dispatch:
//!
//!   * `QmvFastTemplateAwqLut` ≡ QmvFast(use_lut=true, use_nf4=false,
//!                                       use_zero_points=true, use_mlx_quant=false)
//!   * `QmvFastTemplateNf4Lut` ≡ QmvFast(use_lut=true, use_nf4=true,
//!                                       use_zero_points=false, use_mlx_quant=false)
//!
//! K-block reduction order is identical between the two kernels (same K-loop
//! structure, same qdot helper), so output should match to bf16 ULP slack.
//! We allow `worst_rel < 1e-2` (same tolerance as nf4_lut256_graft_test).

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

fn pack_zp_row(values: &[u8]) -> Vec<u8> {
    values
        .chunks(2)
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
    /// u32-packed 4-bit weight nibbles (QmvFast layout, 8 nibbles per word).
    w_u32: Vec<u32>,
    scales: Vec<f32>,
    /// QmvFast-layout packed zero-points (4-bit indices two-per-byte).
    zp: Vec<u8>,
    x: Vec<f32>,
}

fn make_problem(
    m: usize,
    k: usize,
    n: usize,
) -> Problem {
    let num_groups = k / GROUP_SIZE;

    let weights_raw: Vec<u8> = (0..(n * k)).map(|i| ((i * 7 + 1) % 16) as u8).collect();
    let w_u32 = pack_weights_u32(&weights_raw);

    let scales: Vec<f32> = (0..(n * num_groups)).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();

    let zp_stride = (num_groups + 1) / 2;
    let mut zp: Vec<u8> = Vec::with_capacity(n * zp_stride);
    for j in 0..n {
        let row: Vec<u8> = (0..num_groups).map(|g| ((j * 2 + g * 3) % 16) as u8).collect();
        let mut packed = pack_zp_row(&row);
        packed.resize(zp_stride, 0);
        zp.extend_from_slice(&packed);
    }

    let x: Vec<f32> = (0..(m * k)).map(|i| ((i % 257) as f32) / 257.0).collect();

    Problem {
        m,
        k,
        n,
        w_u32,
        scales,
        zp,
        x,
    }
}

/// Reference for AWQ-template: production QmvFast(use_lut=true, use_nf4=false,
/// use_zero_points=true).
fn run_qmv_fast_awq_lut(
    ctx: &Ctx,
    p: &Problem,
) -> Vec<f32> {
    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
        ctx,
        DataType::BF16,
        GROUP_SIZE as u32,
        BITS as u32,
        true,  // use_zero_points
        false, // use_mlx_quant
        false, // use_hadamard
        true,  // use_lut
        false, // use_nf4
    )
    .expect("QmvFast awq-lut kernel build");

    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_u32);
    let s_buf = bf16_buf(ctx, &p.scales);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.zp);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    kernel.encode(
        &w_buf,
        &s_buf,
        Some(&zp_buf),
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

/// Reference for NF4-template: `Nf4QmvByte256` (production NF4 byte-LUT
/// kernel). NOTE: we deliberately do NOT use QmvFast(use_nf4=true,use_lut=true)
/// as the reference because that path currently contains a perf-only
/// "dummy zp load" probe (qmv_fast.metal:251-262) that makes the kernel
/// numerically wrong. The template-NF4 kernel implements the *correct* NF4
/// LUT graft (no probe), so its output should match Nf4QmvByte256 within
/// bf16 ULP slack — same math, same byte-batched LUT pattern.
fn run_nf4_byte256(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_u32);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode(
        Nf4Variant::Byte256,
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

fn run_tmpl_awq(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_u32);
    let s_buf = bf16_buf(ctx, &p.scales);
    let zp_buf = alloc_buffer_with_data::<B, u8>(ctx, &p.zp);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode_tmpl_awq(
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

fn run_tmpl_nf4(
    ctx: &Ctx,
    bench: &Nf4QmvBench,
    p: &Problem,
) -> Vec<f32> {
    let w_buf = alloc_buffer_with_data::<B, u32>(ctx, &p.w_u32);
    let s_buf = bf16_buf(ctx, &p.scales);
    let x_buf = bf16_buf(ctx, &p.x);
    let mut y_buf = ctx.create_buffer(p.m * p.n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx).unwrap();
    bench.encode(
        Nf4Variant::QmvFastTemplateNf4Lut,
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
fn qmv_fast_template_awq_equivalent_to_qmv_fast() {
    let ctx = Ctx::new().expect("Metal context required");
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (4, 1024, 128), (4, 2048, 256)] {
        let p = make_problem(m, k, n);
        let expected = run_qmv_fast_awq_lut(&ctx, &p);
        let actual = run_tmpl_awq(&ctx, &bench, &p);
        let (worst_rel, worst_abs) = worst_errors(&expected, &actual);
        eprintln!(
            "[qmv_fast_tmpl_awq] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        assert!(
            worst_rel < 1e-2,
            "QmvFastTemplateAwqLut not equivalent to QmvFast(awq-lut) \
             (M={} K={} N={}): worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}

#[uzu_test]
fn qmv_fast_template_nf4_equivalent_to_qmv_fast() {
    let ctx = Ctx::new().expect("Metal context required");
    let bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    for &(m, k, n) in &[(1usize, 512usize, 128usize), (2, 512, 128), (4, 512, 256), (4, 1024, 128), (4, 2048, 256)] {
        let p = make_problem(m, k, n);
        let expected = run_nf4_byte256(&ctx, &bench, &p);
        let actual = run_tmpl_nf4(&ctx, &bench, &p);
        let (worst_rel, worst_abs) = worst_errors(&expected, &actual);
        eprintln!(
            "[qmv_fast_tmpl_nf4] M={} K={} N={} worst_rel={:.3e} worst_abs={:.3e}",
            m, k, n, worst_rel, worst_abs
        );
        // Bf16 LUT vs half2 LUT in Nf4QmvByte256: bfloat has 7 mantissa bits
        // (~1/128 ULP) vs half's 10 bits (~1/1024). Codebook value rounding
        // dominates residual error. Same tolerance as nf4_lut256_graft_test.
        assert!(
            worst_rel < 2.5e-2,
            "QmvFastTemplateNf4Lut not equivalent to Nf4QmvByte256 \
             (M={} K={} N={}): worst_rel={:.3e} worst_abs={:.3e}",
            m,
            k,
            n,
            worst_rel,
            worst_abs
        );
    }
}
