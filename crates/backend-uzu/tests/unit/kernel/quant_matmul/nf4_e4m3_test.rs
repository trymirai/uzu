//! Correctness tests for the NF4 E4M3-scale Metal kernels (`Nf4QmvE4m3`,
//! `Nf4QmmE4m3`) versus a CPU reference that round-trips each per-group
//! scale through E4M3. These kernels are bench-only (no PUBLIC CPU pair),
//! so they are driven through the `quant_matmul_nf4_bench` dispatcher.
#![cfg(metal_backend)]

use backend_uzu::{
    DataType,
    backends::{
        common::{Backend, Context, DenseBuffer, Encoder},
        cpu::nf4_e4m3::{f32_to_e4m3, nf4_qmm_e4m3_ref, nf4_qmv_e4m3_ref},
        metal::{
            Metal,
            kernel::quant_matmul_nf4_bench::{Nf4QmmBench, Nf4QmmTile, Nf4QmvBench, Nf4Variant},
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

/// Pack 4-bit nibbles (NF4 codebook indices) into u32 words, 8 per word,
/// matching the GPU's transposed `[N, K]` layout.
fn pack_nibbles_u32(values: &[u8]) -> Vec<u32> {
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

/// Build a deterministic NF4-E4M3 problem and return (packed_w, scale_bytes,
/// x_f32, cpu_expected).
#[allow(clippy::type_complexity)]
fn make_problem(
    m: usize,
    k: usize,
    n: usize,
    is_qmm: bool,
) -> (Vec<u32>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups_k = k.div_ceil(GROUP_SIZE);

    // Weights [N, K] of NF4 codebook indices 0..16.
    let weights_raw: Vec<u8> = (0..(n * k)).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let w_packed = pack_nibbles_u32(&weights_raw);

    // Per-group scales [N, num_groups_k]; encode each to E4M3 bytes. Use a
    // range that exercises normals and a few subnormals.
    let scales_f32: Vec<f32> =
        (0..(n * num_groups_k)).map(|i| 0.008 + (i % 11) as f32 * 0.013).collect();
    let scale_bytes: Vec<u8> = scales_f32.iter().map(|&s| f32_to_e4m3(s)).collect();

    // Activations [M, K].
    let x_f32: Vec<f32> = (0..(m * k)).map(|i| 0.15 * f32::sin(i as f32 * 0.037) + 0.4).collect();

    let mut expected = vec![0.0f32; m * n];
    if is_qmm {
        nf4_qmm_e4m3_ref(
            w_packed.as_ptr(),
            scale_bytes.as_ptr(),
            x_f32.as_ptr(),
            expected.as_mut_ptr(),
            k,
            n,
            m,
            GROUP_SIZE,
        );
    } else {
        nf4_qmv_e4m3_ref(
            w_packed.as_ptr(),
            scale_bytes.as_ptr(),
            x_f32.as_ptr(),
            expected.as_mut_ptr(),
            k,
            n,
            m,
            GROUP_SIZE,
        );
    }

    (w_packed, scale_bytes, x_f32, expected)
}

fn max_rel_err(
    expected: &[f32],
    got: &[f32],
) -> f64 {
    let mut worst = 0.0f64;
    for (&e, &g) in expected.iter().zip(got.iter()) {
        let diff = (e - g).abs() as f64;
        let denom = (e.abs() as f64).max(1e-3);
        worst = worst.max(diff / denom);
    }
    worst
}

#[uzu_test]
fn nf4_qmv_e4m3_vs_cpu() {
    let ctx = Ctx::new().expect("Metal context required");
    let qmv_bench = Nf4QmvBench::new(&ctx).expect("Nf4QmvBench build");

    // QMV consumes a full block of 512 K-elements per simdgroup iteration, so
    // K must be a multiple of 512; N must be a multiple of 32.
    let m = 2usize;
    let k = 512usize;
    let n = 128usize;

    let (w_packed, scale_bytes, x_f32, expected) = make_problem(m, k, n, false);

    let w_buf = alloc_buffer_with_data::<B, u32>(&ctx, &w_packed);
    let s_buf = alloc_buffer_with_data::<B, u8>(&ctx, &scale_bytes);
    let x_buf = bf16_buf(&ctx, &x_f32);
    let mut y_buf = ctx.create_buffer(m * n * DataType::BF16.size_in_bytes()).expect("y buf");

    let mut encoder = Encoder::new(ctx.as_ref()).unwrap();
    qmv_bench.encode(
        Nf4Variant::E4m3,
        &w_buf,
        &s_buf,
        &x_buf,
        &mut y_buf,
        k as u32,
        n as u32,
        m as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("submit");

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const bf16;
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(y_ptr, m * n) }.iter().map(|&v| v.to_f32()).collect();

    let mre = max_rel_err(&expected, &got);
    eprintln!("[nf4_qmv_e4m3] max_rel_err = {mre:.5}");
    assert!(mre <= 0.03, "Nf4QmvE4m3 vs CPU max rel err {mre:.5} > 0.03 (m={m} k={k} n={n})");
}

#[uzu_test]
fn nf4_qmm_e4m3_vs_cpu() {
    let ctx = Ctx::new().expect("Metal context required");
    let qmm_bench = Nf4QmmBench::new(&ctx).expect("Nf4QmmBench build");

    let m = 2usize;
    let k = 256usize;
    let n = 128usize;

    let (w_packed, scale_bytes, x_f32, expected) = make_problem(m, k, n, true);

    let w_buf = alloc_buffer_with_data::<B, u32>(&ctx, &w_packed);
    let s_buf = alloc_buffer_with_data::<B, u8>(&ctx, &scale_bytes);
    let x_buf = bf16_buf(&ctx, &x_f32);
    let mut y_buf = ctx.create_buffer(m * n * DataType::BF16.size_in_bytes()).expect("y buf");

    // M=2 < 48 -> Small tile (BM=8/BK=32/BN=32/WM=WN=1), as the production
    // dispatcher would pick.
    let mut encoder = Encoder::new(ctx.as_ref()).unwrap();
    qmm_bench.encode(
        Nf4Variant::E4m3,
        Nf4QmmTile::Small,
        &w_buf,
        &s_buf,
        &x_buf,
        &mut y_buf,
        k as u32,
        n as u32,
        m as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("submit");

    let y_ptr = y_buf.cpu_ptr().as_ptr() as *const bf16;
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(y_ptr, m * n) }.iter().map(|&v| v.to_f32()).collect();

    let mre = max_rel_err(&expected, &got);
    eprintln!("[nf4_qmm_e4m3] max_rel_err = {mre:.5}");
    assert!(mre <= 0.03, "Nf4QmmE4m3 vs CPU max rel err {mre:.5} > 0.03 (m={m} k={k} n={n})");
}
