//! The trellis GEMM on the production `MatmulKernel` dispatch.
//!
//! Two checks, and they fail for different reasons: a numeric one against the
//! shared f64 oracle, swept over the M values that select each of the three
//! compiled tiles, a ragged `N` and every split-K factor; and the shared
//! bit-exact probe, which pins the kernel's window read and the tape packing
//! against the trellis recurrence with no tolerance at all. Both come from
//! [`crate::tests::matmul::trellis_fixture`].
//!
//! TOLERANCE. Int8 activations force a bf16 `D` (`gemm.metal` constrains
//! `A_PROLOGUE == Int8Symmetric` to `AT == DT == bfloat`). bf16 keeps 7 mantissa
//! bits, so storing the result costs up to `2^-8 = 3.9e-3` relative on its own,
//! and split-K rounds each partial to bf16 before the reduce sums them, for one
//! more of the same. The bar is therefore 8e-3 -- a property of the OUTPUT TYPE,
//! not of this kernel: the int32 accumulator is exact (`|product| <= 127*127`
//! and `K <= 8192`, so under 2^27), the probe verifies the decode with no
//! tolerance at all, and the measured maximum across this sweep is 3.6e-3.

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::kernel::{
        activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
        matmul::{MatmulDOps, trellis_format::TRELLIS_BLOCK_K},
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        matmul::trellis_fixture::{
            CONFIG, Fixture, assert_probe_row, max_relative_error, oracle, probe_points, row_scales,
        },
        util::shared_metal_context,
    },
};

/// Weight rows and the reduction depth. `N = 130` is ragged against all three
/// compiled N tiles (32, 64, 64), so both arms of `TrellisCursor::stage`'s
/// hoisted N guard run; `K` is a multiple of the 128-column activation group and
/// of the staging block (`TRELLIS_BLOCK_K`).
const N: u32 = 130;
const K: u32 = 512;

/// `d = a_codes * weight_codes^T`, scaled the way the epilogue scales it.
fn run(
    context: &MetalContext,
    fixture: &Fixture,
    a_codes: &[i8],
    a_scales: &[f32],
    m: u32,
    k: u32,
    split_k: u32,
) -> Vec<bf16> {
    let a = alloc_allocation_with_data::<Metal, i8>(context, a_codes);
    let scales = alloc_allocation_with_data::<Metal, f32>(context, a_scales);
    let mut d = alloc_allocation::<Metal, bf16>(context, (m * N) as usize);

    let arguments = MatmulArguments {
        a: MatmulA::Int8Symmetric {
            values: &a,
            scales: &scales,
            group_sums: None,
            group_size: ACTIVATION_SCALE_GROUP_SIZE,
        },
        b: fixture.weights(),
        b_leading_dimension: None,
        b_transpose: true,
        d: &mut d,
        d_transform: MatmulDOps::none(),
        gather_indices: None,
        m,
        n: N,
        k,
    };
    let shape = MatmulShape::from_arguments(&arguments);
    let mut plan = GemmProblem::new(
        shape,
        DataType::BF16,
        DataType::BF16,
        context.supports_mxu,
        context.apple_gpu_family,
        context.gpu_core_count,
    )
    .select_plan();
    plan.split_k = split_k;

    let mut kernel = GemmKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16).expect("GemmKernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
    kernel.encode_plan(arguments, plan, &mut encoder).expect("trellis GEMM encode failed");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<Metal, bf16>(&d)
}

/// Deterministic int8 activations plus their per-group f32 scales.
fn activations(
    m: u32,
    k: u32,
) -> (Vec<i8>, Vec<f32>) {
    let codes = (0..m * k).map(|i| ((i * 37 + i / 11) % 191) as i32 as i8).collect();
    let scales = (0..m * (k / ACTIVATION_SCALE_GROUP_SIZE)).map(|i| 0.0009 + 0.0003 * ((i % 7) as f32)).collect();
    (codes, scales)
}

#[uzu_test]
fn trellis_gemm_matches_oracle() {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    let fixture = Fixture::new(&context, CONFIG, N, K, row_scales(N));

    // M picks the tile: 16 -> 16x32, 17 and 32 -> 32x64, 64 and 128 -> 64x64.
    // Split-K is only legal where `M * N` is a multiple of four, which is what
    // the shipped reduce kernel requires.
    for m in [16u32, 17, 32, 64, 128] {
        let (a_codes, a_scales) = activations(m, K);
        let groups = (K / ACTIVATION_SCALE_GROUP_SIZE) as usize;
        let expected = oracle(&fixture, m, N, K, |row, index| {
            f64::from(a_codes[row * K as usize + index])
                * f64::from(a_scales[row * groups + index / ACTIVATION_SCALE_GROUP_SIZE as usize])
        });
        // A forced split has to leave every partition at least one staging
        // block of K, which the shipped policy guarantees and this test has to
        // reproduce: `split_k * TRELLIS_BLOCK_K <= K`. Past that, partitions
        // read off the end of the tape.
        let max_split = K / TRELLIS_BLOCK_K;
        let splits: Vec<u32> = if (m * N).is_multiple_of(4) {
            [1, 4, 8].into_iter().filter(|split_k| *split_k <= max_split).collect()
        } else {
            vec![1]
        };
        for split_k in splits {
            let got = run(&context, &fixture, &a_codes, &a_scales, m, K, split_k);
            let error = max_relative_error(&got[..(m * N) as usize], &expected);
            assert!(error < 8e-3, "M={m} split_k={split_k}: max relative error {error:e}");
        }
    }
}

#[uzu_test]
fn trellis_gemm_states_are_bit_exact() {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    // A longer row so the probe reaches steps whose window read is misaligned in
    // every one of the 32 possible ways.
    let k = 1024u32;
    let fixture = Fixture::new(&context, CONFIG, N, k, vec![bf16::ONE; N as usize]);

    let probes = probe_points(CONFIG.steps(k));
    let m = probes.len() as u32;
    let mut a_codes = vec![0i8; (m * k) as usize];
    for (row, &(step, coordinate)) in probes.iter().enumerate() {
        a_codes[row * k as usize + (step * 4 + coordinate) as usize] = 1;
    }
    let a_scales = vec![1.0f32; (m * (k / ACTIVATION_SCALE_GROUP_SIZE)) as usize];

    let got = run(&context, &fixture, &a_codes, &a_scales, m, k, 1);
    for (row, &(step, coordinate)) in probes.iter().enumerate() {
        assert_probe_row(&fixture, &got[row * N as usize..(row + 1) * N as usize], step, coordinate, "gemm");
    }
}
