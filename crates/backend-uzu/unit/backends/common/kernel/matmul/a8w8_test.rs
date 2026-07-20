use proc_macros::uzu_test;
use rstest::rstest;

use crate::{
    backends::common::gpu_types::QuantizationMethod,
    tests::matmul::{QuantInput, run_quant_cpu},
};

const SEED: u64 = 0xA8B8_0011_2233_4455;

fn relative_l2(
    baseline: &[f32],
    actual: &[f32],
) -> f32 {
    let numerator: f32 = baseline.iter().zip(actual).map(|(b, a)| (b - a) * (b - a)).sum();
    let denominator: f32 = baseline.iter().map(|b| b * b).sum::<f32>().max(1e-6);
    (numerator / denominator).sqrt()
}

#[rstest]
#[test_attr(uzu_test)]
fn a8w8_cpu_min_max_symmetric_group_32_matches_full_precision_activations() {
    let group_size = 32;
    for method in [QuantizationMethod::ScaleSymmetric, QuantizationMethod::ScaleZeroPoint] {
        for (m, k, n) in [(16usize, 128usize, 32usize), (8, 256, 64)] {
            let baseline = run_quant_cpu(&QuantInput::<f32>::new(m, k, n, group_size, 8, method, SEED));
            let a8w8 = run_quant_cpu(&QuantInput::<f32>::new(m, k, n, group_size, 8, method, SEED).with_prepared_a());
            let rel = relative_l2(&baseline, &a8w8);
            assert!(rel < 0.08, "{method:?} rel-L2 {rel} too high for shape {m}x{k}x{n}");
            assert!(a8w8.iter().all(|value| value.is_finite()));
        }
    }
}
