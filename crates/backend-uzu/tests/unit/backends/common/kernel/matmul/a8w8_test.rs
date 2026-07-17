use proc_macros::uzu_test;

use crate::{
    backends::common::gpu_types::{ActivationScaleStatistic, QuantizationMethod},
    tests::matmul::{QuantInput, run_quant_cpu},
};

const SEED: u64 = 0xA8B8_0011_2233_4455;

fn symmetric_input(
    m: usize,
    k: usize,
    n: usize,
    group_size: u32,
) -> QuantInput<f32> {
    QuantInput::<f32>::new(m, k, n, group_size, 8, QuantizationMethod::ScaleSymmetric, SEED)
}

fn relative_l2(
    baseline: &[f32],
    actual: &[f32],
) -> f32 {
    let numerator: f32 = baseline.iter().zip(actual).map(|(b, a)| (b - a) * (b - a)).sum();
    let denominator: f32 = baseline.iter().map(|b| b * b).sum::<f32>().max(1e-6);
    (numerator / denominator).sqrt()
}

#[uzu_test]
fn a8w8_symmetric_groupwise_absmax_matches_full_precision_activations() {
    let group_size = 32;
    for (m, k, n) in [(16usize, 128usize, 32usize), (8, 256, 64), (33, 128, 48)] {
        let baseline = run_quant_cpu(&symmetric_input(m, k, n, group_size));
        let a8w8 =
            run_quant_cpu(&symmetric_input(m, k, n, group_size).with_prepared_a(ActivationScaleStatistic::AbsMax));
        let rel = relative_l2(&baseline, &a8w8);
        assert!(rel < 0.05, "group-wise absmax A8W8 rel-L2 error {rel} too high for shape {m}x{k}x{n}");
    }
}

#[uzu_test]
fn a8w8_symmetric_rms_runs_and_is_finite() {
    let (m, k, n) = (16, 128, 32);
    let a8w8 = run_quant_cpu(&symmetric_input(m, k, n, 32).with_prepared_a(ActivationScaleStatistic::Rms));
    assert!(a8w8.iter().all(|v| v.is_finite()), "RMS A8W8 produced non-finite output");
}
