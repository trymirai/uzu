use std::{fmt::Debug, mem::size_of};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::LogitTransformKernel},
        cpu::Cpu,
        vulkan::{Error, LogitTransformVulkanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// The CPU kernel through the shared trait over `(logits, scale, soft_cap, has_soft_cap)`, dispatched `repeats`
/// times in place.
fn cpu_output<T: ArrayElement + Float>(
    (logits, scale, soft_cap, has_soft_cap): (&[T], f32, f32, bool),
    repeats: usize,
) -> Vec<T> {
    let context = create_context::<Cpu>();
    let kernel =
        <<Cpu as Backend>::Kernels as Kernels>::LogitTransformKernel::new(&context, T::data_type(), has_soft_cap)
            .expect("CPU LogitTransform");
    let mut buffer = create_buffer_with_data::<Cpu, T>(&context, logits);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    for _ in 0..repeats {
        kernel.encode(&mut buffer, logits.len() as u32, scale, soft_cap, &mut command_buffer);
    }
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&buffer)
}

/// Test-only reference in the Metal order: the FP32 scaled logit rounded to `T`, then the soft cap in FP64, rounded
/// to `T` when stored.
fn metal_order<T: ArrayElement + Float>(
    (logits, scale, soft_cap, has_soft_cap): (&[T], f32, f32, bool),
    repeats: usize,
) -> Vec<T> {
    let mut stored = logits.to_vec();
    for _ in 0..repeats {
        for value in &mut stored {
            *value = T::from(value.to_f32().unwrap() * scale).unwrap();
            if has_soft_cap {
                let cap = f64::from(soft_cap);
                *value = T::from(cap * (value.to_f64().unwrap() / cap).tanh()).unwrap();
            }
        }
    }
    stored
}

/// The Vulkan kernel in place over a guarded range: one submission per entry of `submissions`, each dispatching that
/// many times. Returns the logits after the guards are checked.
fn gpu_output<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &LogitTransformVulkanKernel,
    (logits, scale, soft_cap, _): (&[T], f32, f32, bool),
    submissions: &[usize],
) -> Vec<T> {
    let sentinel = T::from(-7.0).unwrap();
    let buffer = fixture.guarded(logits, sentinel);
    for &repeats in submissions {
        let mut encoding = fixture.encoding();
        for _ in 0..repeats {
            // SAFETY: the range holds `length` aligned `T`s, the only indices the kernel touches.
            unsafe {
                kernel.encode((&buffer.0, buffer.1.clone()), logits.len() as u32, scale, soft_cap, &mut encoding);
            }
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using the buffer has completed.
    unsafe { KernelFixture::read_guarded(&buffer, sentinel) }
}

fn kernel<T: ArrayElement>(
    fixture: &KernelFixture,
    has_soft_cap: bool,
) -> LogitTransformVulkanKernel {
    LogitTransformVulkanKernel::new(&fixture.context, T::data_type(), has_soft_cap).expect("Vulkan LogitTransform")
}

/// Compares one case against the CPU kernel and the Metal-order reference, F32 within relative 2e-6 or absolute 1e-7
/// and BF16 within 2 steps, labeling the errors by oracle, type and soft cap.
fn check<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    case: (&[T], f32, f32, bool),
    submissions: &[usize],
) -> [(String, [f64; 4]); 2] {
    let repeats = submissions.iter().sum();
    let gpu = gpu_output(fixture, &kernel::<T>(fixture, case.3), case, submissions);
    let label = |oracle: &str| format!("{:?} soft_cap {} vs {oracle}", T::data_type(), case.3);
    let detail = |oracle: &str| format!("{} length {} submissions {submissions:?}", label(oracle), case.0.len());
    [
        (label("CPU"), KernelFixture::compare(&cpu_output(case, repeats), &gpu, &detail("CPU"), 2e-6, 1e-7)),
        (
            label("Metal order"),
            KernelFixture::compare(&metal_order(case, repeats), &gpu, &detail("Metal order"), 2e-6, 1e-7),
        ),
    ]
}

/// Logits in [-100, 100], beyond typical soft caps; `seed` varies the pattern.
fn logits<T: Float>(
    length: usize,
    seed: usize,
) -> Vec<T> {
    (0..length).map(|i| T::from(((i * 7919 + seed) % 2003) as f32 / 10.0 - 100.0).unwrap()).collect()
}

/// Odd lengths, both soft-cap modes, a vocabulary-sized row, and chained in-place dispatches within and across
/// submissions; then infinities, NaN and overflow.
fn matches_cpu<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let mut errors = Vec::new();
    for has_soft_cap in [false, true] {
        for length in [1, 31, 255, 256, 257, 4097, 151_936] {
            let values = logits::<T>(length, length);
            errors.extend(check(&fixture, (&values, 0.7, 30.0, has_soft_cap), &[1]));
        }
        let values = logits::<T>(1000, 3);
        for submissions in [&[3][..], &[2, 1]] {
            errors.extend(check(&fixture, (&values, 1.3, 50.0, has_soft_cap), submissions));
        }
        let specials =
            [T::nan(), T::infinity(), T::neg_infinity(), T::max_value(), T::min_value(), T::zero(), -T::zero()];
        errors.extend(check(&fixture, (&specials, 2.0, 30.0, has_soft_cap), &[1]));
    }
    KernelFixture::report("LogitTransform", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_f32() {
    matches_cpu::<f32>();
}

#[uzu_test]
fn matches_cpu_bf16() {
    matches_cpu::<bf16>();
}

/// The F32 soft cap by |x| = |logit / cap| against CPU and FP64: log-spaced |x| from 1e-6 to 20, every FP32 value
/// within 4096 steps of ±0.1 where the kernel switches from its Taylor polynomial to tanh, and a dense sweep of
/// [-0.1, 0.1], with caps 1 (x is the logit exactly) and 30. Signed zeros keep their sign.
#[uzu_test]
fn soft_cap_matches_fp64_by_magnitude() {
    let fixture = KernelFixture::new();
    let log_spaced = (0..200_000).map(|i| {
        10f32.powf(-6.0 + 7.3 * (i / 2) as f32 / 100_000.0)
            * if i % 2 == 0 {
                1.0
            } else {
                -1.0
            }
    });
    let boundary = (0..8192).flat_map(|k| [1.0, -1.0].map(|sign| sign * f32::from_bits(0.1f32.to_bits() - 4096 + k)));
    let dense = (0..=200_000).map(|i| i as f32 / 1_000_000.0 - 0.1);
    let xs = log_spaced.chain(boundary).chain(dense).collect::<Vec<_>>();
    let edges = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 4.0, f32::INFINITY];
    let mut errors = Vec::new();
    for bin in edges.windows(2) {
        for cap in [1.0f32, 30.0] {
            let logits = xs.iter().filter(|x| (bin[0]..bin[1]).contains(&x.abs())).map(|x| x * cap).collect::<Vec<_>>();
            let label = format!("|x| in [{:.0e}, {:.0e}) cap {cap}", bin[0], bin[1]);
            errors.extend(
                check(&fixture, (&logits, 1.0, cap, true), &[1])
                    .map(|(oracle, error)| (format!("{label} {oracle}"), error)),
            );
        }
    }
    let zeros = [0.0f32, -0.0];
    let signed = gpu_output(&fixture, &kernel::<f32>(&fixture, true), (&zeros, 1.0, 30.0, true), &[1]);
    assert_eq!(signed.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), zeros.map(f32::to_bits), "signed zeros");
    KernelFixture::report("LogitTransform", errors);
    fixture.assert_clean();
}

#[uzu_test]
fn zero_length_records_nothing() {
    let fixture = KernelFixture::new();
    for has_soft_cap in [false, true] {
        assert!(
            gpu_output::<f32>(&fixture, &kernel::<f32>(&fixture, has_soft_cap), (&[], 1.0, 30.0, has_soft_cap), &[1])
                .is_empty()
        );
    }
    fixture.assert_clean();
}

/// Only the F32 and BF16 model storage types have variants; the kernel has no optional arguments.
#[uzu_test]
fn rejects_invalid_types() {
    let fixture = KernelFixture::new();
    for data_type in [DataType::F16, DataType::I32] {
        assert!(matches!(
            LogitTransformVulkanKernel::new(&fixture.context, data_type, true),
            Err(Error::KernelVariant {
                kernel: "LogitTransform",
                ..
            })
        ));
    }
    fixture.assert_clean();
}

/// Run alone: `cargo test ... logit_transform_test::throughput -- --ignored --nocapture`. Pipelines are created before
/// timing; each sample is one submission, applied in place to the previous sample's output.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(
        fixture: &KernelFixture,
        (rows, vocabulary): (usize, usize),
        has_soft_cap: bool,
        spread: f32,
    ) {
        let kernel = kernel::<T>(fixture, has_soft_cap);
        let length = rows * vocabulary;
        let values = logits::<f32>(length, 1).into_iter().map(|value| T::from(value * spread / 100.0).unwrap());
        let buffer = fixture.buffer(&values.collect::<Vec<_>>());
        let bytes = length * size_of::<T>();
        let (gpu, wall) = fixture.median_times(|encoding| {
            // SAFETY: the buffer holds `length` elements.
            unsafe { kernel.encode((&buffer, 0..bytes as u64), length as u32, 1.0, 30.0, encoding) }
        });
        let rate = |time: std::time::Duration| 2.0 * bytes as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "LogitTransform {:?} {rows}x{vocabulary} soft_cap {has_soft_cap} logits in ±{spread}: read {bytes} B, written {bytes} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            T::data_type(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    // Wide logits mostly take tanh; small ones, typical of model outputs relative to a soft cap of 30, the polynomial.
    for shape in [(1, 151_936), (64, 151_936)] {
        for (has_soft_cap, spread) in [(false, 100.0), (true, 100.0), (true, 3.0)] {
            measure::<f32>(&fixture, shape, has_soft_cap, spread);
            measure::<bf16>(&fixture, shape, has_soft_cap, spread);
        }
    }
    fixture.assert_clean();
}
