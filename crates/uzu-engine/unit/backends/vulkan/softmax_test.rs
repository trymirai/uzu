use std::{
    fmt::Debug,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    time::{Duration, Instant},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::SoftmaxKernel},
        cpu::Cpu,
        vulkan::{Error, SoftmaxVulkanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// The CPU kernel through the shared trait, dispatched `repeats` times in place over `(values, sinks, row_length,
/// outer_dim, batch_dim)`; sinks are present exactly when `has_sinks`.
fn cpu_output<T: ArrayElement + Float>(
    (values, sinks, row_length, outer_dim, batch_dim): (&[T], Option<&[T]>, u32, u32, u32),
    repeats: usize,
) -> Vec<T> {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::SoftmaxKernel::new(&context, T::data_type(), sinks.is_some())
        .expect("CPU Softmax");
    let mut values_buffer = create_buffer_with_data::<Cpu, T>(&context, values);
    let sinks_buffer = sinks.map(|sinks| create_buffer_with_data::<Cpu, T>(&context, sinks));
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    for _ in 0..repeats {
        kernel.encode(&mut values_buffer, sinks_buffer.as_ref(), row_length, outer_dim, batch_dim, &mut command_buffer);
    }
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&values_buffer)
}

/// The Vulkan kernel in place over guarded ranges: one submission per entry of `submissions`, each dispatching
/// that many times, all on the same buffers. Returns the values after the guards are checked.
fn gpu_output<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &SoftmaxVulkanKernel,
    (values, sinks, row_length, outer_dim, batch_dim): (&[T], Option<&[T]>, u32, u32, u32),
    submissions: &[usize],
) -> Vec<T> {
    let sentinel = T::from(-7.0).unwrap();
    let values_buffer = fixture.guarded(values, sentinel);
    let sinks_buffer = sinks.map(|sinks| fixture.guarded(sinks, sentinel));
    for &repeats in submissions {
        let mut encoding = fixture.encoding();
        for _ in 0..repeats {
            // SAFETY: values hold outer_dim * batch_dim rows of row_length aligned `T`s and sinks hold outer_dim,
            // which bounds every index the kernel computes; nothing aliases.
            unsafe {
                kernel.encode(
                    (&values_buffer.0, values_buffer.1.clone()),
                    sinks_buffer.as_ref().map(|(buffer, range)| (buffer, range.clone())),
                    row_length,
                    outer_dim,
                    batch_dim,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        if let (Some(sinks), Some(sinks_buffer)) = (sinks, &sinks_buffer) {
            let read = KernelFixture::read_guarded(sinks_buffer, sentinel);
            assert_eq!(bytemuck::cast_slice::<T, u8>(&read), bytemuck::cast_slice::<T, u8>(sinks), "sinks changed");
        }
        KernelFixture::read_guarded(&values_buffer, sentinel)
    }
}

fn kernel<T: ArrayElement>(
    fixture: &KernelFixture,
    has_sinks: bool,
) -> SoftmaxVulkanKernel {
    SoftmaxVulkanKernel::new(&fixture.context, T::data_type(), has_sinks).expect("Vulkan Softmax")
}

/// Position of a value in the total order of its storage type, so the difference counts representable steps.
fn ordinal<T: ArrayElement>(value: T) -> i64 {
    let (bits, sign) = match *bytemuck::bytes_of(&value) {
        [a, b] => (i64::from(u16::from_ne_bytes([a, b])), 1 << 15),
        [a, b, c, d] => (i64::from(u32::from_ne_bytes([a, b, c, d])), 1 << 31),
        _ => unreachable!("Softmax storage types are 16 or 32 bits"),
    };
    if bits & sign != 0 {
        -(bits & !sign)
    } else {
        bits
    }
}

/// Test-only mathematical Softmax of the stored inputs: max, exponentials and normalizer in FP64, rounded to `T` only
/// when each full application is stored. Sinks are shared by the batch rows of their outer index.
fn reference<T: ArrayElement + Float>(
    (values, sinks, row_length, _, batch_dim): (&[T], Option<&[T]>, u32, u32, u32),
    repeats: usize,
) -> Vec<T> {
    let mut stored = values.to_vec();
    for _ in 0..repeats {
        for (row_index, row) in stored.chunks_mut(row_length as usize).enumerate() {
            let sink = sinks.map(|sinks| sinks[row_index / batch_dim as usize].to_f64().unwrap());
            let inputs = row.iter().map(|value| value.to_f64().unwrap()).collect::<Vec<_>>();
            let max = inputs.iter().chain(&sink).fold(f64::NEG_INFINITY, |max, &value| max.max(value));
            let norm = inputs.iter().chain(&sink).map(|value| (value - max).exp()).sum::<f64>();
            for (output, input) in row.iter_mut().zip(&inputs) {
                *output = T::from((input - max).exp() / norm).unwrap();
            }
        }
    }
    stored
}

/// Compares Vulkan with an expected result: NaN must occur exactly where expected; other elements must be within 2
/// representable `T` steps or, for FP32 storage, within `relative`. Returns the max absolute, relative and step errors
/// plus the number of bound violations, printing the first violation with exact bits.
fn compare<T: ArrayElement + Float + Debug>(
    expected: &[T],
    actual: &[T],
    case: &str,
    relative: f64,
) -> [f64; 4] {
    assert_eq!(expected.len(), actual.len(), "{case}: length");
    let relative = if size_of::<T>() == 4 {
        relative
    } else {
        0.0
    };
    let mut max = [0.0f64; 4];
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let values =
            format!("expected {expected:?} ({:#x}), Vulkan {actual:?} ({:#x})", ordinal(expected), ordinal(actual));
        assert_eq!(expected.is_nan(), actual.is_nan(), "{case}: element {index}: {values}");
        if expected.is_nan() {
            continue;
        }
        let (e, a) = (expected.to_f64().unwrap(), actual.to_f64().unwrap());
        let steps = (ordinal(actual) - ordinal(expected)).abs() as f64;
        let error = [(a - e).abs(), (a - e).abs() / e.abs().max(f64::MIN_POSITIVE), steps];
        if steps > 2.0 && error[1] > relative {
            if max[3] == 0.0 {
                eprintln!("{case}: element {index} exceeds the bound: {values}");
            }
            max[3] += 1.0;
        }
        max = [max[0].max(error[0]), max[1].max(error[1]), max[2].max(error[2]), max[3]];
    }
    max
}

/// Runs one GPU case and compares it with the CPU kernel (FP32 relative 1e-5) and the FP64 reference (FP32 relative
/// 2e-6), both applied once per dispatch of `submissions`.
fn check<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &SoftmaxVulkanKernel,
    case: (&[T], Option<&[T]>, u32, u32, u32),
    submissions: &[usize],
    label: &str,
) -> [[f64; 4]; 2] {
    let repeats = submissions.iter().sum();
    let gpu = gpu_output(fixture, kernel, case, submissions);
    [
        compare(&cpu_output(case, repeats), &gpu, &format!("{label} vs CPU"), 1e-5),
        compare(&reference(case, repeats), &gpu, &format!("{label} vs FP64"), 2e-6),
    ]
}

/// Prints the max errors and bound violations against the CPU kernel and the FP64 reference, then fails if any
/// element violated either bound.
fn report(
    name: &str,
    errors: impl IntoIterator<Item = [[f64; 4]; 2]>,
) {
    let totals = errors.into_iter().fold([[0.0f64; 4]; 2], |totals, errors| {
        [0, 1].map(|i| {
            let (total, error) = (totals[i], errors[i]);
            [total[0].max(error[0]), total[1].max(error[1]), total[2].max(error[2]), total[3] + error[3]]
        })
    });
    for (oracle, [absolute, relative, steps, violations]) in ["CPU", "FP64"].into_iter().zip(totals) {
        eprintln!(
            "Softmax {name} vs {oracle}: max absolute {absolute:.3e}, max relative {relative:.3e}, max {steps} storage \
             steps, {violations} bound violations"
        );
    }
    assert!(totals.iter().all(|total| total[3] == 0.0), "Softmax {name}: elements exceed the bound");
}

/// `outer * batch` rows with values in [-8, 8]; `seed` varies the pattern.
fn rows<T: Float>(
    row_length: usize,
    rows: usize,
    seed: usize,
) -> Vec<T> {
    (0..row_length * rows).map(|i| T::from(((i * 37 + seed) % 199) as f32 * 0.08 - 8.0).unwrap()).collect()
}

fn matches_cpu<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let mut errors = Vec::new();
    for has_sinks in [false, true] {
        let kernel = kernel::<T>(&fixture, has_sinks);
        let sinks = [T::from(0.5).unwrap(), T::from(-1.0).unwrap()];
        let sinks = has_sinks.then_some(&sinks[..]);
        for row_length in [1, 3, 31, 255, 256, 257, 1024, 32769] {
            let values = rows::<T>(row_length, 6, row_length);
            let case = (&values[..], sinks, row_length as u32, 2, 3);
            let label = format!("{:?} row {row_length} sinks {has_sinks}", T::data_type());
            errors.push(check(&fixture, &kernel, case, &[1], &label));
        }
        let values = rows::<T>(257, 6, 5);
        for submissions in [&[2][..], &[1, 1]] {
            let case = (&values[..], sinks, 257, 2, 3);
            let label = format!("{:?} sinks {has_sinks} submissions {submissions:?}", T::data_type());
            errors.push(check(&fixture, &kernel, case, submissions, &label));
        }
    }
    report(&format!("{:?}", T::data_type()), errors);
    fixture.assert_clean();
}

/// Uniform rows, rows of the lowest finite value (masked tails must not add to the normalizer), a dominant logit
/// and a dominant sink.
fn extreme_rows<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let (plain, sinked) = (kernel::<T>(&fixture, false), kernel::<T>(&fixture, true));
    let lowest = T::min_value();
    let mut errors = Vec::new();
    for row_length in [1, 3, 31, 255, 256, 257] {
        let values = vec![lowest; row_length * 4];
        for (kernel, sinks) in [(&plain, None), (&sinked, Some(&[lowest; 2][..]))] {
            let case = (&values[..], sinks, row_length as u32, 2, 2);
            let mass = row_length as f32 / (row_length + usize::from(sinks.is_some())) as f32;
            let label = format!("{:?} lowest row {row_length} sinks {}", T::data_type(), sinks.is_some());
            errors.push(check(&fixture, kernel, case, &[1], &label));
            let cpu_mass = cpu_output(case, 1)[..row_length].iter().map(|value| value.to_f32().unwrap()).sum::<f32>();
            assert!((cpu_mass - mass).abs() < 0.02, "{label}: CPU row mass {cpu_mass}, expected {mass}");
        }
    }
    let zeros = vec![T::zero(); 257 * 2];
    let mut dominant = vec![T::from(-20.0).unwrap(); 1024];
    dominant[517] = T::from(20.0).unwrap();
    let cases: [(&[T], Option<&[T]>, u32, u32, u32); 3] = [
        (&zeros, None, 257, 2, 1),
        (&dominant, None, 1024, 1, 1),
        (&zeros, Some(&[T::from(30.0).unwrap(), T::from(30.0).unwrap()]), 257, 2, 1),
    ];
    for (index, case) in cases.into_iter().enumerate() {
        let kernel = if case.1.is_some() {
            &sinked
        } else {
            &plain
        };
        let label = format!("{:?} extreme case {index}", T::data_type());
        errors.push(check(&fixture, kernel, case, &[1], &label));
    }
    report(&format!("{:?} extreme rows", T::data_type()), errors);
    fixture.assert_clean();
}

/// Rows containing NaN or +Inf, rows of -Inf, and NaN/+Inf/-Inf sinks shared by two batch rows produce NaN exactly
/// where the CPU kernel and the FP64 reference do.
fn nonfinite_rows<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let (plain, sinked) = (kernel::<T>(&fixture, false), kernel::<T>(&fixture, true));
    let finite = [rows::<T>(33, 1, 0), rows::<T>(33, 1, 0)].concat();
    let mut errors = Vec::new();
    for (index, poison) in [T::nan(), T::infinity()].into_iter().enumerate() {
        let mut values = finite.clone();
        values[40] = poison;
        let label = format!("{:?} nonfinite value {index}", T::data_type());
        errors.push(check(&fixture, &plain, (&values, None, 33, 1, 2), &[1], &label));
    }
    let values = vec![T::neg_infinity(); 33];
    errors.push(check(&fixture, &plain, (&values, None, 33, 1, 1), &[1], "-Inf row"));
    for sink in [T::nan(), T::infinity(), T::neg_infinity()] {
        let label = format!("{:?} sink {sink:?}", T::data_type());
        errors.push(check(&fixture, &sinked, (&finite, Some(&[sink]), 33, 1, 2), &[1], &label));
    }
    report(&format!("{:?} nonfinite rows", T::data_type()), errors);
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_f32() {
    matches_cpu::<f32>();
}

#[uzu_test]
fn matches_cpu_f16() {
    matches_cpu::<f16>();
}

#[uzu_test]
fn matches_cpu_bf16() {
    matches_cpu::<bf16>();
}

#[uzu_test]
fn extreme_rows_f32() {
    extreme_rows::<f32>();
}

#[uzu_test]
fn extreme_rows_f16() {
    extreme_rows::<f16>();
}

#[uzu_test]
fn extreme_rows_bf16() {
    extreme_rows::<bf16>();
}

#[uzu_test]
fn nonfinite_rows_all_types() {
    nonfinite_rows::<f32>();
    nonfinite_rows::<f16>();
    nonfinite_rows::<bf16>();
}

#[uzu_test]
fn zero_groups_record_nothing() {
    let fixture = KernelFixture::new();
    let kernel = kernel::<f32>(&fixture, true);
    for (row_length, outer_dim, batch_dim) in [(4, 0, 2), (4, 2, 0), (0, 2, 2)] {
        let output = gpu_output::<f32>(
            &fixture,
            &kernel,
            (&[], Some(&[0.0; 2][..outer_dim as usize]), row_length, outer_dim, batch_dim),
            &[1],
        );
        assert!(output.is_empty());
    }
    fixture.assert_clean();
}

#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    assert!(matches!(
        SoftmaxVulkanKernel::new(&fixture.context, DataType::I32, false),
        Err(Error::KernelVariant {
            kernel: "Softmax",
            ..
        })
    ));
    let values = fixture.buffer(&[1.0f32; 4]);
    let mut encoding = fixture.encoding();
    for has_sinks in [false, true] {
        let kernel = kernel::<f32>(&fixture, has_sinks);
        let wrong = (!has_sinks).then(|| (&values, 0..4));
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the optional-argument assertion fails before recording.
            kernel.encode((&values, 0..16), wrong, 4, 1, 1, &mut encoding);
        });
        assert!(catch_unwind(encode).is_err(), "has_sinks {has_sinks} accepted wrong sinks");
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded no dispatch.
    assert_eq!(unsafe { KernelFixture::read::<f32>(&values) }, [1.0; 4]);
    fixture.assert_clean();
}

/// Run alone: `cargo test ... softmax_test::throughput -- --ignored --nocapture`. Pipelines are created before
/// timing; each sample is one submission. Bytes count both passes' reads and the final write.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(
        fixture: &KernelFixture,
        (row_length, outer_dim, batch_dim): (usize, usize, usize),
    ) {
        let kernel = kernel::<T>(fixture, false);
        let length = row_length * outer_dim * batch_dim;
        let values = fixture.buffer(&rows::<T>(row_length, outer_dim * batch_dim, 1));
        let mut samples = (0..13)
            .map(|_| {
                let start = Instant::now();
                let mut encoding = fixture.encoding();
                // SAFETY: values hold every row; no sinks.
                unsafe {
                    kernel.encode(
                        (&values, 0..(length * size_of::<T>()) as u64),
                        None,
                        row_length as u32,
                        outer_dim as u32,
                        batch_dim as u32,
                        &mut encoding,
                    );
                }
                (KernelFixture::complete(encoding).gpu_execution_time(), start.elapsed())
            })
            .skip(3)
            .collect::<Vec<_>>();
        let mut median = |key: fn(&(Duration, Duration)) -> Duration| {
            samples.sort_by_key(key);
            key(&samples[samples.len() / 2])
        };
        let (gpu, wall) = (median(|sample| sample.0), median(|sample| sample.1));
        let (read, written) = (2 * length * size_of::<T>(), length * size_of::<T>());
        let rate = |time: Duration| (read + written) as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "Softmax {:?} {outer_dim}x{batch_dim}x{row_length}: read {read} B, written {written} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            T::data_type(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    for shape in [(8192, 32, 8), (131072, 1, 1)] {
        measure::<f32>(&fixture, shape);
        measure::<f16>(&fixture, shape);
        measure::<bf16>(&fixture, shape);
    }
    fixture.assert_clean();
}
