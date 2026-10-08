use std::{
    fmt::Debug,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    time::Duration,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::TensorAddScaleKernel},
        cpu::Cpu,
        vulkan::{Error, TensorAddScaleVulkanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// The CPU kernel through the shared trait, dispatched `repeats` times into one command buffer.
fn cpu_output<T: ArrayElement + Float>(
    input: &[T],
    bias: &[T],
    num_cols: u32,
    scale: f32,
    in_place: bool,
    repeats: usize,
) -> Vec<T> {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(&context, T::data_type(), in_place)
        .expect("CPU TensorAddScale");
    let input_buffer = (!in_place).then(|| create_buffer_with_data::<Cpu, T>(&context, input));
    let bias_buffer = create_buffer_with_data::<Cpu, T>(&context, bias);
    let mut output = create_buffer_with_data::<Cpu, T>(&context, input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    for _ in 0..repeats {
        kernel.encode(
            input_buffer.as_ref(),
            &bias_buffer,
            &mut output,
            num_cols,
            input.len() as u32,
            scale,
            &mut command_buffer,
        );
    }
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&output)
}

/// The Vulkan kernel over guarded ranges: one submission per entry of `submissions`, each dispatching
/// that many times, all on the same buffers. Asserts the guards are untouched and returns the output range.
fn gpu_output<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    kernel: &TensorAddScaleVulkanKernel,
    (input, bias, num_cols, scale, in_place): (&[T], &[T], u32, f32, bool),
    submissions: &[usize],
) -> Vec<T> {
    let sentinel = T::from(-7.0).unwrap();
    let initial = match in_place {
        true => input.to_vec(),
        false => vec![sentinel; input.len()],
    };
    let output = fixture.guarded(&initial, sentinel);
    let input_buffer = (!in_place).then(|| fixture.guarded(input, sentinel));
    let bias_buffer = fixture.guarded(bias, sentinel);
    for &repeats in submissions {
        let mut encoding = fixture.encoding();
        for _ in 0..repeats {
            // SAFETY: input/output ranges hold `length` aligned `T`s and bias holds `num_cols`, so every
            // index `position < length` and `position % num_cols` is inside them; output aliases only itself.
            unsafe {
                kernel.encode(
                    input_buffer.as_ref().map(|(buffer, range)| (buffer, range.clone())),
                    (&bias_buffer.0, bias_buffer.1.clone()),
                    (&output.0, output.1.clone()),
                    num_cols,
                    input.len() as u32,
                    scale,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    for (buffer, values) in input_buffer.iter().map(|buffer| (buffer, input)).chain([(&bias_buffer, bias)]) {
        unsafe { KernelFixture::assert_unchanged(buffer, sentinel, values, "read-only data") };
    }
    // SAFETY: the only command buffer writing `output` has completed.
    unsafe { KernelFixture::read_guarded(&output, sentinel) }
}

fn kernel<T: ArrayElement>(
    fixture: &KernelFixture,
    in_place: bool,
) -> TensorAddScaleVulkanKernel {
    TensorAddScaleVulkanKernel::new(&fixture.context, T::data_type(), in_place).expect("Vulkan TensorAddScale")
}

/// CPU and GPU outputs of one dispatch.
fn outputs<T: ArrayElement + Float + Debug>(
    fixture: &KernelFixture,
    case: (&[T], &[T], u32, f32, bool),
) -> (Vec<T>, Vec<T>) {
    let (input, bias, num_cols, scale, in_place) = case;
    let gpu = gpu_output(fixture, &kernel::<T>(fixture, in_place), case, &[1]);
    (cpu_output(input, bias, num_cols, scale, in_place, 1), gpu)
}

fn sequence<T: Float>(
    len: usize,
    seed: usize,
) -> Vec<T> {
    (0..len).map(|index| T::from(((index * 37 + seed) % 199) as f32 * 0.173 - 17.0).unwrap()).collect()
}

fn matches_cpu<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    for in_place in [false, true] {
        let kernel = kernel::<T>(&fixture, in_place);
        for (length, num_cols) in [(1, 1), (31, 7), (33, 5), (1003, 13), (1_000_003, 1000)] {
            let (input, bias) = (sequence::<T>(length, 3), sequence::<T>(num_cols, 11));
            let case = (&input[..], &bias[..], num_cols as u32, 0.3, in_place);
            let gpu = gpu_output(&fixture, &kernel, case, &[1]);
            let cpu = cpu_output(&input, &bias, num_cols as u32, 0.3, in_place, 1);
            KernelFixture::assert_bits(
                &cpu,
                &gpu,
                &format!("{:?} length {length} in_place {in_place}", T::data_type()),
            );
        }
    }
    fixture.assert_clean();
}

/// `1 + ulp/2` ties to even `1`; `1 + ulp + ulp/2` ties to even `1 + 2 ulp`; likewise negated.
fn rounds_ties_to_even<T: ArrayElement + Float + Debug>(ulp: f32) {
    let fixture = KernelFixture::new();
    let input = [1.0, 1.0 + ulp, -1.0, -1.0 - ulp].map(|value| T::from(value).unwrap());
    let bias = [ulp, ulp, -ulp, -ulp].map(|value| T::from(value / 2.0).unwrap());
    let (cpu, gpu) = outputs(&fixture, (&input, &bias, 4, 1.0, false));
    let expected = [1.0, 1.0 + 2.0 * ulp, -1.0, -1.0 - 2.0 * ulp].map(|value| T::from(value).unwrap());
    KernelFixture::assert_bits(&expected, &cpu, "CPU ties");
    KernelFixture::assert_bits(&cpu, &gpu, &format!("{:?} ties", T::data_type()));
    fixture.assert_clean();
}

/// NaN, infinities, overflow and signed zeros.
fn propagates_specials<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let (nan, inf, zero, max) = (T::nan(), T::infinity(), T::zero(), T::max_value());
    let input = [nan, inf, -inf, zero, -zero, -zero, max, inf, T::one()];
    let bias = [T::one(), T::one(), T::one(), zero, -zero, zero, max, -inf, nan];
    let (cpu, gpu) = outputs(&fixture, (&input, &bias, 9, 1.0, false));
    assert!(cpu[0].is_nan() && cpu[6].is_infinite() && cpu[7].is_nan() && cpu[4].is_sign_negative());
    KernelFixture::assert_bits(&cpu, &gpu, &format!("{:?} specials", T::data_type()));
    let (cpu, gpu) = outputs(&fixture, (&[zero, T::one()], &[zero, -T::one()], 2, -1.0, false));
    assert!(cpu.iter().all(|value| *value == zero && value.is_sign_negative()));
    KernelFixture::assert_bits(&cpu, &gpu, &format!("{:?} negated zeros", T::data_type()));
    fixture.assert_clean();
}

/// Stores of NaN results set the type's quiet bit, as `half` and IEEE conversions do.
fn quiets_nans<T: ArrayElement + Float + Debug>(
    signaling: [T; 2],
    quiet_bit: u32,
    bits: fn(T) -> u32,
) {
    let fixture = KernelFixture::new();
    let (cpu, gpu) = outputs(&fixture, (&signaling, &[T::zero()], 1, 1.0, false));
    for value in cpu.iter().chain(&gpu) {
        assert!(
            value.is_nan() && bits(*value) & quiet_bit != 0,
            "{:?}: {:#x} is not a quiet NaN",
            T::data_type(),
            bits(*value)
        );
    }
    fixture.assert_clean();
}

/// Subnormal operands and results. FP32 arithmetic here may flush denormals (DenormPreserve32 is false), so
/// each element must bit-match either the CPU kernel (preserved) or the CPU kernel run on operands flushed
/// to signed zero with a subnormal result flushed likewise (FTZ). Returns the elements matching only FTZ.
fn subnormals<T: ArrayElement + Float + Debug>() -> usize {
    let fixture = KernelFixture::new();
    let tiny = T::min_positive_value();
    let quarter = T::from(0.25).unwrap();
    let input = [tiny * quarter, -tiny * quarter, tiny, tiny, T::one()];
    let bias = [T::zero(), T::zero(), -tiny * T::from(0.75).unwrap(), tiny * quarter, T::zero()];
    let (cpu, gpu) = outputs(&fixture, (&input, &bias, 5, 1.0, false));
    let flush = |value: T| match value != T::zero() && value.abs() < tiny {
        true if value.is_sign_negative() => -T::zero(),
        true => T::zero(),
        false => value,
    };
    let ftz = cpu_output(&input.map(flush), &bias.map(flush), 5, 1.0, false, 1).into_iter().map(flush);
    let bits = |value: &T| bytemuck::bytes_of(value).iter().rev().map(|byte| format!("{byte:02x}")).collect::<String>();
    let mut flushed = 0;
    for (index, ((expected, flushed_expected), actual)) in cpu.iter().zip(ftz).zip(&gpu).enumerate() {
        let (expected, flushed_expected, actual) = (bits(expected), bits(&flushed_expected), bits(actual));
        eprintln!(
            "{:?} subnormal element {index}: CPU {expected}, FTZ {flushed_expected}, Vulkan {actual}",
            T::data_type()
        );
        assert!(actual == expected || actual == flushed_expected, "element {index} is neither preserved nor FTZ");
        flushed += usize::from(actual != expected);
    }
    fixture.assert_clean();
    flushed
}

/// A thousand dependent in-place dispatches in one submission, then two submissions on the same buffers where
/// the second consumes the first's output.
fn repeats_dependent_dispatches<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let kernel = kernel::<T>(&fixture, true);
    let (input, bias) =
        (sequence::<T>(1003, 5), sequence::<T>(13, 7).iter().map(|&b| b * T::from(0.01).unwrap()).collect::<Vec<_>>());
    for submissions in [&[1000][..], &[7, 5]] {
        let gpu = gpu_output(&fixture, &kernel, (&input, &bias, 13, 1.0, true), submissions);
        let cpu = cpu_output(&input, &bias, 13, 1.0, true, submissions.iter().sum());
        KernelFixture::assert_bits(&cpu, &gpu, &format!("{:?} submissions {submissions:?}", T::data_type()));
    }
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
fn rounds_ties_to_even_all_types() {
    rounds_ties_to_even::<f32>(f32::EPSILON);
    rounds_ties_to_even::<f16>(f16::EPSILON.to_f32());
    rounds_ties_to_even::<bf16>(bf16::EPSILON.to_f32());
}

#[uzu_test]
fn propagates_specials_all_types() {
    propagates_specials::<f32>();
    propagates_specials::<f16>();
    propagates_specials::<bf16>();
}

#[uzu_test]
fn stores_quiet_nans() {
    quiets_nans([f32::from_bits(0x7f80_0001), f32::from_bits(0xff80_0001)], 0x0040_0000, f32::to_bits);
    quiets_nans([f16::from_bits(0x7c01), f16::from_bits(0xfc01)], 0x0200, |value| value.to_bits().into());
    quiets_nans([bf16::from_bits(0x7f81), bf16::from_bits(0xff81)], 0x0040, |value| value.to_bits().into());
}

#[uzu_test]
fn measures_subnormals() {
    assert_eq!(subnormals::<f16>(), 0, "F16 subnormals are normal in FP32 and must round-trip exactly");
    subnormals::<f32>();
    subnormals::<bf16>();
}

#[uzu_test]
fn repeats_dependent_dispatches_all_types() {
    repeats_dependent_dispatches::<f32>();
    repeats_dependent_dispatches::<f16>();
    repeats_dependent_dispatches::<bf16>();
}

#[uzu_test]
fn zero_length_records_nothing() {
    let fixture = KernelFixture::new();
    for in_place in [false, true] {
        let output =
            gpu_output::<f32>(&fixture, &kernel::<f32>(&fixture, in_place), (&[], &[], 1, 1.0, in_place), &[1]);
        assert!(output.is_empty());
    }
    fixture.assert_clean();
}

#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    assert!(matches!(
        TensorAddScaleVulkanKernel::new(&fixture.context, DataType::I32, false),
        Err(Error::KernelVariant {
            kernel: "TensorAddScale",
            ..
        })
    ));
    let values = fixture.buffer(&[1.0f32; 4]);
    let mut encoding = fixture.encoding();
    for in_place in [false, true] {
        let kernel = kernel::<f32>(&fixture, in_place);
        let wrong = in_place.then_some((&values, 0..16));
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the optional-argument assertion fails before recording.
            kernel.encode(wrong, (&values, 0..16), (&values, 0..16), 4, 4, 1.0, &mut encoding);
        });
        assert!(catch_unwind(encode).is_err(), "in_place {in_place} accepted a wrong optional input");
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded no dispatch.
    assert_eq!(unsafe { KernelFixture::read::<f32>(&values) }, [1.0; 4]);
    fixture.assert_clean();
}

/// Run explicitly: `cargo test ... throughput -- --ignored --nocapture`. Pipelines are created before timing.
#[uzu_test]
#[ignore]
fn throughput() {
    const LENGTH: usize = 1 << 25;
    const NUM_COLS: usize = 4096;
    fn measure<T: ArrayElement + Float + Debug>(fixture: &KernelFixture) {
        let kernel = kernel::<T>(fixture, false);
        let (input, bias, output) = (
            fixture.buffer(&sequence::<T>(LENGTH, 1)),
            fixture.buffer(&sequence::<T>(NUM_COLS, 2)),
            fixture.buffer(&std::iter::repeat_n(T::zero(), LENGTH).collect::<Vec<_>>()),
        );
        let bytes = |len: usize| 0..(len * size_of::<T>()) as u64;
        let (gpu, wall) = fixture.median_times(|encoding| {
            // SAFETY: input/output hold LENGTH elements and bias NUM_COLS; output aliases nothing.
            unsafe {
                kernel.encode(
                    Some((&input, bytes(LENGTH))),
                    (&bias, bytes(NUM_COLS)),
                    (&output, bytes(LENGTH)),
                    NUM_COLS as u32,
                    LENGTH as u32,
                    0.5,
                    encoding,
                );
            }
        });
        let (read, written) = ((LENGTH + NUM_COLS) * size_of::<T>(), LENGTH * size_of::<T>());
        let rate = |time: Duration| (read + written) as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "TensorAddScale {:?} x{LENGTH}: read {read} B, written {written} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            T::data_type(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    measure::<f32>(&fixture);
    measure::<f16>(&fixture);
    measure::<bf16>(&fixture);
    fixture.assert_clean();
}
