use std::{fmt::Debug, mem::size_of};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::kernel_fixture::KernelFixture;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::PoolingMeanKernel},
        cpu::Cpu,
        vulkan::{Error, PoolingMeanVulkanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// The CPU kernel through the shared trait over `(input, seq_len, hidden_dim, batch_size)`, with `input` of shape
/// [batch_size, seq_len, hidden_dim].
fn cpu_output<T: ArrayElement + Float>((input, seq_len, hidden_dim, batch_size): (&[T], u32, u32, u32)) -> Vec<T> {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::PoolingMeanKernel::new(&context, T::data_type())
        .expect("CPU PoolingMean");
    // CPU buffers cannot be empty; with an empty sequence the kernel reads none of this placeholder.
    let placeholder = [T::zero()];
    let input = create_buffer_with_data::<Cpu, T>(
        &context,
        if input.is_empty() {
            &placeholder
        } else {
            input
        },
    );
    let mut output = create_buffer_with_data::<Cpu, T>(&context, &vec![T::zero(); (hidden_dim * batch_size) as usize]);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    kernel.encode(&input, &mut output, seq_len, hidden_dim, batch_size, &mut command_buffer);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&output)
}

/// The Vulkan kernel over guarded ranges, dispatched once per submission in `submissions` separate submissions.
/// Asserts the input and every guard are unchanged and returns the output.
fn gpu_output<T: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &PoolingMeanVulkanKernel,
    (input, seq_len, hidden_dim, batch_size): (&[T], u32, u32, u32),
    submissions: usize,
) -> Vec<T> {
    let sentinel = T::from(-7.0).unwrap();
    let input_buffer = fixture.guarded(input, sentinel);
    let output = fixture.guarded(&vec![sentinel; (hidden_dim * batch_size) as usize], sentinel);
    for _ in 0..submissions {
        let mut encoding = fixture.encoding();
        // SAFETY: input holds batch_size * seq_len * hidden_dim and output batch_size * hidden_dim aligned `T`s, which
        // bound every index; they do not alias.
        unsafe {
            kernel.encode(
                (&input_buffer.0, input_buffer.1.clone()),
                (&output.0, output.1.clone()),
                seq_len,
                hidden_dim,
                batch_size,
                &mut encoding,
            );
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        KernelFixture::assert_unchanged(&input_buffer, sentinel, input, "input");
        KernelFixture::read_guarded(&output, sentinel)
    }
}

fn kernel<T: ArrayElement>(fixture: &KernelFixture) -> PoolingMeanVulkanKernel {
    PoolingMeanVulkanKernel::new(&fixture.context, T::data_type()).expect("Vulkan PoolingMean")
}

/// Values in [-50, 50]; `seed` varies the pattern.
fn values<T: Float>(
    length: usize,
    seed: usize,
) -> Vec<T> {
    (0..length).map(|i| T::from(((i * 7919 + seed) % 1009) as f32 / 10.09 - 50.0).unwrap()).collect()
}

/// Asserts NaN and infinities occur exactly where the CPU has them and every finite output is within 1 storage step:
/// both sum in the same FP32 order, but the final FP32 division may differ by an ulp, which can straddle a 16-bit
/// rounding. Returns the number of differing outputs and the most steps.
fn assert_within_step<T: ArrayElement + Float + Debug>(
    expected: &[T],
    actual: &[T],
    case: &str,
) -> (usize, i64) {
    assert_eq!(expected.len(), actual.len(), "{case}: length");
    let (mut differing, mut most) = (0, 0);
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let steps = (KernelFixture::ordinal(expected) - KernelFixture::ordinal(actual)).abs();
        let within = match (expected.is_nan(), expected.is_finite()) {
            (true, _) => actual.is_nan(),
            (false, false) => steps == 0,
            (false, true) => actual.is_finite() && steps <= 1,
        };
        assert!(within, "{case}: element {index}: CPU {expected:?}, Vulkan {actual:?}");
        if !expected.is_nan() {
            (differing, most) = (differing + usize::from(steps != 0), most.max(steps));
        }
    }
    (differing, most)
}

/// Hidden and batch sizes that do not divide the workgroup, sequence lengths from 1 to 2049, then infinities, NaN
/// and FP32 overflow of the sum, against the CPU kernel within `assert_within_step`.
fn matches_cpu<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let kernel = kernel::<T>(&fixture);
    let (mut differing, mut most, mut total) = (0, 0, 0);
    for (seq_len, hidden_dim, batch_size) in
        [(1, 1, 1), (3, 17, 5), (7, 33, 17), (1, 4096, 3), (128, 1000, 3), (513, 4096, 2), (2049, 31, 1)]
    {
        let input = values::<T>((seq_len * hidden_dim * batch_size) as usize, seq_len as usize);
        let case = (&input[..], seq_len, hidden_dim, batch_size);
        let label = format!("{:?} seq {seq_len} hidden {hidden_dim} batch {batch_size}", T::data_type());
        let cpu = cpu_output(case);
        let (count, steps) = assert_within_step(&cpu, &gpu_output(&fixture, &kernel, case, 1), &label);
        (differing, most, total) = (differing + count, most.max(steps), total + cpu.len());
    }
    let mut input = values::<T>(4 * 3 * 2, 0);
    input[..8].copy_from_slice(&[
        T::nan(),
        T::infinity(),
        T::neg_infinity(),
        T::max_value(),
        T::nan(),
        T::infinity(),
        T::infinity(),
        T::max_value(),
    ]);
    input[12..16].copy_from_slice(&[T::max_value(), T::max_value(), T::neg_infinity(), T::max_value()]);
    let case = (&input[..], 3, 4, 2);
    let label = format!("{:?} nonfinite", T::data_type());
    let (count, steps) = assert_within_step(&cpu_output(case), &gpu_output(&fixture, &kernel, case, 2), &label);
    eprintln!(
        "PoolingMean {:?}: {} of {} outputs differ from the CPU, at most {} storage steps",
        T::data_type(),
        differing + count,
        total + input.len() / 3,
        most.max(steps)
    );
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

/// An empty sequence still writes every [batch, hidden] output, NaN like the CPU's 0 / 0, reading nothing from its
/// empty input span.
fn empty_sequence<T: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let case = (&[][..], 0, 17, 3);
    let gpu = gpu_output::<T>(&fixture, &kernel::<T>(&fixture), case, 1);
    assert!(gpu.len() == 51 && gpu.iter().all(|value| value.is_nan()));
    KernelFixture::assert_bits(&cpu_output(case), &gpu, &format!("{:?} empty sequence", T::data_type()));
    fixture.assert_clean();
}

#[uzu_test]
fn empty_sequence_all_types() {
    empty_sequence::<f32>();
    empty_sequence::<f16>();
    empty_sequence::<bf16>();
}

/// Zero hidden or batch sizes dispatch no group and write nothing.
#[uzu_test]
fn zero_outputs_record_nothing() {
    let fixture = KernelFixture::new();
    let kernel = kernel::<f32>(&fixture);
    for (seq_len, hidden_dim, batch_size) in [(4, 0, 2), (4, 3, 0)] {
        assert!(gpu_output::<f32>(&fixture, &kernel, (&[], seq_len, hidden_dim, batch_size), 1).is_empty());
    }
    fixture.assert_clean();
}

#[uzu_test]
fn rejects_invalid_types() {
    let fixture = KernelFixture::new();
    assert!(matches!(
        PoolingMeanVulkanKernel::new(&fixture.context, DataType::I32),
        Err(Error::KernelVariant {
            kernel: "PoolingMean",
            ..
        })
    ));
    fixture.assert_clean();
}

/// Run alone: `cargo test ... pooling_mean_test::throughput -- --ignored --nocapture`. Pipelines are created before
/// timing; each sample is one submission.
#[uzu_test]
#[ignore]
fn throughput() {
    fn measure<T: ArrayElement + Float>(
        fixture: &KernelFixture,
        (batch_size, seq_len, hidden_dim): (usize, usize, usize),
    ) {
        let kernel = kernel::<T>(fixture);
        let (read, written) =
            (batch_size * seq_len * hidden_dim * size_of::<T>(), batch_size * hidden_dim * size_of::<T>());
        let input = fixture.buffer(&values::<T>(read / size_of::<T>(), 1));
        let output = fixture.buffer(&vec![T::zero(); written / size_of::<T>()]);
        let (gpu, wall) = fixture.median_times(|encoding| {
            // SAFETY: input and output hold every element the shape indexes; they do not alias.
            unsafe {
                kernel.encode(
                    (&input, 0..read as u64),
                    (&output, 0..written as u64),
                    seq_len as u32,
                    hidden_dim as u32,
                    batch_size as u32,
                    encoding,
                )
            }
        });
        let rate = |time: std::time::Duration| (read + written) as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "PoolingMean {:?} batch {batch_size} seq {seq_len} hidden {hidden_dim}: read {read} B, written {written} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            T::data_type(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    for shape in [(1, 512, 768), (8, 512, 1024), (32, 128, 4096), (2, 4096, 4096)] {
        measure::<f32>(&fixture, shape);
        measure::<f16>(&fixture, shape);
        measure::<bf16>(&fixture, shape);
    }
    fixture.assert_clean();
}
