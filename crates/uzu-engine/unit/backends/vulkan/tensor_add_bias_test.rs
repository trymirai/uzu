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
        common::{Backend, Context, Kernels, kernel::TensorAddBiasKernel},
        cpu::Cpu,
        vulkan::{Error, TensorAddBiasVulkanKernel},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// Runs `$check::<T, BiasT>()` for all nine storage type combinations.
macro_rules! for_each_combination {
    ($check:ident) => {
        $check::<f32, f32>();
        $check::<f32, f16>();
        $check::<f32, bf16>();
        $check::<f16, f32>();
        $check::<f16, f16>();
        $check::<f16, bf16>();
        $check::<bf16, f32>();
        $check::<bf16, f16>();
        $check::<bf16, bf16>();
    };
}

/// The CPU kernel through the shared trait, dispatched `repeats` times into one command buffer.
fn cpu_output<T: ArrayElement + Float, B: ArrayElement + Float>(
    (input, bias, num_cols, in_place): (&[T], &[B], u32, bool),
    repeats: usize,
) -> Vec<T> {
    let context = create_context::<Cpu>();
    let kernel = <<Cpu as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(
        &context,
        T::data_type(),
        B::data_type(),
        in_place,
    )
    .expect("CPU TensorAddBias");
    let input_buffer = (!in_place).then(|| create_buffer_with_data::<Cpu, T>(&context, input));
    let bias_buffer = create_buffer_with_data::<Cpu, B>(&context, bias);
    let mut output = create_buffer_with_data::<Cpu, T>(&context, input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
    for _ in 0..repeats {
        kernel.encode(
            input_buffer.as_ref(),
            &bias_buffer,
            &mut output,
            num_cols,
            input.len() as u32,
            &mut command_buffer,
        );
    }
    submit_command_buffer(command_buffer);
    buffer_to_vec::<Cpu, T>(&output)
}

/// The Vulkan kernel over guarded ranges: one submission per entry of `submissions`, each dispatching that many
/// times, all on the same buffers. Returns the output range after the guards are checked.
fn gpu_output<T: ArrayElement + Float, B: ArrayElement + Float>(
    fixture: &KernelFixture,
    kernel: &TensorAddBiasVulkanKernel,
    (input, bias, num_cols, in_place): (&[T], &[B], u32, bool),
    submissions: &[usize],
) -> Vec<T> {
    let sentinel = T::from(-7.0).unwrap();
    let initial = match in_place {
        true => input.to_vec(),
        false => vec![sentinel; input.len()],
    };
    let output = fixture.guarded(&initial, sentinel);
    let input_buffer = (!in_place).then(|| fixture.guarded(input, sentinel));
    let bias_buffer = fixture.guarded(bias, B::from(-7.0).unwrap());
    for &repeats in submissions {
        let mut encoding = fixture.encoding();
        for _ in 0..repeats {
            // SAFETY: input/output ranges hold `length` aligned `T`s and bias holds `num_cols` `B`s, so every
            // index `position < length` and `position % num_cols` is inside them; output aliases only itself.
            unsafe {
                kernel.encode(
                    input_buffer.as_ref().map(|(buffer, range)| (buffer, range.clone())),
                    (&bias_buffer.0, bias_buffer.1.clone()),
                    (&output.0, output.1.clone()),
                    num_cols,
                    input.len() as u32,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
    }
    // SAFETY: every command buffer using these buffers has completed.
    unsafe {
        if let Some(input_buffer) = &input_buffer {
            let read = KernelFixture::read_guarded(input_buffer, sentinel);
            assert_eq!(bytemuck::cast_slice::<T, u8>(&read), bytemuck::cast_slice::<T, u8>(input), "input changed");
        }
        let read = KernelFixture::read_guarded(&bias_buffer, B::from(-7.0).unwrap());
        assert_eq!(bytemuck::cast_slice::<B, u8>(&read), bytemuck::cast_slice::<B, u8>(bias), "bias changed");
        KernelFixture::read_guarded(&output, sentinel)
    }
}

fn kernel<T: ArrayElement, B: ArrayElement>(
    fixture: &KernelFixture,
    in_place: bool,
) -> TensorAddBiasVulkanKernel {
    TensorAddBiasVulkanKernel::new(&fixture.context, T::data_type(), B::data_type(), in_place).expect("TensorAddBias")
}

fn name<T: ArrayElement, B: ArrayElement>() -> String {
    format!("{:?}+{:?}", T::data_type(), B::data_type())
}

/// Odd shapes, both in-place modes, and a 1000-dispatch plus a two-submission dependent chain, bit-exact vs CPU.
fn matches_cpu<T: ArrayElement + Float + Debug, B: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    for in_place in [false, true] {
        let kernel = kernel::<T, B>(&fixture, in_place);
        for (length, num_cols) in [(1, 1), (31, 7), (33, 5), (1025, 129), (1_000_003, 1000)] {
            let input = (0..length).map(|i| T::from((i as f32).sin() * 30.0).unwrap()).collect::<Vec<_>>();
            let bias = (0..num_cols).map(|i| B::from((i as f32).cos() * 30.0).unwrap()).collect::<Vec<_>>();
            let case = (&input[..], &bias[..], num_cols as u32, in_place);
            KernelFixture::assert_bits(
                &cpu_output(case, 1),
                &gpu_output(&fixture, &kernel, case, &[1]),
                &format!("{} length {length} in_place {in_place}", name::<T, B>()),
            );
        }
    }
    let kernel = kernel::<T, B>(&fixture, true);
    let input = (0..1003).map(|i| T::from(i as f32 * 0.25 - 100.0).unwrap()).collect::<Vec<_>>();
    let bias = (0..13).map(|i| B::from(i as f32 * 0.0625 - 0.375).unwrap()).collect::<Vec<_>>();
    for submissions in [&[1000][..], &[7, 5]] {
        let case = (&input[..], &bias[..], 13, true);
        KernelFixture::assert_bits(
            &cpu_output(case, submissions.iter().sum()),
            &gpu_output(&fixture, &kernel, case, submissions),
            &format!("{} submissions {submissions:?}", name::<T, B>()),
        );
    }
    fixture.assert_clean();
}

/// Ties to even in both directions (`1 + ulp/2`, `1 + ulp + ulp/2`) plus NaN, infinities, overflow and signed
/// zeros, all with a bias of a different storage type.
fn rounds_and_propagates_specials<T: ArrayElement + Float + Debug, B: ArrayElement + Float + Debug>() {
    let fixture = KernelFixture::new();
    let ulp = T::epsilon().to_f32().unwrap();
    let t = |value: f32| T::from(value).unwrap();
    let b = |value: f32| B::from(value).unwrap();
    let input = [t(1.0), t(1.0 + ulp), t(-1.0), t(-1.0 - ulp), T::nan(), T::infinity(), T::zero(), -T::zero()];
    let input = [&input[..], &[-T::zero(), T::max_value(), T::infinity(), T::one()]].concat();
    let bias = [b(ulp / 2.0), b(ulp / 2.0), b(-ulp / 2.0), b(-ulp / 2.0), B::one(), B::one(), B::zero(), -B::zero()];
    let bias = [&bias[..], &[B::zero(), B::max_value(), B::neg_infinity(), B::nan()]].concat();
    let case = (&input[..], &bias[..], bias.len() as u32, false);
    let cpu = cpu_output(case, 1);
    let expected_ties = [t(1.0), t(1.0 + 2.0 * ulp), t(-1.0), t(-1.0 - 2.0 * ulp)];
    KernelFixture::assert_bits(&expected_ties, &cpu[..4], "CPU ties");
    assert!(cpu[4].is_nan() && cpu[5].is_infinite() && cpu[10].is_nan() && cpu[7].is_sign_negative());
    KernelFixture::assert_bits(
        &cpu,
        &gpu_output(&fixture, &kernel::<T, B>(&fixture, false), case, &[1]),
        &name::<T, B>(),
    );
    fixture.assert_clean();
}

#[uzu_test]
fn matches_cpu_all_combinations() {
    for_each_combination!(matches_cpu);
}

#[uzu_test]
fn rounds_and_propagates_specials_all_combinations() {
    for_each_combination!(rounds_and_propagates_specials);
}

#[uzu_test]
fn zero_length_records_nothing() {
    let fixture = KernelFixture::new();
    for in_place in [false, true] {
        let output =
            gpu_output::<f16, f32>(&fixture, &kernel::<f16, f32>(&fixture, in_place), (&[], &[], 1, in_place), &[1]);
        assert!(output.is_empty());
    }
    fixture.assert_clean();
}

#[uzu_test]
fn rejects_invalid_contracts() {
    let fixture = KernelFixture::new();
    assert!(matches!(
        TensorAddBiasVulkanKernel::new(&fixture.context, DataType::F32, DataType::I32, false),
        Err(Error::KernelVariant {
            kernel: "TensorAddBias",
            ..
        })
    ));
    let values = fixture.buffer(&[1.0f32; 4]);
    let mut encoding = fixture.encoding();
    for in_place in [false, true] {
        let kernel = kernel::<f32, f32>(&fixture, in_place);
        let wrong = in_place.then(|| (&values, 0..16));
        let encode = AssertUnwindSafe(|| unsafe {
            // SAFETY: never dispatched: the optional-argument assertion fails before recording.
            kernel.encode(wrong, (&values, 0..16), (&values, 0..16), 4, 4, &mut encoding);
        });
        assert!(catch_unwind(encode).is_err(), "in_place {in_place} accepted a wrong optional input");
    }
    KernelFixture::complete(encoding);
    // SAFETY: the completed command buffer recorded no dispatch.
    assert_eq!(unsafe { KernelFixture::read::<f32>(&values) }, [1.0; 4]);
    fixture.assert_clean();
}

/// Run alone: `cargo test ... tensor_add_bias_test::throughput -- --ignored --nocapture`. Pipelines are created
/// before timing; each sample is one submission.
#[uzu_test]
#[ignore]
fn throughput() {
    const LENGTH: usize = 1 << 25;
    const NUM_COLS: usize = 4096;
    fn measure<T: ArrayElement + Float, B: ArrayElement + Float>(fixture: &KernelFixture) {
        let kernel = kernel::<T, B>(fixture, false);
        let input = fixture.buffer(&vec![T::one(); LENGTH]);
        let bias = fixture.buffer(&vec![B::one(); NUM_COLS]);
        let output = fixture.buffer(&vec![T::zero(); LENGTH]);
        let bytes = |len: usize, size: usize| 0..(len * size) as u64;
        let mut samples = (0..13)
            .map(|_| {
                let start = Instant::now();
                let mut encoding = fixture.encoding();
                // SAFETY: input/output hold LENGTH elements and bias NUM_COLS; output aliases nothing.
                unsafe {
                    kernel.encode(
                        Some((&input, bytes(LENGTH, size_of::<T>()))),
                        (&bias, bytes(NUM_COLS, size_of::<B>())),
                        (&output, bytes(LENGTH, size_of::<T>())),
                        NUM_COLS as u32,
                        LENGTH as u32,
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
        let (read, written) = (LENGTH * size_of::<T>() + NUM_COLS * size_of::<B>(), LENGTH * size_of::<T>());
        let rate = |time: Duration| (read + written) as f64 / time.as_secs_f64() / 1e9;
        eprintln!(
            "TensorAddBias {} x{LENGTH}: read {read} B, written {written} B; median of 10 after 3 warm-up: GPU {gpu:?} ({:.1} GB/s), wall {wall:?} ({:.1} GB/s)",
            name::<T, B>(),
            rate(gpu),
            rate(wall)
        );
    }
    let fixture = KernelFixture::new();
    measure::<f32, f32>(&fixture);
    measure::<f16, f32>(&fixture);
    measure::<bf16, bf16>(&fixture);
    fixture.assert_clean();
}
