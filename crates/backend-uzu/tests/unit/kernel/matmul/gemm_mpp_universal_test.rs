#![cfg(metal_backend)]

use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{DeviceExt, MatmulDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::{assert::assert_eq_float, type_short_name},
    uzu_bench, uzu_test,
};

#[derive(Clone)]
struct Input<T: ArrayElement + Float> {
    left_storage: Box<[T]>,
    right_storage: Box<[T]>,
    destination_prefill: Option<Box<[T]>>,
    bias: Option<Box<[T]>>,
    batch_dim: usize,
    input_dim: usize,
    output_dim: usize,
    ab_scale: f32,
    a_offset: usize,
}

fn create_input<T: ArrayElement + Float>(
    batch_dim: usize,
    input_dim: usize,
    output_dim: usize,
    ab_scale: f32,
    accumulate: bool,
    use_bias: bool,
    a_offset_elements: usize,
) -> Input<T> {
    let logical_left: Vec<T> = (0..batch_dim * input_dim)
        .map(|index| T::from(((index % 13) as f32) * 0.1 - 0.6).unwrap())
        .collect();
    let left_storage = (0..a_offset_elements)
        .map(|index| T::from(((index % 7) as f32) * 0.05 - 0.15).unwrap())
        .chain(logical_left)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let right_storage: Vec<T> = (0..output_dim * input_dim)
        .map(|index| T::from(((index % 17) as f32) * 0.1 - 0.8).unwrap())
        .collect();
    let destination_prefill = accumulate.then(|| {
        (0..batch_dim * output_dim)
            .map(|index| T::from(((index % 11) as f32) * 0.02 - 0.1).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let bias = use_bias.then(|| {
        (0..output_dim)
            .map(|index| T::from(((index % 9) as f32) * 0.03 - 0.12).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });

    Input {
        left_storage,
        right_storage: right_storage.into_boxed_slice(),
        destination_prefill,
        bias,
        batch_dim,
        input_dim,
        output_dim,
        ab_scale,
        a_offset: a_offset_elements * std::mem::size_of::<T>(),
    }
}

fn encode_argument_c<'a, B: Backend>(
    bias: Option<&'a B::Buffer>,
    accumulate: bool,
) -> MatmulArgumentC<'a, B> {
    if let Some(bias_buffer) = bias {
        MatmulArgumentC::Bias(bias_buffer)
    } else if accumulate {
        MatmulArgumentC::Accumulate
    } else {
        MatmulArgumentC::None
    }
}

fn get_cpu_output<T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let left_array = context.create_array_from(&[input.left_storage.len()], &input.left_storage, "");
    let right_array = context.create_array_from(&[input.right_storage.len()], &input.right_storage, "");
    let bias_array = input
        .bias
        .as_ref()
        .map(|bias| context.create_array_from(&[input.output_dim], bias, ""));
    let destination_array = if let Some(prefill) = input.destination_prefill.as_ref() {
        context.create_array_from(&[input.batch_dim, input.output_dim], prefill, "")
    } else {
        context.create_array_uninitialized(&[input.batch_dim, input.output_dim], T::data_type(), "")
    };

    let left_buffer = left_array.buffer();
    let left_ref = left_buffer.borrow();
    let right_buffer = right_array.buffer();
    let right_ref = right_buffer.borrow();
    let bias_buffer = bias_array.as_ref().map(|array| array.buffer());
    let bias_ref = bias_buffer.as_ref().map(|buffer| buffer.borrow());
    let destination_buffer = destination_array.buffer();
    let mut destination_ref = destination_buffer.borrow_mut();

    let mut kernel = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("CPU MatmulKernel");
    let mut encoder = Encoder::new(context.as_ref()).expect("CPU encoder");
    kernel.encode(
        &context,
        MatmulArguments {
            a: left_ref.deref(),
            a_offset: input.a_offset as u64,
            b: right_ref.deref(),
            ab_scale: input.ab_scale,
            c: encode_argument_c::<Cpu>(bias_ref.as_ref().map(|buffer| buffer.deref()), input.destination_prefill.is_some()),
            d: destination_ref.deref_mut(),
            batch_dim: input.batch_dim as u32,
            input_dim: input.input_dim as u32,
            output_dim: input.output_dim as u32,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    drop(destination_ref);
    destination_array.as_slice().to_vec()
}

fn get_metal_output<T: ArrayElement + Float>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    input: &Input<T>,
    path: MatmulDispatchPath,
) -> Vec<T> {
    let left_array = context.create_array_from(&[input.left_storage.len()], &input.left_storage, "");
    let right_array = context.create_array_from(&[input.right_storage.len()], &input.right_storage, "");
    let bias_array = input
        .bias
        .as_ref()
        .map(|bias| context.create_array_from(&[input.output_dim], bias, ""));
    let destination_array = if let Some(prefill) = input.destination_prefill.as_ref() {
        context.create_array_from(&[input.batch_dim, input.output_dim], prefill, "")
    } else {
        context.create_array_uninitialized(&[input.batch_dim, input.output_dim], T::data_type(), "")
    };

    let left_buffer = left_array.buffer();
    let left_ref = left_buffer.borrow();
    let right_buffer = right_array.buffer();
    let right_ref = right_buffer.borrow();
    let bias_buffer = bias_array.as_ref().map(|array| array.buffer());
    let bias_ref = bias_buffer.as_ref().map(|buffer| buffer.borrow());
    let destination_buffer = destination_array.buffer();
    let mut destination_ref = destination_buffer.borrow_mut();

    let mut encoder = Encoder::new(context).expect("Metal encoder");
    kernel.encode_with_path(
        context,
        MatmulArguments {
            a: left_ref.deref(),
            a_offset: input.a_offset as u64,
            b: right_ref.deref(),
            ab_scale: input.ab_scale,
            c: encode_argument_c::<Metal>(bias_ref.as_ref().map(|buffer| buffer.deref()), input.destination_prefill.is_some()),
            d: destination_ref.deref_mut(),
            batch_dim: input.batch_dim as u32,
            input_dim: input.input_dim as u32,
            output_dim: input.output_dim as u32,
        },
        &mut encoder,
        path,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    drop(destination_ref);
    destination_array.as_slice().to_vec()
}

fn create_metal_context_and_kernel<T: ArrayElement>() -> (Rc<MetalContext>, <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel) {
    let context = MetalContext::new().expect("Metal context");
    let kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("MatmulKernel");
    (context, kernel)
}

fn assert_universal_matches_cpu<T: ArrayElement + Float + Debug + Display>(
    input: Input<T>,
    tolerance: f32,
    message: &str,
) {
    let expected = get_cpu_output(&input);
    let (context, mut kernel) = create_metal_context_and_kernel::<T>();
    let actual = get_metal_output(&context, &mut kernel, &input, MatmulDispatchPath::GemmMppUniversal);
    assert_eq_float(&expected, &actual, tolerance, message);
}

fn assert_universal_matches_path<T: ArrayElement + Float + Debug + Display>(
    input: Input<T>,
    path: MatmulDispatchPath,
    tolerance: f32,
    message: &str,
) {
    let (context, mut kernel) = create_metal_context_and_kernel::<T>();
    let expected = get_metal_output(&context, &mut kernel, &input, path);
    let actual = get_metal_output(&context, &mut kernel, &input, MatmulDispatchPath::GemmMppUniversal);
    assert_eq_float(&expected, &actual, tolerance, message);
}

#[uzu_test]
fn test_f16_aligned() {
    assert_universal_matches_cpu(create_input::<f16>(64, 64, 64, 1.0, false, false, 0), 0.05, "F16 aligned universal");
}

#[uzu_test]
fn test_bf16_aligned() {
    assert_universal_matches_cpu(create_input::<bf16>(64, 64, 64, 1.0, false, false, 0), 0.2, "BF16 aligned universal");
}

#[uzu_test]
fn test_f32_aligned() {
    assert_universal_matches_cpu(create_input::<f32>(64, 64, 64, 1.0, false, false, 0), 0.01, "F32 aligned universal");
}

#[uzu_test]
fn test_f16_ragged() {
    assert_universal_matches_cpu(create_input::<f16>(33, 35, 29, 1.0, false, false, 0), 0.05, "F16 ragged universal");
}

#[uzu_test]
fn test_bf16_accumulate() {
    assert_universal_matches_cpu(create_input::<bf16>(33, 65, 31, 1.0, true, false, 0), 0.3, "BF16 accumulate universal");
}

#[uzu_test]
fn test_f32_bias() {
    assert_universal_matches_cpu(create_input::<f32>(16, 96, 33, 1.0, false, true, 0), 0.01, "F32 bias universal");
}

#[uzu_test]
fn test_f16_ab_scale() {
    assert_universal_matches_cpu(create_input::<f16>(32, 96, 40, 0.5, false, false, 0), 0.05, "F16 scaled universal");
}

#[uzu_test]
fn test_f16_a_offset() {
    assert_universal_matches_cpu(create_input::<f16>(16, 64, 48, 1.0, false, false, 11), 0.05, "F16 a_offset universal");
}

#[uzu_test]
fn test_bf16_matches_classic_gemm() {
    assert_universal_matches_path(
        create_input::<bf16>(128, 2048, 2048, 1.0, false, false, 0),
        MatmulDispatchPath::Gemm,
        1.0,
        "Universal MPP must match classic GEMM on BF16",
    );
}

#[uzu_test]
fn test_f32_matches_classic_gemm() {
    assert_universal_matches_path(
        create_input::<f32>(64, 512, 256, 1.0, false, false, 0),
        MatmulDispatchPath::Gemm,
        0.01,
        "Universal MPP must match classic GEMM on F32",
    );
}

#[uzu_test]
fn test_bf16_matches_current_mpp_when_available() {
    let (context, _) = create_metal_context_and_kernel::<bf16>();
    if !context.device.supports_mxu() {
        eprintln!("Skipping current-MPP comparison: device does not support MXU");
        return;
    }
    drop(context);

    assert_universal_matches_path(
        create_input::<bf16>(128, 2048, 2048, 0.5, true, false, 0),
        MatmulDispatchPath::GemmMpp,
        1.0,
        "Universal MPP must match current MPP on BF16",
    );
}

const BENCHMARK_SHAPES: &[(usize, usize, usize)] =
    &[(128, 2048, 8192), (128, 4096, 14336), (256, 4096, 4096), (512, 8192, 2048)];

#[uzu_bench]
fn bench_gemm_mpp_universal(criterion: &mut Criterion) {
    let metal_context = MetalContext::new().unwrap();
    let mut matmul_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&metal_context, bf16::data_type())
            .expect("MatmulKernel");

    let mut benchmark_group =
        criterion.benchmark_group(format!("{}/Kernel/Matmul/GEMM_MPP_UNIVERSAL", type_short_name::<Metal>()));

    for &(batch_dim, input_dim, output_dim) in BENCHMARK_SHAPES {
        let left_array = metal_context.create_array_uninitialized(&[batch_dim, input_dim], bf16::data_type(), "");
        let right_array = metal_context.create_array_uninitialized(&[output_dim, input_dim], bf16::data_type(), "");
        let destination_array =
            metal_context.create_array_uninitialized(&[batch_dim, output_dim], bf16::data_type(), "");

        let floating_point_operations = 2 * batch_dim * input_dim * output_dim;
        benchmark_group.throughput(Throughput::Elements(floating_point_operations as u64));

        benchmark_group.bench_function(
            BenchmarkId::new("BF16", format!("M[{batch_dim}]K[{input_dim}]N[{output_dim}]")),
            |bencher| {
                let left_buffer = left_array.buffer();
                let left_ref = left_buffer.borrow();
                let right_buffer = right_array.buffer();
                let right_ref = right_buffer.borrow();
                let destination_buffer = destination_array.buffer();

                bencher.iter_custom(|iteration_count| {
                    let mut encoder = Encoder::<Metal>::new(&metal_context).unwrap();

                    for _ in 0..iteration_count {
                        let mut destination_ref = destination_buffer.borrow_mut();
                        matmul_kernel.encode_with_path(
                            &metal_context,
                            MatmulArguments {
                                a: left_ref.deref(),
                                a_offset: 0,
                                b: right_ref.deref(),
                                ab_scale: 1.0,
                                c: MatmulArgumentC::None,
                                d: destination_ref.deref_mut(),
                                batch_dim: batch_dim as u32,
                                input_dim: input_dim as u32,
                                output_dim: output_dim as u32,
                            },
                            &mut encoder,
                            MatmulDispatchPath::GemmMppUniversal,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            },
        );
    }
}

#[uzu_bench]
fn bench_gemm_auto(criterion: &mut Criterion) {
    let metal_context = MetalContext::new().unwrap();
    let mut matmul_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&metal_context, bf16::data_type())
            .expect("MatmulKernel");

    let mut benchmark_group =
        criterion.benchmark_group(format!("{}/Kernel/Matmul/GEMM_AUTO", type_short_name::<Metal>()));

    for &(batch_dim, input_dim, output_dim) in BENCHMARK_SHAPES {
        let left_array = metal_context.create_array_uninitialized(&[batch_dim, input_dim], bf16::data_type(), "");
        let right_array = metal_context.create_array_uninitialized(&[output_dim, input_dim], bf16::data_type(), "");
        let destination_array =
            metal_context.create_array_uninitialized(&[batch_dim, output_dim], bf16::data_type(), "");

        let floating_point_operations = 2 * batch_dim * input_dim * output_dim;
        benchmark_group.throughput(Throughput::Elements(floating_point_operations as u64));

        benchmark_group.bench_function(
            BenchmarkId::new("BF16", format!("M[{batch_dim}]K[{input_dim}]N[{output_dim}]")),
            |bencher| {
                let left_buffer = left_array.buffer();
                let left_ref = left_buffer.borrow();
                let right_buffer = right_array.buffer();
                let right_ref = right_buffer.borrow();
                let destination_buffer = destination_array.buffer();

                bencher.iter_custom(|iteration_count| {
                    let mut encoder = Encoder::<Metal>::new(&metal_context).unwrap();

                    for _ in 0..iteration_count {
                        let mut destination_ref = destination_buffer.borrow_mut();
                        matmul_kernel.encode_with_path(
                            &metal_context,
                            MatmulArguments {
                                a: left_ref.deref(),
                                a_offset: 0,
                                b: right_ref.deref(),
                                ab_scale: 1.0,
                                c: MatmulArgumentC::None,
                                d: destination_ref.deref_mut(),
                                batch_dim: batch_dim as u32,
                                input_dim: input_dim as u32,
                                output_dim: output_dim as u32,
                            },
                            &mut encoder,
                            MatmulDispatchPath::Auto,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            },
        );
    }
}

#[derive(Debug)]
struct PerfComparisonResult {
    data_type: &'static str,
    shape: (usize, usize, usize),
    gemm_ms: Option<f64>,
    universal_ms: Option<f64>,
    auto_ms: Option<f64>,
    current_mpp_ms: Option<f64>,
}

fn average_gpu_time_ms<T: ArrayElement + Float>(
    context: &MetalContext,
    kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    path: MatmulDispatchPath,
    shape: (usize, usize, usize),
) -> f64 {
    let (batch_dim, input_dim, output_dim) = shape;
    let left_array = context.create_array_uninitialized(&[batch_dim, input_dim], T::data_type(), "");
    let right_array = context.create_array_uninitialized(&[output_dim, input_dim], T::data_type(), "");
    let destination_array = context.create_array_uninitialized(&[batch_dim, output_dim], T::data_type(), "");

    let left_buffer = left_array.buffer();
    let left_ref = left_buffer.borrow();
    let right_buffer = right_array.buffer();
    let right_ref = right_buffer.borrow();
    let destination_buffer = destination_array.buffer();

    for _ in 0..3 {
        let mut encoder = Encoder::<Metal>::new(context).unwrap();
        let mut destination_ref = destination_buffer.borrow_mut();
        kernel.encode_with_path(
            context,
            MatmulArguments {
                a: left_ref.deref(),
                a_offset: 0,
                b: right_ref.deref(),
                ab_scale: 1.0,
                c: MatmulArgumentC::None,
                d: destination_ref.deref_mut(),
                batch_dim: batch_dim as u32,
                input_dim: input_dim as u32,
                output_dim: output_dim as u32,
            },
            &mut encoder,
            path,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    }

    let total_gpu_time = (0..10)
        .map(|_| {
            let mut encoder = Encoder::<Metal>::new(context).unwrap();
            let mut destination_ref = destination_buffer.borrow_mut();
            kernel.encode_with_path(
                context,
                MatmulArguments {
                    a: left_ref.deref(),
                    a_offset: 0,
                    b: right_ref.deref(),
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: destination_ref.deref_mut(),
                    batch_dim: batch_dim as u32,
                    input_dim: input_dim as u32,
                    output_dim: output_dim as u32,
                },
                &mut encoder,
                path,
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time().as_secs_f64() * 1000.0
        })
        .sum::<f64>();

    total_gpu_time / 10.0
}

#[uzu_test]
#[ignore]
fn compare_path_performance() {
    let context = MetalContext::new().unwrap();
    let mut bf16_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, bf16::data_type()).unwrap();
    let mut f32_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, f32::data_type()).unwrap();

    let mut results = Vec::new();
    for &shape in BENCHMARK_SHAPES {
        results.push(PerfComparisonResult {
            data_type: "BF16",
            shape,
            gemm_ms: Some(average_gpu_time_ms::<bf16>(&context, &mut bf16_kernel, MatmulDispatchPath::Gemm, shape)),
            universal_ms: Some(average_gpu_time_ms::<bf16>(
                &context,
                &mut bf16_kernel,
                MatmulDispatchPath::GemmMppUniversal,
                shape,
            )),
            auto_ms: Some(average_gpu_time_ms::<bf16>(&context, &mut bf16_kernel, MatmulDispatchPath::Auto, shape)),
            current_mpp_ms: context.device.supports_mxu().then(|| {
                average_gpu_time_ms::<bf16>(&context, &mut bf16_kernel, MatmulDispatchPath::GemmMpp, shape)
            }),
        });
        results.push(PerfComparisonResult {
            data_type: "F32",
            shape,
            gemm_ms: Some(average_gpu_time_ms::<f32>(&context, &mut f32_kernel, MatmulDispatchPath::Gemm, shape)),
            universal_ms: Some(average_gpu_time_ms::<f32>(
                &context,
                &mut f32_kernel,
                MatmulDispatchPath::GemmMppUniversal,
                shape,
            )),
            auto_ms: Some(average_gpu_time_ms::<f32>(&context, &mut f32_kernel, MatmulDispatchPath::Auto, shape)),
            current_mpp_ms: None,
        });
    }

    for result in results {
        eprintln!(
            "[perf] {} {:?}: gemm={:?}ms universal={:?}ms auto={:?}ms current_mpp={:?}ms",
            result.data_type,
            result.shape,
            result.gemm_ms,
            result.universal_ms,
            result.auto_ms,
            result.current_mpp_ms,
        );
    }
}
