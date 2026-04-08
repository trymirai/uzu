use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use itertools::iproduct;
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Buffer, Context, Encoder, Kernels, kernel::RMSNormKernel},
        cpu::Cpu,
    },
};

use crate::{
    common::{assert::assert_eq_float, type_short_name},
    uzu_bench, uzu_test,
};

static BOOL_ALL: &[bool] = &[true, false];
static BOOL_FALSE: &[bool] = &[false];

struct Input<InputT: ArrayElement + Float, ScaleT: ArrayElement + Float, OutputT: ArrayElement + Float> {
    input: Box<[InputT]>,
    scales: Box<[ScaleT]>,
    output: Box<[OutputT]>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    scale_offset: f32,
    full_layer: bool,
    in_place: bool,
}

fn get_test_data_basic<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    full_layer: bool,
    in_place: bool,
) -> (Input<InputT, ScaleT, OutputT>, Vec<OutputT>) {
    let batch_size = 1usize;
    let model_dim = 4096usize;
    let epsilon = 1e-5f32;
    let scale_offset = 0.123f32;

    let mut input_data: Vec<InputT> = vec![InputT::zero(); model_dim];
    let mut output_data: Vec<OutputT> = vec![OutputT::zero(); model_dim];
    let scales_data: Vec<ScaleT> = vec![ScaleT::one(); model_dim];
    for i in 0..model_dim {
        let input_value = 2.0 * f32::sin(i as f32 * 0.01) + 0.1 * (i as f32 / model_dim as f32);
        input_data[i] = InputT::from(input_value).unwrap();
        output_data[i] = OutputT::from(input_value).unwrap();
    }

    let input = Input::<InputT, ScaleT, OutputT> {
        input: input_data.into_boxed_slice(),
        scales: scales_data.into_boxed_slice(),
        output: output_data.into_boxed_slice(),
        batch_size: batch_size as u32,
        element_count: model_dim as u32,
        epsilon,
        scale_offset,
        full_layer,
        in_place,
    };

    let (output, _, _) = get_output::<Cpu, InputT, ScaleT, OutputT, AccumT>(&input);
    (input, output)
}

fn get_test_data_edge<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    full_layer: bool,
    in_place: bool,
) -> (Input<InputT, ScaleT, OutputT>, Vec<OutputT>) {
    let batch_size = 1usize;
    let model_dim = 4usize;
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;

    // Test with very small values (near epsilon)
    let input_data_f32: Vec<f32> = vec![1e-4, 2e-4, 3e-4, 4e-4]; // Scaled up for F16 precision
    let scales_data_f32: Vec<f32> = vec![2.0, 0.5, 1.5, 0.8]; // Non-unit scales
    let input_data: Vec<InputT> = input_data_f32.iter().map(|&x| InputT::from(x).unwrap()).collect();
    let output_data: Vec<OutputT> = input_data_f32.iter().map(|&x| OutputT::from(x).unwrap()).collect();
    let scales_data: Vec<ScaleT> = scales_data_f32.iter().map(|&x| ScaleT::from(x).unwrap()).collect();

    let input = Input::<InputT, ScaleT, OutputT> {
        input: input_data.into_boxed_slice(),
        scales: scales_data.into_boxed_slice(),
        output: output_data.into_boxed_slice(),
        batch_size: batch_size as u32,
        element_count: model_dim as u32,
        epsilon,
        scale_offset,
        full_layer,
        in_place,
    };

    let (output, _, _) = get_output::<Cpu, InputT, ScaleT, OutputT, AccumT>(&input);
    (input, output)
}

fn get_output<
    B: Backend,
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    input: &Input<InputT, ScaleT, OutputT>
) -> (Vec<OutputT>, f64, Duration) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::RMSNormKernel::new(
        &context,
        InputT::data_type(),
        ScaleT::data_type(),
        OutputT::data_type(),
        AccumT::data_type(),
        input.in_place,
        input.full_layer,
        false,
        false,
    )
    .expect("Failed to create RMSNormKernel");

    let input_size = input.input.len();
    let input_array = context.create_array_from(&[input_size], &input.input, "");
    let input_array_buffer_rc = input_array.buffer();
    let input_array_borrow = input_array_buffer_rc.borrow();
    let input_array_deref = input_array_borrow.deref();
    let input_buffer = (!input.in_place).then(|| input_array_deref);

    let scales_array = context.create_array_from(&[input.scales.len()], &input.scales, "");
    let output_array = match input.in_place {
        true => context.create_array_from(&[input_size], &input.output, ""),
        false => context.create_array_uninitialized(&[input_size], OutputT::data_type(), ""),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_buffer,
        scales_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        None::<&mut B::Buffer>,
        input.batch_size,
        input.element_count,
        input.epsilon,
        input.scale_offset,
        &mut encoder,
    );

    let instant = Instant::now();
    let completed = encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    let host_elapsed_ms = instant.elapsed().as_secs_f64() * 1e3;
    let gpu_elapsed_ms = completed.gpu_execution_time();

    (output_array.as_slice().to_vec(), host_elapsed_ms, gpu_elapsed_ms)
}

fn test_internal<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>(
    input: &Input<InputT, ScaleT, OutputT>,
    expected: &[OutputT],
) {
    let eps = if matches!(InputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(ScaleT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(OutputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(AccumT::data_type(), DataType::F16 | DataType::BF16)
    {
        0.016f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let (output, _, _) = get_output::<B, InputT, ScaleT, OutputT, AccumT>(input);
        let msg = format!(
            "RMSNorm kernel test failed with backend={}, full_layer={}, in_place={}",
            std::any::type_name::<B>(),
            input.full_layer,
            input.in_place,
        );
        assert_eq_float::<OutputT>(&expected, &output, eps, &msg);
    });
}

fn test_basic<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    let in_place_values: &[bool] = if InputT::data_type() == OutputT::data_type() {
        &BOOL_ALL
    } else {
        &BOOL_FALSE
    };

    for in_place in in_place_values {
        for full_layer in [true, false] {
            let (input, expected) = get_test_data_basic::<InputT, ScaleT, OutputT, AccumT>(full_layer, *in_place);
            test_internal::<InputT, ScaleT, OutputT, AccumT>(&input, expected.as_slice())
        }
    }
}

fn test_edge<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
    AccumT: ArrayElement + Float,
>() {
    let in_place_values: &[bool] = if InputT::data_type() == OutputT::data_type() {
        &BOOL_ALL
    } else {
        &BOOL_FALSE
    };

    for in_place in in_place_values {
        let (input, expected) = get_test_data_edge::<InputT, ScaleT, OutputT, AccumT>(false, *in_place);
        test_internal::<InputT, ScaleT, OutputT, AccumT>(&input, expected.as_slice())
    }
}

fn get_rms_norm_data(
    seed: u64,
    batch_size: usize,
    model_dim: usize,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let input_size = batch_size * model_dim;
    let mut input_data = vec![0.0f32; input_size];
    for x in input_data.iter_mut() {
        *x = rng.random_range(-2.0f32..2.0f32);
    }
    let mut scale_data = vec![0.0f32; model_dim];
    for x in scale_data.iter_mut() {
        *x = rng.random_range(0.1f32..3.0f32);
    }
    (input_data.into_boxed_slice(), scale_data.into_boxed_slice())
}

// AccumT f32
#[uzu_test]
fn test_f32_f32_f32_f32() {
    test_basic::<f32, f32, f32, f32>();
}

// AccumT f32 + f16
#[uzu_test]
fn test_f16_f32_f32_f32() {
    test_basic::<f16, f32, f32, f32>();
}

#[uzu_test]
fn test_f32_f16_f32_f32() {
    test_basic::<f32, f16, f32, f32>();
}

#[uzu_test]
fn test_f32_f32_f16_f32() {
    test_basic::<f32, f32, f16, f32>();
}

#[uzu_test]
fn test_f16_f16_f32_f32() {
    test_basic::<f16, f16, f32, f32>();
}

#[uzu_test]
fn test_f16_f32_f16_f32() {
    test_basic::<f16, f32, f16, f32>();
}

#[uzu_test]
fn test_f32_f16_f16_f32() {
    test_basic::<f32, f16, f16, f32>();
}

#[uzu_test]
fn test_f16_f16_f16_f32() {
    test_basic::<f16, f16, f16, f32>();
}

// AccumT f32 + bf16
#[uzu_test]
fn test_bf16_f32_f32_f32() {
    test_basic::<bf16, f32, f32, f32>();
}

#[uzu_test]
fn test_f32_bf16_f32_f32() {
    test_basic::<f32, bf16, f32, f32>();
}

#[uzu_test]
fn test_f32_f32_bf16_f32() {
    test_basic::<f32, f32, bf16, f32>();
}

#[uzu_test]
fn test_bf16_bf16_f32_f32() {
    test_basic::<bf16, bf16, f32, f32>();
}

#[uzu_test]
fn test_bf16_f32_bf16_f32() {
    test_basic::<bf16, f32, bf16, f32>();
}

#[uzu_test]
fn test_bf16_bf16_bf16_f32() {
    test_basic::<bf16, bf16, bf16, f32>();
}

// AccumT f16
#[uzu_test]
fn test_f32_f32_f32_f16() {
    test_basic::<f32, f32, f32, f16>();
}

#[uzu_test]
fn test_f16_f32_f32_f16() {
    test_basic::<f16, f32, f32, f16>();
}

#[uzu_test]
fn test_f32_f16_f32_f16() {
    test_basic::<f32, f16, f32, f16>();
}

#[uzu_test]
fn test_f32_f32_f16_f16() {
    test_basic::<f32, f32, f16, f16>();
}

#[uzu_test]
fn test_f16_f16_f32_f16() {
    test_basic::<f16, f16, f32, f16>();
}

#[uzu_test]
fn test_f16_f32_f16_f16() {
    test_basic::<f16, f32, f16, f16>();
}

#[uzu_test]
fn test_f32_f16_f16_f16() {
    test_basic::<f32, f16, f16, f16>();
}

#[uzu_test]
fn test_f16_f16_f16_f16() {
    test_basic::<f16, f16, f16, f16>();
}

// edge cases
#[uzu_test]
fn test_edge_f32_f32_f32_f32() {
    test_edge::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_edge_f32_f16_f16_f32() {
    test_edge::<f32, f16, f16, f32>();
}

#[uzu_test]
fn test_edge_f16_f16_f16_f16() {
    test_edge::<f16, f16, f16, f16>();
}

// benchmarks
#[uzu_bench]
fn bench_rms_norm(c: &mut Criterion) {
    type T = f32;
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;

    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().unwrap();

        let kernel = <<B as Backend>::Kernels as Kernels>::RMSNormKernel::new(
            &context,
            T::data_type(),
            T::data_type(),
            T::data_type(),
            T::data_type(),
            false,
            false,
            false,
            false,
        )
        .unwrap();

        let mut group = c.benchmark_group(format!("{}/Kernel/RMSNorm", type_short_name::<B>()));

        for (batch_size, model_dim) in iproduct!(
            [1, 4, 32, 128, 1024], // Batch sizes
            [
                1024, // LFM2-350M
                1152, // Gemma-3-1b
                2048, // Llama-3.2-1B
                2560, // Qwen3-4B
                3072, // Llama-3.2-3B
                4096, // Qwen3-8B
                5120, // Qwen2.5-32B
            ]
        ) {
            let (input_data, scale_data) = get_rms_norm_data(1337, batch_size, model_dim);
            let input_size = batch_size * model_dim;

            let input_buffer = context.create_buffer(input_size * std::mem::size_of::<T>()).unwrap();
            unsafe {
                std::slice::from_raw_parts_mut::<T>(input_buffer.cpu_ptr().as_ptr() as *mut T, input_size)
                    .copy_from_slice(input_data.as_ref());
            }

            let scales_buffer = context.create_buffer(model_dim * std::mem::size_of::<T>()).unwrap();
            unsafe {
                std::slice::from_raw_parts_mut::<T>(scales_buffer.cpu_ptr().as_ptr() as *mut T, model_dim)
                    .copy_from_slice(scale_data.as_ref());
            }

            let mut output_buffer = context.create_buffer(input_size * std::mem::size_of::<T>()).unwrap();

            group.throughput(Throughput::Elements((batch_size * model_dim) as u64));

            group.bench_function(BenchmarkId::from_parameter(format!("Batch[{batch_size}]Dim[{model_dim}]")), |b| {
                b.iter_custom(|n_iters| {
                    let mut encoder = Encoder::<B>::new(&context).unwrap();

                    for _ in 0..n_iters {
                        kernel.encode(
                            Some(&input_buffer),
                            &scales_buffer,
                            &mut output_buffer,
                            None::<&mut <B as Backend>::Buffer>,
                            batch_size as u32,
                            model_dim as u32,
                            epsilon,
                            scale_offset,
                            &mut encoder,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            });
        }
    });
}
