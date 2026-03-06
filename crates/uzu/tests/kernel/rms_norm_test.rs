use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    time::Instant,
};

use half::{bf16, f16};
use num_traits::Float;
use rand::{RngExt, SeedableRng, prelude::StdRng};
#[cfg(feature = "metal")]
use uzu::backends::metal::Metal;
use uzu::{
    ArrayElement, DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial,
            CommandBufferPending, Context, Kernels, kernel::RMSNormKernel,
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

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
) -> (Vec<OutputT>, f64, Option<f64>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::RMSNormKernel::new(
        &context,
        InputT::data_type(),
        ScaleT::data_type(),
        OutputT::data_type(),
        AccumT::data_type(),
        input.in_place,
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

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        input_buffer,
        scales_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.element_count,
        input.epsilon,
        input.scale_offset,
        input.full_layer,
        &mut command_buffer,
    );

    let instant = Instant::now();
    let completed =
        command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    let host_elapsed_ms = instant.elapsed().as_secs_f64() * 1e3;
    let gpu_elapsed_ms = completed.gpu_execution_time_ms();

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
        0.012f32
    } else {
        1e-5
    };

    for_each_backend!(|B| {
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

fn test_performance<B: Backend>(
    batch_size: usize,
    model_dim: usize,
) {
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;
    let mut rng = StdRng::seed_from_u64(42);

    let input_size = batch_size * model_dim;
    let mut input_data = vec![0.0f32; batch_size * model_dim];
    let mut output_data = vec![0.0f32; batch_size * model_dim];
    for i in 0..input_size {
        input_data[i] = rng.random_range(-2.0f32..2.0f32);
        output_data[i] = input_data[i]
    }

    let mut scale_data = vec![0.0f32; model_dim];
    for x in scale_data.iter_mut() {
        *x = rng.random_range(0.1f32..3.0f32);
    }

    let input = Input::<f32, f32, f32> {
        input: input_data.into_boxed_slice(),
        scales: scale_data.into_boxed_slice(),
        output: output_data.into_boxed_slice(),
        batch_size: batch_size as u32,
        element_count: model_dim as u32,
        epsilon,
        scale_offset,
        full_layer: false,
        in_place: false,
    };
    let (output_data, host_elapsed_ms, gpu_elapsed_ms) = get_output::<B, f32, f32, f32, f32>(&input);
    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "RMS norm perf (backend={}, batch={}, model_dim={}): GPU={:.2} ms, Host-side={:.2} ms",
                std::any::type_name::<B>(),
                batch_size,
                model_dim,
                gpu_time,
                host_elapsed_ms
            );
        },
        None => {
            println!(
                "RMS norm perf (backend={}, batch={}, model_dim={}): Host-side={:.2} ms (GPU timing unavailable)",
                std::any::type_name::<B>(),
                batch_size,
                model_dim,
                host_elapsed_ms
            );
        },
    }

    // Sample check for large outputs
    let sample_size = std::cmp::min(1000, output_data.len());
    for i in (0..output_data.len()).step_by(std::cmp::max(1, output_data.len() / sample_size)) {
        assert!(output_data[i].is_finite(), "Output at index {} is not finite: {}", i, output_data[i]);
    }

    // Check that normalization is working
    let mean_abs = output_data.iter().take(sample_size).map(|x| x.abs()).sum::<f32>() / sample_size as f32;
    assert!(
        mean_abs > 0.01 && mean_abs < 100.0,
        "Mean absolute value {} seems unreasonable - normalization may not be working",
        mean_abs
    );
}

// AccumT f32
#[test]
fn test_f32_f32_f32_f32() {
    test_basic::<f32, f32, f32, f32>();
}

// AccumT f32 + f16
#[test]
fn test_f16_f32_f32_f32() {
    test_basic::<f16, f32, f32, f32>();
}

#[test]
fn test_f32_f16_f32_f32() {
    test_basic::<f32, f16, f32, f32>();
}

#[test]
fn test_f32_f32_f16_f32() {
    test_basic::<f32, f32, f16, f32>();
}

#[test]
fn test_f16_f16_f32_f32() {
    test_basic::<f16, f16, f32, f32>();
}

#[test]
fn test_f16_f32_f16_f32() {
    test_basic::<f16, f32, f16, f32>();
}

#[test]
fn test_f32_f16_f16_f32() {
    test_basic::<f32, f16, f16, f32>();
}

#[test]
fn test_f16_f16_f16_f32() {
    test_basic::<f16, f16, f16, f32>();
}

// AccumT f32 + bf16
#[test]
fn test_bf16_f32_f32_f32() {
    test_basic::<bf16, f32, f32, f32>();
}

#[test]
fn test_f32_bf16_f32_f32() {
    test_basic::<f32, bf16, f32, f32>();
}

#[test]
fn test_f32_f32_bf16_f32() {
    test_basic::<f32, f32, bf16, f32>();
}

#[test]
fn test_bf16_bf16_f32_f32() {
    test_basic::<bf16, bf16, f32, f32>();
}

#[test]
fn test_bf16_f32_bf16_f32() {
    test_basic::<bf16, f32, bf16, f32>();
}

#[test]
fn test_bf16_bf16_bf16_f32() {
    test_basic::<bf16, bf16, bf16, f32>();
}

// AccumT f16
#[test]
fn test_f32_f32_f32_f16() {
    test_basic::<f32, f32, f32, f16>();
}

#[test]
fn test_f16_f32_f32_f16() {
    test_basic::<f16, f32, f32, f16>();
}

#[test]
fn test_f32_f16_f32_f16() {
    test_basic::<f32, f16, f32, f16>();
}

#[test]
fn test_f32_f32_f16_f16() {
    test_basic::<f32, f32, f16, f16>();
}

#[test]
fn test_f16_f16_f32_f16() {
    test_basic::<f16, f16, f32, f16>();
}

#[test]
fn test_f16_f32_f16_f16() {
    test_basic::<f16, f32, f16, f16>();
}

#[test]
fn test_f32_f16_f16_f16() {
    test_basic::<f32, f16, f16, f16>();
}

#[test]
fn test_f16_f16_f16_f16() {
    test_basic::<f16, f16, f16, f16>();
}

// edge cases
#[test]
fn test_edge_f32_f32_f32_f32() {
    test_edge::<f32, f32, f32, f32>();
}

#[test]
fn test_edge_f32_f16_f16_f32() {
    test_edge::<f32, f16, f16, f32>();
}

#[test]
fn test_edge_f16_f16_f16_f16() {
    test_edge::<f16, f16, f16, f16>();
}

// performance tests
#[test]
fn test_perf_8k() {
    if cfg!(feature = "metal") {
        test_performance::<Metal>(8, 8192); // Large model like LLaMA-70B
    }
}

#[test]
fn test_perf_16k() {
    if cfg!(feature = "metal") {
        test_performance::<Metal>(16, 16384); // Huge model dimension
    }
}
