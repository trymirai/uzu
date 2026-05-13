use std::{
    fmt::{Debug, Display},
    time::{Duration, Instant},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Allocation, Context, Encoder, Kernels, kernel::RMSNormKernel, Backend},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::assert::assert_eq_float,
    uzu_test,
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
        false,
    )
    .expect("Failed to create RMSNormKernel");

    let input_size = input.input.len();
    let input_buffer = (!input.in_place).then(|| context.create_array_from(&[input_size], &input.input).into_allocation());

    let scales_array = context.create_array_from(&[input.scales.len()], &input.scales);
    let mut output = match input.in_place {
        true => context.create_array_from(&[input_size], &input.output).into_allocation(),
        false => context
            .create_array_uninitialized(&[input_size], OutputT::data_type())
            .into_allocation(),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_buffer.as_ref(),
        scales_array.allocation(),
        &mut output,
        None::<&mut Allocation<B>>,
        None::<&Allocation<B>>,
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

    (
        crate::common::helpers::allocation_to_vec(&output),
        host_elapsed_ms,
        gpu_elapsed_ms,
    )
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

