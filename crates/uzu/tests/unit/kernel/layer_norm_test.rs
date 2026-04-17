use std::{
    fmt::{Debug, Display},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::LayerNormKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

static BOOL_ALL: &[bool] = &[true, false];
static BOOL_FALSE: &[bool] = &[false];

struct Input<IN: ArrayElement + Float, SC: ArrayElement + Float, OUT: ArrayElement + Float> {
    input: Box<[IN]>,
    scales: Box<[SC]>,
    output: Box<[OUT]>,
    batch_size: u32,
    model_dim: u32,
    epsilon: f32,
    scale_offset: f32,
    full_layer: u32,
    in_place: bool,
}

fn get_test_data_basic<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float,
    ACC: ArrayElement + Float,
>(
    full_layer: u32,
    in_place: bool,
) -> (Input<IN, SC, OUT>, Vec<OUT>) {
    let batch_size = 2usize;
    let model_dim = 4096usize;
    let epsilon = 1e-5f32;
    let scale_offset = 0.123f32;

    let total = batch_size * model_dim;
    let mut input_data: Vec<IN> = vec![IN::zero(); total];
    let mut output_data: Vec<OUT> = vec![OUT::zero(); total];
    let scales_data: Vec<SC> = (0..model_dim).map(|i| SC::from(1.0 + 0.01 * (i as f32).sin()).unwrap()).collect();

    for i in 0..total {
        let val = 2.0 * f32::sin(i as f32 * 0.01) + 0.1 * (i as f32 / total as f32);
        input_data[i] = IN::from(val).unwrap();
        output_data[i] = OUT::from(val).unwrap();
    }

    let input = Input::<IN, SC, OUT> {
        input: input_data.into_boxed_slice(),
        scales: scales_data.into_boxed_slice(),
        output: output_data.into_boxed_slice(),
        batch_size: batch_size as u32,
        model_dim: model_dim as u32,
        epsilon,
        scale_offset,
        full_layer,
        in_place,
    };

    let output = get_output::<Cpu, IN, SC, OUT, ACC>(&input);
    (input, output)
}

fn get_test_data_edge<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float,
    ACC: ArrayElement + Float,
>(
    full_layer: u32,
    in_place: bool,
) -> (Input<IN, SC, OUT>, Vec<OUT>) {
    let batch_size = 1usize;
    let model_dim = 4usize;
    let epsilon = 1e-6f32;
    let scale_offset = 0.0f32;

    let input_data_f32: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
    let scales_data_f32: Vec<f32> = vec![2.0, 0.5, 1.5, 0.8];
    let input_data: Vec<IN> = input_data_f32.iter().map(|&x| IN::from(x).unwrap()).collect();
    let output_data: Vec<OUT> = input_data_f32.iter().map(|&x| OUT::from(x).unwrap()).collect();
    let scales_data: Vec<SC> = scales_data_f32.iter().map(|&x| SC::from(x).unwrap()).collect();

    let input = Input::<IN, SC, OUT> {
        input: input_data.into_boxed_slice(),
        scales: scales_data.into_boxed_slice(),
        output: output_data.into_boxed_slice(),
        batch_size: batch_size as u32,
        model_dim: model_dim as u32,
        epsilon,
        scale_offset,
        full_layer,
        in_place,
    };

    let output = get_output::<Cpu, IN, SC, OUT, ACC>(&input);
    (input, output)
}

fn get_output<
    B: Backend,
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float,
    ACC: ArrayElement + Float,
>(
    input: &Input<IN, SC, OUT>
) -> Vec<OUT> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::LayerNormKernel::new(
        &context,
        IN::data_type(),
        SC::data_type(),
        OUT::data_type(),
        ACC::data_type(),
        input.in_place,
    )
    .expect("Failed to create LayerNormKernel");

    let input_size = input.input.len();
    let input_buffer = (!input.in_place).then(|| context.create_array_from(&[input_size], &input.input, "").into_allocation());

    let scales_array = context.create_array_from(&[input.scales.len()], &input.scales, "");
    let mut output = match input.in_place {
        true => context.create_array_from(&[input_size], &input.output, "").into_allocation(),
        false => context
            .create_array_uninitialized(&[input_size], OUT::data_type(), "")
            .into_allocation(),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_buffer.as_ref(),
        scales_array.allocation(),
        &mut output,
        0,
        input.batch_size,
        input.model_dim,
        input.epsilon,
        input.scale_offset,
        input.full_layer,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    crate::common::helpers::allocation_to_vec(&output)
}

fn test_internal<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float + Debug + Display,
    ACC: ArrayElement + Float,
>(
    input: &Input<IN, SC, OUT>,
    expected: &[OUT],
) {
    let eps = if matches!(IN::data_type(), DataType::F16 | DataType::BF16)
        || matches!(SC::data_type(), DataType::F16 | DataType::BF16)
        || matches!(OUT::data_type(), DataType::F16 | DataType::BF16)
    {
        0.012f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, IN, SC, OUT, ACC>(input);
        let msg = format!(
            "LayerNorm kernel test failed with backend={}, full_layer={}, in_place={}",
            std::any::type_name::<B>(),
            input.full_layer,
            input.in_place,
        );
        assert_eq_float::<OUT>(expected, &output, eps, &msg);
    });
}

fn test_basic<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float + Debug + Display,
    ACC: ArrayElement + Float,
>() {
    let in_place_values: &[bool] = if IN::data_type() == OUT::data_type() {
        &BOOL_ALL
    } else {
        &BOOL_FALSE
    };

    for in_place in in_place_values {
        for full_layer in [1u32, 0u32] {
            let (input, expected) = get_test_data_basic::<IN, SC, OUT, ACC>(full_layer, *in_place);
            test_internal::<IN, SC, OUT, ACC>(&input, expected.as_slice())
        }
    }
}

fn test_edge<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float + Debug + Display,
    ACC: ArrayElement + Float,
>() {
    let in_place_values: &[bool] = if IN::data_type() == OUT::data_type() {
        &BOOL_ALL
    } else {
        &BOOL_FALSE
    };

    for in_place in in_place_values {
        let (input, expected) = get_test_data_edge::<IN, SC, OUT, ACC>(0, *in_place);
        test_internal::<IN, SC, OUT, ACC>(&input, expected.as_slice())
    }
}

// basic tests - AccumT f32
#[uzu_test]
fn test_f32_f32_f32_f32() {
    test_basic::<f32, f32, f32, f32>();
}

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

// basic tests - AccumT f32 + bf16
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

// edge tests
#[uzu_test]
fn test_edge_f32_f32_f32_f32() {
    test_edge::<f32, f32, f32, f32>();
}

#[uzu_test]
fn test_edge_f32_f16_f16_f32() {
    test_edge::<f32, f16, f16, f32>();
}

#[uzu_test]
fn test_edge_f16_f16_f16_f32() {
    test_edge::<f16, f16, f16, f32>();
}
