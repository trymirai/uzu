use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use backend_uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::TensorAddScaleKernel},
        cpu::Cpu,
    },
};

use crate::{
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    input: Box<[T]>,
    bias: Box<[T]>,
    num_cols: u32,
    length: u32,
    scale: f32,
}

fn get_input<T: ArrayElement + Float>(
    num_rows: usize,
    num_cols: usize,
) -> (Vec<T>, Vec<T>) {
    let length = num_rows * num_cols;
    let input: Vec<T> = (0..length).map(|i| T::from((i as f32 * 0.3) - 5.0).unwrap()).collect();
    let bias: Vec<T> = (0..num_cols).map(|i| T::from((i as f32 * 0.7) - 2.0).unwrap()).collect();
    (input, bias)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TensorAddScaleKernel::new(&context, T::data_type())
        .expect("Failed to create TensorAddScaleKernel");

    let length = input.length as usize;
    let num_cols = input.num_cols as usize;
    let input_allocation = alloc_allocation_with_data::<B, T>(&context, &input.input[..length]);
    let bias_allocation = alloc_allocation_with_data::<B, T>(&context, &input.bias[..num_cols]);
    let mut output_allocation = alloc_allocation::<B, T>(&context, length);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &input_allocation,
        &bias_allocation,
        &mut output_allocation,
        input.num_cols,
        input.length,
        input.scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output_allocation)
}

fn get_test_data_basic<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let num_rows = 4usize;
    let num_cols = 32usize;
    let (data, bias) = get_input::<T>(num_rows, num_cols);

    let input = Input {
        input: data.into_boxed_slice(),
        bias: bias.into_boxed_slice(),
        num_cols: num_cols as u32,
        length: (num_rows * num_cols) as u32,
        scale: 0.5,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_large<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let num_rows = 64usize;
    let num_cols = 128usize;
    let (data, bias) = get_input::<T>(num_rows, num_cols);

    let input = Input {
        input: data.into_boxed_slice(),
        bias: bias.into_boxed_slice(),
        num_cols: num_cols as u32,
        length: (num_rows * num_cols) as u32,
        scale: 2.0,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_edge<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    // Single row, small dim, scale = 1.0
    let num_rows = 1usize;
    let num_cols = 4usize;
    let data: Vec<T> = [1.0f32, -1.0, 0.0, 3.5].iter().map(|&v| T::from(v).unwrap()).collect();
    let bias: Vec<T> = [0.5f32, -0.5, 1.0, 0.0].iter().map(|&v| T::from(v).unwrap()).collect();

    let input = Input {
        input: data.into_boxed_slice(),
        bias: bias.into_boxed_slice(),
        num_cols: num_cols as u32,
        length: (num_rows * num_cols) as u32,
        scale: 1.0,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.05f32
    } else {
        1e-5
    };
    let (input, expected) = get_test_data_basic::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("TensorAddScale basic test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.05f32
    } else {
        1e-5
    };
    let (input, expected) = get_test_data_large::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("TensorAddScale large test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_edge<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.05f32
    } else {
        1e-5
    };
    let (input, expected) = get_test_data_edge::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("TensorAddScale edge test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

// basic tests
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

// large tests
#[uzu_test]
fn test_large_f32() {
    test_large::<f32>();
}

#[uzu_test]
fn test_large_f16() {
    test_large::<f16>();
}

#[uzu_test]
fn test_large_bf16() {
    test_large::<bf16>();
}

// edge tests
#[uzu_test]
fn test_edge_f32() {
    test_edge::<f32>();
}

#[uzu_test]
fn test_edge_f16() {
    test_edge::<f16>();
}

#[uzu_test]
fn test_edge_bf16() {
    test_edge::<bf16>();
}
