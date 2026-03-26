use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SigmoidKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    data: Box<[T]>,
}

fn get_input<T: ArrayElement + Float>(n: usize) -> Vec<T> {
    (0..n).map(|i| T::from((i as f32 * 0.3) - 5.0).unwrap()).collect()
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::SigmoidKernel::new(&context, T::data_type())
        .expect("Failed to create SigmoidKernel");

    let n = input.data.len();
    let input_array = context.create_array_from(&[n], &input.data, "");
    let output_array = context.create_array_uninitialized(&[n], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        n as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn get_test_data_basic<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let data = get_input::<T>(128);
    let input = Input {
        data: data.into_boxed_slice(),
    };
    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_large<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let data = get_input::<T>(4096);
    let input = Input {
        data: data.into_boxed_slice(),
    };
    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_edge<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    // Values at sigmoid boundaries: very negative, zero, very positive
    let values: Vec<T> = [-100.0f32, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0].iter().map(|&v| T::from(v).unwrap()).collect();
    let input = Input {
        data: values.into_boxed_slice(),
    };
    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
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
            &format!("Sigmoid basic test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
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
            &format!("Sigmoid large test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_edge<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
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
            &format!("Sigmoid edge test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

// basic tests
#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

// large tests
#[test]
fn test_large_f32() {
    test_large::<f32>();
}

#[test]
fn test_large_f16() {
    test_large::<f16>();
}

#[test]
fn test_large_bf16() {
    test_large::<bf16>();
}

// edge tests
#[test]
fn test_edge_f32() {
    test_edge::<f32>();
}

#[test]
fn test_edge_f16() {
    test_edge::<f16>();
}

#[test]
fn test_edge_bf16() {
    test_edge::<bf16>();
}
