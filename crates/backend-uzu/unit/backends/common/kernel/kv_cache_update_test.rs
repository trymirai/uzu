use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::Copy, kernel::KVCacheUpdateKernel},
        cpu::Cpu,
    },
    tests::assert::assert_eq_float,
};

struct Input<T: ArrayElement + Float> {
    keys: Box<[T]>,
    values: Box<[T]>,
    copies: Vec<Copy>,
    max_sequence_length: u32,
    element_dim: u32,
}

fn make_data<T: ArrayElement + Float>(
    max_sequence_length: usize,
    element_dim: usize,
) -> (Vec<T>, Vec<T>) {
    let total = max_sequence_length * element_dim;
    let mut keys = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    for i in 0..total {
        keys.push(T::from(1.0 + i as f32 * 0.1).unwrap());
        values.push(T::from(100.0 + i as f32 * 0.1).unwrap());
    }
    (keys, values)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, T::data_type())
        .expect("Failed to create KVCacheUpdateKernel");

    let total = input.max_sequence_length as usize * input.element_dim as usize;
    let mut keys = context.create_array_from(&[total], &input.keys).into_allocation();
    let mut values = context.create_array_from(&[total], &input.values).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
    kernel.encode(&mut keys, &mut values, &input.copies, input.copies.len() as u32, input.element_dim, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (crate::tests::helpers::allocation_to_vec(&keys), crate::tests::helpers::allocation_to_vec(&values))
}

/// Single copy between two different positions.
fn get_test_data_single_copy<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 4usize;
    let head_dim = 3usize;
    let element_dim = num_heads * head_dim;

    let (keys, values) = make_data::<T>(max_sequence_length, element_dim);

    let copies = vec![Copy {
        source: 0,
        destination: 2,
    }];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        copies: copies.clone(),
        max_sequence_length: max_sequence_length as u32,
        element_dim: element_dim as u32,
    };

    let (expected_keys, expected_values) = get_output::<T, Cpu>(&input);
    (input, expected_keys, expected_values)
}

/// Multiple copies.
fn get_test_data_multi_copy<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 8usize;
    let head_dim = 4usize;
    let element_dim = num_heads * head_dim;

    let (keys, values) = make_data::<T>(max_sequence_length, element_dim);

    let copies = vec![
        Copy {
            source: 1,
            destination: 3,
        },
        Copy {
            source: 0,
            destination: 5,
        },
    ];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        copies: copies.clone(),
        max_sequence_length: max_sequence_length as u32,
        element_dim: element_dim as u32,
    };

    let (expected_keys, expected_values) = get_output::<T, Cpu>(&input);
    (input, expected_keys, expected_values)
}

/// No copies — data should remain unchanged.
fn get_test_data_no_copy<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 4usize;
    let head_dim = 3usize;
    let element_dim = num_heads * head_dim;

    let (keys, values) = make_data::<T>(max_sequence_length, element_dim);

    let expected_keys = keys.clone();
    let expected_values = values.clone();

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        copies: vec![],
        max_sequence_length: max_sequence_length as u32,
        element_dim: element_dim as u32,
    };

    (input, expected_keys, expected_values)
}

/// Larger dimensions to exercise more threads.
fn get_test_data_large<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 8usize;
    let max_sequence_length = 64usize;
    let head_dim = 128usize;
    let element_dim = num_heads * head_dim;

    let (keys, values) = make_data::<T>(max_sequence_length, element_dim);

    let copies = vec![
        Copy {
            source: 0,
            destination: 63,
        },
        Copy {
            source: 10,
            destination: 30,
        },
        Copy {
            source: 5,
            destination: 20,
        },
    ];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        copies: copies.clone(),
        max_sequence_length: max_sequence_length as u32,
        element_dim: element_dim as u32,
    };

    let (expected_keys, expected_values) = get_output::<T, Cpu>(&input);
    (input, expected_keys, expected_values)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected_keys: &[T],
    expected_values: &[T],
    test_name: &str,
) {
    let eps = 1e-5;
    for_each_non_cpu_backend!(|B| {
        let (actual_keys, actual_values) = get_output::<T, B>(input);
        let msg = format!("KVCacheUpdate {} keys failed with backend={}", test_name, std::any::type_name::<B>(),);
        assert_eq_float::<T>(expected_keys, &actual_keys, eps, &msg);
        let msg = format!("KVCacheUpdate {} values failed with backend={}", test_name, std::any::type_name::<B>(),);
        assert_eq_float::<T>(expected_values, &actual_values, eps, &msg);
    });
}

fn test_single_copy_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_single_copy::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "single_copy");
}

fn test_multi_copy_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_multi_copy::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "multi_copy");
}

fn test_no_copy_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_no_copy::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "no_copy");
}

fn test_large_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_large::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "large");
}

// Single copy tests
#[uzu_test]
fn test_single_copy_f32() {
    test_single_copy_internal::<f32>();
}

#[uzu_test]
fn test_single_copy_f16() {
    test_single_copy_internal::<f16>();
}

#[uzu_test]
fn test_single_copy_bf16() {
    test_single_copy_internal::<bf16>();
}

// Multi copy tests
#[uzu_test]
fn test_multi_copy_f32() {
    test_multi_copy_internal::<f32>();
}

#[uzu_test]
fn test_multi_copy_f16() {
    test_multi_copy_internal::<f16>();
}

#[uzu_test]
fn test_multi_copy_bf16() {
    test_multi_copy_internal::<bf16>();
}

// No copy tests
#[uzu_test]
fn test_no_copy_f32() {
    test_no_copy_internal::<f32>();
}

#[uzu_test]
fn test_no_copy_f16() {
    test_no_copy_internal::<f16>();
}

#[uzu_test]
fn test_no_copy_bf16() {
    test_no_copy_internal::<bf16>();
}

// Large dimension tests
#[uzu_test]
fn test_large_f32() {
    test_large_internal::<f32>();
}

#[uzu_test]
fn test_large_f16() {
    test_large_internal::<f16>();
}

#[uzu_test]
fn test_large_bf16() {
    test_large_internal::<bf16>();
}
