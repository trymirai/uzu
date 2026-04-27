use std::{
    fmt::{Debug, Display},
    ops::DerefMut,
};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::Swap, kernel::KVCacheUpdateKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    keys: Box<[T]>,
    values: Box<[T]>,
    swaps: Vec<Swap>,
    num_heads: u32,
    max_sequence_length: u32,
    head_dim: u32,
}

fn make_data<T: ArrayElement + Float>(
    num_heads: usize,
    max_sequence_length: usize,
    head_dim: usize,
) -> (Vec<T>, Vec<T>) {
    let total = num_heads * max_sequence_length * head_dim;
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

    let total = input.num_heads as usize * input.max_sequence_length as usize * input.head_dim as usize;
    let keys_array = context.create_array_from(&[total], &input.keys, "");
    let values_array = context.create_array_from(&[total], &input.values, "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
    kernel.encode(
        keys_array.buffer().borrow_mut().deref_mut(),
        values_array.buffer().borrow_mut().deref_mut(),
        &input.swaps,
        input.swaps.len() as u32,
        input.num_heads,
        input.head_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (keys_array.as_slice().to_vec(), values_array.as_slice().to_vec())
}

/// Single swap between two different positions.
fn get_test_data_single_swap<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 4usize;
    let head_dim = 3usize;

    let (keys, values) = make_data::<T>(num_heads, max_sequence_length, head_dim);

    let swaps = vec![Swap {
        source: 0,
        destination: 2,
    }];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        swaps: swaps.clone(),
        num_heads: num_heads as u32,
        max_sequence_length: max_sequence_length as u32,
        head_dim: head_dim as u32,
    };

    let (expected_keys, expected_values) = get_output::<T, Cpu>(&input);
    (input, expected_keys, expected_values)
}

/// Multiple swaps.
fn get_test_data_multi_swap<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 8usize;
    let head_dim = 4usize;

    let (keys, values) = make_data::<T>(num_heads, max_sequence_length, head_dim);

    let swaps = vec![
        Swap {
            source: 1,
            destination: 3,
        },
        Swap {
            source: 0,
            destination: 5,
        },
    ];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        swaps: swaps.clone(),
        num_heads: num_heads as u32,
        max_sequence_length: max_sequence_length as u32,
        head_dim: head_dim as u32,
    };

    let (expected_keys, expected_values) = get_output::<T, Cpu>(&input);
    (input, expected_keys, expected_values)
}

/// No swaps — data should remain unchanged.
fn get_test_data_no_swap<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 2usize;
    let max_sequence_length = 4usize;
    let head_dim = 3usize;

    let (keys, values) = make_data::<T>(num_heads, max_sequence_length, head_dim);

    let expected_keys = keys.clone();
    let expected_values = values.clone();

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        swaps: vec![],
        num_heads: num_heads as u32,
        max_sequence_length: max_sequence_length as u32,
        head_dim: head_dim as u32,
    };

    (input, expected_keys, expected_values)
}

/// Larger dimensions to exercise more threads.
fn get_test_data_large<T: ArrayElement + Float>() -> (Input<T>, Vec<T>, Vec<T>) {
    let num_heads = 8usize;
    let max_sequence_length = 64usize;
    let head_dim = 128usize;

    let (keys, values) = make_data::<T>(num_heads, max_sequence_length, head_dim);

    let swaps = vec![
        Swap {
            source: 0,
            destination: 63,
        },
        Swap {
            source: 10,
            destination: 30,
        },
        Swap {
            source: 5,
            destination: 20,
        },
    ];

    let input = Input {
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        swaps: swaps.clone(),
        num_heads: num_heads as u32,
        max_sequence_length: max_sequence_length as u32,
        head_dim: head_dim as u32,
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

fn test_single_swap_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_single_swap::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "single_swap");
}

fn test_multi_swap_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_multi_swap::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "multi_swap");
}

fn test_no_swap_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_no_swap::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "no_swap");
}

fn test_large_internal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_keys, expected_values) = get_test_data_large::<T>();
    test_internal::<T>(&input, &expected_keys, &expected_values, "large");
}

// Single swap tests
#[uzu_test]
fn test_single_swap_f32() {
    test_single_swap_internal::<f32>();
}

#[uzu_test]
fn test_single_swap_f16() {
    test_single_swap_internal::<f16>();
}

#[uzu_test]
fn test_single_swap_bf16() {
    test_single_swap_internal::<bf16>();
}

// Multi swap tests
#[uzu_test]
fn test_multi_swap_f32() {
    test_multi_swap_internal::<f32>();
}

#[uzu_test]
fn test_multi_swap_f16() {
    test_multi_swap_internal::<f16>();
}

#[uzu_test]
fn test_multi_swap_bf16() {
    test_multi_swap_internal::<bf16>();
}

// No swap tests
#[uzu_test]
fn test_no_swap_f32() {
    test_no_swap_internal::<f32>();
}

#[uzu_test]
fn test_no_swap_f16() {
    test_no_swap_internal::<f16>();
}

#[uzu_test]
fn test_no_swap_bf16() {
    test_no_swap_internal::<bf16>();
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
