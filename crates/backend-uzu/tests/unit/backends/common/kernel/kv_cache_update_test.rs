use std::fmt::{Debug, Display};

use half::{bf16, f16};
#[cfg(metal_backend)]
use ndarray::{Array, Array3};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::Copy, kernel::KVCacheUpdateKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec},
    },
};
#[cfg(metal_backend)]
use crate::{
    backends::metal::Metal,
    data_type::DataType,
    tests::helpers::{sparse_buffer_create_with, sparse_buffer_read_vec},
};

struct Input<T: ArrayElement + Float> {
    keys: Box<[T]>,
    values: Box<[T]>,
    copies: Vec<Copy>,
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

    let mut keys = alloc_allocation_with_data::<B, T>(&context, &input.keys);
    let mut values = alloc_allocation_with_data::<B, T>(&context, &input.values);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
    kernel.encode(&mut keys, &mut values, &input.copies, input.copies.len() as u32, input.element_dim, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&keys), allocation_to_vec(&values))
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

#[cfg(metal_backend)]
fn apply_copies_3d<T: Clone>(
    array: &mut Array3<T>,
    copies: &[Copy],
) {
    let (_seq_len, num_heads, head_dim) = array.dim();
    for head in 0..num_heads {
        for channel in 0..head_dim {
            for copy in copies {
                let src = copy.source as usize;
                let dst = copy.destination as usize;
                array[(dst, head, channel)] = array[(src, head, channel)].clone();
            }
        }
    }
}

#[cfg(metal_backend)]
#[uzu_test]
fn test_sparse_random_pattern_f32() {
    let context = match <Metal as Backend>::Context::new() {
        Ok(context) => context,
        Err(error) => {
            eprintln!("Failed to create Metal context: {error:?}. Skipping test.");
            return;
        },
    };

    let kernel = <<Metal as Backend>::Kernels as Kernels>::KVCacheUpdateKernel::new(&context, DataType::F32)
        .expect("Failed to create KVCacheUpdateKernel");

    let num_heads = 3usize;
    let seq_len = 15usize;
    let head_dim = 7usize;
    let element_dim = num_heads * head_dim;

    let key_data = Array3::<f32>::from_shape_fn((seq_len, num_heads, head_dim), |(token, head, channel)| {
        (token * 1_000_000 + head * 100 + channel * 10) as f32
    });

    let value_data = Array3::<f32>::from_shape_fn((seq_len, num_heads, head_dim), |(token, head, channel)| {
        (token * 1_000_000 + head * 100 + channel * 10 + 1_000) as f32
    });

    let copies = vec![
        Copy {
            source: 0,
            destination: 14,
        },
        Copy {
            source: 3,
            destination: 11,
        },
        Copy {
            source: 6,
            destination: 8,
        },
        Copy {
            source: 9,
            destination: 5,
        },
        Copy {
            source: 12,
            destination: 2,
        },
        Copy {
            source: 2,
            destination: 12,
        },
        Copy {
            source: 5,
            destination: 9,
        },
        Copy {
            source: 8,
            destination: 6,
        },
        Copy {
            source: 11,
            destination: 3,
        },
        Copy {
            source: 14,
            destination: 0,
        },
    ];

    let mut expected_keys = key_data.clone();
    let mut expected_values = value_data.clone();
    apply_copies_3d(&mut expected_keys, &copies);
    apply_copies_3d(&mut expected_values, &copies);

    let mut key_buffer = sparse_buffer_create_with::<Metal, f32>(&context, key_data.as_slice().unwrap());
    let mut value_buffer = sparse_buffer_create_with::<Metal, f32>(&context, value_data.as_slice().unwrap());

    let mut encoder = Encoder::<Metal>::new(&context).unwrap();
    kernel.encode(&mut key_buffer, &mut value_buffer, &copies, copies.len() as u32, element_dim as u32, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let elements_count = seq_len * num_heads * head_dim;
    let key_values: Vec<f32> = sparse_buffer_read_vec::<Metal, f32>(&context, &key_buffer, elements_count);
    let key_result = Array::from_shape_vec((seq_len, num_heads, head_dim), key_values)
        .expect("Failed to convert key result to ndarray");

    let value_values: Vec<f32> = sparse_buffer_read_vec::<Metal, f32>(&context, &value_buffer, elements_count);
    let value_result = Array::from_shape_vec((seq_len, num_heads, head_dim), value_values)
        .expect("Failed to convert value result to ndarray");

    assert_eq!(key_result, expected_keys);
    assert_eq!(value_result, expected_values);
}
