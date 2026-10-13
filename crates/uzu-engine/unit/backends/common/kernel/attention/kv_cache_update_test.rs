use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use crate::backends::metal::Metal;
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::Copy,
            kernel::{KVCacheUpdateInt8Kernel, KVCacheUpdateKernel, kv_layout::Int8KVLayout},
        },
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
    },
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

    let mut keys = create_buffer_with_data::<B, T>(&context, &input.keys);
    let mut values = create_buffer_with_data::<B, T>(&context, &input.values);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
    kernel.encode(
        &mut keys,
        &mut values,
        &input.copies,
        input.copies.len() as u32,
        input.element_dim,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    (buffer_to_vec(&keys), buffer_to_vec(&values))
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
        Copy {
            source: 3,
            destination: 6,
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
fn test_single_copy_bf16() {
    test_single_copy_internal::<bf16>();
}

// Multi copy tests
#[uzu_test]
fn test_multi_copy_bf16() {
    test_multi_copy_internal::<bf16>();
}

// No copy tests
#[uzu_test]
fn test_no_copy_bf16() {
    test_no_copy_internal::<bf16>();
}

// Large dimension tests
#[uzu_test]
fn test_large_bf16() {
    test_large_internal::<bf16>();
}

#[uzu_test]
fn test_int8_copy() {
    test_int8_copy_for::<Cpu>();
    #[cfg(backend = "metal")]
    test_int8_copy_for::<Metal>();
}

fn test_int8_copy_for<B: Backend>() {
    let layout = Int8KVLayout::new(256, 2, 64);
    let k_codes: Vec<i8> = (0..layout.code_bytes()).map(|i| ((i * 37) ^ (i >> 8)) as i8).collect();
    let v_codes: Vec<u8> = (0..layout.code_bytes()).map(|i| ((i * 19) ^ (i >> 7)) as u8).collect();
    let k_scales: Vec<bf16> = (0..layout.scale_count()).map(|i| bf16::from_f32(i as f32 + 1.0)).collect();
    let v_scales: Vec<bf16> = (0..layout.scale_count()).map(|i| bf16::from_f32(i as f32 + 129.0)).collect();
    let copies = [Copy {
        source: 40,
        destination: 5,
    }];
    let context = B::Context::new().unwrap();
    let mut keys = create_buffer_with_data::<B, i8>(&context, &k_codes);
    let mut values = create_buffer_with_data::<B, u8>(&context, &v_codes);
    let mut key_scales = create_buffer_with_data::<B, bf16>(&context, &k_scales);
    let mut value_scales = create_buffer_with_data::<B, bf16>(&context, &v_scales);
    let kernel =
        <B::Kernels as Kernels>::KVCacheUpdateInt8Kernel::new(&context, layout.head_dim, layout.num_kv_heads).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &mut keys,
        &mut values,
        &mut key_scales,
        &mut value_scales,
        &copies,
        copies.len() as u32,
        layout.padded_capacity,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    let (keys, values) = (buffer_to_vec::<B, i8>(&keys), buffer_to_vec::<B, u8>(&values));
    for channel in 0..layout.head_dim {
        assert_eq!(keys[layout.k_code_index(5, 1, channel)], k_codes[layout.k_code_index(40, 1, channel)]);
        assert_eq!(values[layout.v_code_index(5, 1, channel)], v_codes[layout.v_code_index(40, 1, channel)]);
    }
    let (from, to) = ((layout.padded_capacity + 40) as usize, (layout.padded_capacity + 5) as usize);
    assert_eq!(buffer_to_vec::<B, bf16>(&key_scales)[to], k_scales[from]);
    assert_eq!(buffer_to_vec::<B, bf16>(&value_scales)[to], v_scales[from]);
}
