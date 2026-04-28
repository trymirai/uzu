use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::AttentionUpdateKVCacheKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    rotated_keys: Option<Box<[T]>>,
    qkv: Box<[T]>,
    key_cache: Box<[T]>,
    value_cache: Box<[T]>,
    num_groups: u32,
    num_heads: u32,
    head_dim: u32,
    suffix_length: u32,
    prefix_segment_length: u32,
    max_sequence_length: u32,
    keys_in_place: bool,
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(
        &context,
        T::data_type(),
        input.keys_in_place,
    )
    .expect("Failed to create AttentionUpdateKVCacheKernel");

    let cache_size = input.num_groups as usize * input.max_sequence_length as usize * input.head_dim as usize;

    let rotated_keys_array = input.rotated_keys.as_ref().map(|rk| context.create_array_from(&[rk.len()], rk, ""));
    let rotated_keys_buf_rc = rotated_keys_array.as_ref().map(|a| a.buffer());
    let rotated_keys_buf_borrow = rotated_keys_buf_rc.as_ref().map(|rc| rc.borrow());
    let rotated_keys_buffer: Option<&B::Buffer> = rotated_keys_buf_borrow.as_ref().map(|b| b.deref());

    let qkv_array = context.create_array_from(&[input.qkv.len()], &input.qkv, "");
    let key_cache_array = context.create_array_from(&[cache_size], &input.key_cache, "");
    let value_cache_array = context.create_array_from(&[cache_size], &input.value_cache, "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        rotated_keys_buffer,
        qkv_array.buffer().borrow().deref(),
        key_cache_array.buffer().borrow_mut().deref_mut(),
        value_cache_array.buffer().borrow_mut().deref_mut(),
        input.num_groups,
        input.num_heads,
        input.head_dim,
        input.suffix_length,
        input.prefix_segment_length,
        input.max_sequence_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (key_cache_array.as_slice().to_vec(), value_cache_array.as_slice().to_vec())
}

fn get_test_data_basic<T: ArrayElement + Float>(keys_in_place: bool) -> Input<T> {
    let num_groups = 4u32;
    let num_heads = 8u32;
    let head_dim = 4u32;
    let suffix_length = 3u32;
    let prefix_segment_length = 2u32;
    let max_sequence_length = 8u32;

    let cache_size = (num_groups * max_sequence_length * head_dim) as usize;
    let qkv_stride = ((num_heads + 2 * num_groups) * head_dim) as usize;
    let qkv_size = suffix_length as usize * qkv_stride;

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from(((i as f32) * 0.1).sin() * 5.0).unwrap();
    }

    let mut key_cache = vec![T::zero(); cache_size];
    let mut value_cache = vec![T::zero(); cache_size];
    for i in 0..cache_size {
        key_cache[i] = T::from(((i as f32) * 0.3 + 1.0).cos() * 2.0).unwrap();
        value_cache[i] = T::from(((i as f32) * 0.2 + 0.5).sin() * 3.0).unwrap();
    }

    let rotated_keys = if keys_in_place {
        None
    } else {
        let mut rk = vec![T::zero(); cache_size];
        for i in 0..cache_size {
            rk[i] = T::from(((i as f32) * 0.07 + 2.0).sin() * 4.0).unwrap();
        }
        Some(rk.into_boxed_slice())
    };

    Input {
        rotated_keys,
        qkv: qkv.into_boxed_slice(),
        key_cache: key_cache.into_boxed_slice(),
        value_cache: value_cache.into_boxed_slice(),
        num_groups,
        num_heads,
        head_dim,
        suffix_length,
        prefix_segment_length,
        max_sequence_length,
        keys_in_place,
    }
}

fn get_test_data_single_token<T: ArrayElement + Float>(keys_in_place: bool) -> Input<T> {
    let num_groups = 2u32;
    let num_heads = 8u32;
    let head_dim = 4u32;
    let suffix_length = 1u32;
    let prefix_segment_length = 5u32;
    let max_sequence_length = 16u32;

    let cache_size = (num_groups * max_sequence_length * head_dim) as usize;
    let qkv_stride = ((num_heads + 2 * num_groups) * head_dim) as usize;
    let qkv_size = suffix_length as usize * qkv_stride;

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from((i as f32 + 1.0) * 0.25).unwrap();
    }

    let mut key_cache = vec![T::zero(); cache_size];
    let mut value_cache = vec![T::zero(); cache_size];
    for i in 0..cache_size {
        key_cache[i] = T::from(i as f32 * 0.01).unwrap();
        value_cache[i] = T::from(i as f32 * -0.01).unwrap();
    }

    let rotated_keys = if keys_in_place {
        None
    } else {
        let mut rk = vec![T::zero(); cache_size];
        for i in 0..cache_size {
            rk[i] = T::from((i as f32 + 0.5) * 0.1).unwrap();
        }
        Some(rk.into_boxed_slice())
    };

    Input {
        rotated_keys,
        qkv: qkv.into_boxed_slice(),
        key_cache: key_cache.into_boxed_slice(),
        value_cache: value_cache.into_boxed_slice(),
        num_groups,
        num_heads,
        head_dim,
        suffix_length,
        prefix_segment_length,
        max_sequence_length,
        keys_in_place,
    }
}

fn get_test_data_prefix_zero<T: ArrayElement + Float>() -> Input<T> {
    let num_groups = 2u32;
    let num_heads = 4u32;
    let head_dim = 8u32;
    let suffix_length = 2u32;
    let prefix_segment_length = 0u32;
    let max_sequence_length = 4u32;

    let cache_size = (num_groups * max_sequence_length * head_dim) as usize;
    let qkv_stride = ((num_heads + 2 * num_groups) * head_dim) as usize;
    let qkv_size = suffix_length as usize * qkv_stride;

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from(((i as f32) * 0.13).sin() * 3.0).unwrap();
    }

    let key_cache = vec![T::zero(); cache_size];
    let value_cache = vec![T::zero(); cache_size];

    let mut rk = vec![T::zero(); cache_size];
    for i in 0..cache_size {
        rk[i] = T::from(((i as f32) * 0.17 + 1.0).cos() * 2.0).unwrap();
    }

    Input {
        rotated_keys: Some(rk.into_boxed_slice()),
        qkv: qkv.into_boxed_slice(),
        key_cache: key_cache.into_boxed_slice(),
        value_cache: value_cache.into_boxed_slice(),
        num_groups,
        num_heads,
        head_dim,
        suffix_length,
        prefix_segment_length,
        max_sequence_length,
        keys_in_place: false,
    }
}

fn get_test_data_keys_in_place_offset<T: ArrayElement + Float>() -> Input<T> {
    let num_groups = 2u32;
    let num_heads = 4u32;
    let head_dim = 2u32;
    let suffix_length = 1u32;
    let prefix_segment_length = 3u32;
    let max_sequence_length = 5u32;

    let cache_size = (num_groups * max_sequence_length * head_dim) as usize;
    let qkv_stride = ((num_heads + 2 * num_groups) * head_dim) as usize;
    let qkv_size = suffix_length as usize * qkv_stride;

    let mut qkv = vec![T::zero(); qkv_size];
    let value_base_offset = ((num_heads + num_groups) * head_dim) as usize;
    for g in 0..num_groups as usize {
        for d in 0..head_dim as usize {
            let off = value_base_offset + g * head_dim as usize + d;
            qkv[off] = T::from(200.0 + (g * head_dim as usize + d) as f32).unwrap();
        }
    }

    // Place rotated key sentinels at the rotated_key_offset positions of the
    // (in-place) source buffer: (g * suffix_length + t) * head_dim + d.
    let mut key_cache = vec![T::zero(); cache_size];
    for g in 0..num_groups as usize {
        for t in 0..suffix_length as usize {
            for d in 0..head_dim as usize {
                let off = (g * suffix_length as usize + t) * head_dim as usize + d;
                key_cache[off] =
                    T::from(100.0 + ((g * suffix_length as usize + t) * head_dim as usize + d) as f32).unwrap();
            }
        }
    }

    let value_cache = vec![T::zero(); cache_size];

    Input {
        rotated_keys: None,
        qkv: qkv.into_boxed_slice(),
        key_cache: key_cache.into_boxed_slice(),
        value_cache: value_cache.into_boxed_slice(),
        num_groups,
        num_heads,
        head_dim,
        suffix_length,
        prefix_segment_length,
        max_sequence_length,
        keys_in_place: true,
    }
}

fn test_keys_in_place_cache_offset<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data_keys_in_place_offset::<T>();
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.01f32
    } else {
        1e-6
    };

    for_each_non_cpu_backend!(|B| {
        let (key_cache_out, value_cache_out) = get_output::<T, B>(&input);
        let backend_name = std::any::type_name::<B>();

        for g in 0..input.num_groups {
            for t in 0..input.suffix_length {
                for d in 0..input.head_dim {
                    let cache_token_index = input.prefix_segment_length + t;
                    // keys_in_place layout:
                    //   cache_offset = (group * max_sequence_length
                    //                   + cache_token_index) * head_dim + dim
                    let cache_offset =
                        ((g * input.max_sequence_length + cache_token_index) * input.head_dim + d) as usize;

                    let rotated_key_offset = ((g * input.suffix_length + t) * input.head_dim + d) as usize;
                    let expected_key = T::from(100.0 + rotated_key_offset as f32).unwrap();
                    let value_lane = (g * input.head_dim + d) as usize;
                    let expected_value = T::from(200.0 + value_lane as f32).unwrap();

                    assert_eq_float::<T>(
                        &[expected_key],
                        &[key_cache_out[cache_offset]],
                        eps,
                        &format!(
                            "key_cache[{}] mismatch (backend={}, g={}, t={}, d={})",
                            cache_offset, backend_name, g, t, d,
                        ),
                    );
                    assert_eq_float::<T>(
                        &[expected_value],
                        &[value_cache_out[cache_offset]],
                        eps,
                        &format!(
                            "value_cache[{}] mismatch (backend={}, g={}, t={}, d={})",
                            cache_offset, backend_name, g, t, d,
                        ),
                    );
                }
            }
        }
    });
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected_key_cache: &[T],
    expected_value_cache: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.01f32
    } else {
        1e-6
    };

    for_each_non_cpu_backend!(|B| {
        let (key_cache_out, value_cache_out) = get_output::<T, B>(input);
        let backend_name = std::any::type_name::<B>();
        assert_eq_float::<T>(
            expected_key_cache,
            &key_cache_out,
            eps,
            &format!("key_cache mismatch (backend={}, keys_in_place={})", backend_name, input.keys_in_place),
        );
        assert_eq_float::<T>(
            expected_value_cache,
            &value_cache_out,
            eps,
            &format!("value_cache mismatch (backend={}, keys_in_place={})", backend_name, input.keys_in_place),
        );
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    for keys_in_place in [false] {
        let input = get_test_data_basic::<T>(keys_in_place);
        let (expected_key_cache, expected_value_cache) = get_output::<T, Cpu>(&input);
        test_internal(&input, &expected_key_cache, &expected_value_cache);
    }
}

fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    for keys_in_place in [false, true] {
        let input = get_test_data_single_token::<T>(keys_in_place);
        let (expected_key_cache, expected_value_cache) = get_output::<T, Cpu>(&input);
        test_internal(&input, &expected_key_cache, &expected_value_cache);
    }
}

fn test_prefix_zero<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data_prefix_zero::<T>();
    let (expected_key_cache, expected_value_cache) = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected_key_cache, &expected_value_cache);
}

// Basic tests
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

// Single token tests (typical decode step)
#[uzu_test]
fn test_single_token_f32() {
    test_single_token::<f32>();
}

#[uzu_test]
fn test_single_token_f16() {
    test_single_token::<f16>();
}

#[uzu_test]
fn test_single_token_bf16() {
    test_single_token::<bf16>();
}

// Prefix zero (first fill of empty cache)
#[uzu_test]
fn test_prefix_zero_f32() {
    test_prefix_zero::<f32>();
}

#[uzu_test]
fn test_prefix_zero_f16() {
    test_prefix_zero::<f16>();
}

#[uzu_test]
fn test_prefix_zero_bf16() {
    test_prefix_zero::<bf16>();
}

// Explicit cache_offset verification when keys_in_place == true.
#[uzu_test]
fn test_keys_in_place_cache_offset_f32() {
    test_keys_in_place_cache_offset::<f32>();
}

#[uzu_test]
fn test_keys_in_place_cache_offset_f16() {
    test_keys_in_place_cache_offset::<f16>();
}

#[uzu_test]
fn test_keys_in_place_cache_offset_bf16() {
    test_keys_in_place_cache_offset::<bf16>();
}
