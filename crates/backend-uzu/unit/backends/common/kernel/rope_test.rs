use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::RopeKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, for_each_non_cpu_backend},
};

struct Input<T: ArrayElement + Float> {
    qkv: Box<[T]>,
    cosines: Box<[f32]>,
    sines: Box<[f32]>,
    token_positions: Box<[i32]>,
    head_dim: u32,
    rope_dim: u32,
    num_heads: u32,
    num_groups: u32,
    suffix_length: u32,
    max_sequence_length: u32,
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
    rope_dim: u32,
    suffix_length: u32,
    max_sequence_length: u32,
) -> Input<T> {
    let total_heads = (num_heads + 2 * num_groups) as usize;
    let qkv_size = suffix_length as usize * total_heads * head_dim as usize;
    let cos_sin_size = max_sequence_length as usize * rope_dim as usize;

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from(((i as f32) * 0.1).sin() * 2.0).unwrap();
    }

    let mut cosines = vec![0.0f32; cos_sin_size];
    let mut sines = vec![0.0f32; cos_sin_size];
    for i in 0..cos_sin_size {
        cosines[i] = ((i as f32) * 0.05).cos();
        sines[i] = ((i as f32) * 0.05).sin();
    }

    let mut token_positions = vec![0i32; suffix_length as usize];
    for i in 0..suffix_length as usize {
        token_positions[i] = i as i32;
    }

    Input {
        qkv: qkv.into_boxed_slice(),
        cosines: cosines.into_boxed_slice(),
        sines: sines.into_boxed_slice(),
        token_positions: token_positions.into_boxed_slice(),
        head_dim,
        rope_dim,
        num_heads,
        num_groups,
        suffix_length,
        max_sequence_length,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    query_only: bool,
) -> (Vec<T>, Option<Vec<T>>) {
    let context = B::Context::new().expect("Failed to create Context");

    let element_data_type = T::data_type();
    let rope_data_type = DataType::F32;
    let kernel =
        <<B as Backend>::Kernels as Kernels>::RopeKernel::new(&context, element_data_type, rope_data_type, query_only)
            .expect("Failed to create RopeKernel");

    let total_heads = (input.num_heads + 2 * input.num_groups) as usize;
    let qkv_len = input.suffix_length as usize * total_heads * input.head_dim as usize;
    let cos_sin_len = input.suffix_length as usize * input.rope_dim as usize;
    let queries_len = input.num_heads as usize * input.suffix_length as usize * input.head_dim as usize;
    let keys_len = input.num_groups as usize * input.suffix_length as usize * input.head_dim as usize;
    let mut cosines = Vec::with_capacity(cos_sin_len);
    let mut sines = Vec::with_capacity(cos_sin_len);
    for token_index in 0..input.suffix_length as usize {
        let raw_position = input.token_positions[token_index] as usize;
        let absolute_position = if raw_position >= input.max_sequence_length as usize {
            0
        } else {
            raw_position
        };
        let offset = absolute_position * input.rope_dim as usize;
        let end = offset + input.rope_dim as usize;
        cosines.extend_from_slice(&input.cosines[offset..end]);
        sines.extend_from_slice(&input.sines[offset..end]);
    }

    let qkv_array = if query_only {
        context.create_array_from(&[queries_len], &input.qkv[..queries_len])
    } else {
        context.create_array_from(&[qkv_len], &input.qkv)
    };
    let cosines_array = context.create_array_from(&[cos_sin_len], &cosines);
    let sines_array = context.create_array_from(&[cos_sin_len], &sines);
    let mut rotated_queries = context.create_array_uninitialized(&[queries_len], T::data_type()).into_allocation();
    let mut rotated_keys = context.create_array_uninitialized(&[keys_len], T::data_type()).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        qkv_array.allocation(),
        cosines_array.allocation(),
        sines_array.allocation(),
        &mut rotated_queries,
        if query_only {
            None::<&mut backend_uzu::backends::common::Allocation<B>>
        } else {
            Some(&mut rotated_keys)
        },
        input.head_dim,
        input.rope_dim,
        input.num_heads,
        if query_only {
            None
        } else {
            Some(input.num_groups)
        },
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let queries = crate::tests::helpers::allocation_to_vec(&rotated_queries);
    let keys = if query_only {
        None
    } else {
        Some(crate::tests::helpers::allocation_to_vec(&rotated_keys))
    };
    (queries, keys)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(input: &Input<T>) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    let (expected_queries, expected_keys) = get_output::<T, Cpu>(input, false);
    let expected_keys = expected_keys.expect("full rope returns keys");

    for_each_non_cpu_backend!(|B| {
        let (queries, keys) = get_output::<T, B>(input, false);
        let keys = keys.expect("full rope returns keys");
        let msg = format!("Rope queries test failed with backend={}", std::any::type_name::<B>(),);
        assert_eq_float::<T>(&expected_queries, &queries, eps, &msg);

        let msg = format!("Rope keys test failed with backend={}", std::any::type_name::<B>(),);
        assert_eq_float::<T>(&expected_keys, &keys, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // Typical GQA config: 32 query heads, 8 KV groups, head_dim=128, 4 tokens
    let input = get_test_data::<T>(32, 8, 128, 128, 4, 512);
    test_internal(&input);
}

fn test_mha<T: ArrayElement + Float + Debug + Display>() {
    // MHA: num_heads == num_groups
    let input = get_test_data::<T>(8, 8, 64, 64, 2, 256);
    test_internal(&input);
}

fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    // Single token (decode)
    let input = get_test_data::<T>(16, 4, 64, 64, 1, 1024);
    test_internal(&input);
}

fn test_small<T: ArrayElement + Float + Debug + Display>() {
    // Minimal config
    let input = get_test_data::<T>(2, 1, 4, 4, 1, 8);
    test_internal(&input);
}

fn test_partial_rope_basic<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data::<T>(32, 8, 128, 64, 4, 512);
    test_internal(&input);
}

fn test_partial_rope_small<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data::<T>(2, 1, 8, 4, 1, 8);
    test_internal(&input);
}

fn test_nonzero_positions<T: ArrayElement + Float + Debug + Display>() {
    let mut input = get_test_data::<T>(4, 2, 8, 8, 3, 64);
    input.token_positions = vec![10, 11, 12].into_boxed_slice();
    test_internal(&input);
}

fn test_query_only<T: ArrayElement + Float + Debug + Display>(input: &Input<T>) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let (expected, _) = get_output::<T, Cpu>(input, true);
    for_each_non_cpu_backend!(|B| {
        let (actual, _) = get_output::<T, B>(input, true);
        let msg = format!("Rope query-only test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps, &msg);
    });
}

// f32 tests
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_mha_f32() {
    test_mha::<f32>();
}

#[uzu_test]
fn test_single_token_f32() {
    test_single_token::<f32>();
}

#[uzu_test]
fn test_small_f32() {
    test_small::<f32>();
}

#[uzu_test]
fn test_nonzero_positions_f32() {
    test_nonzero_positions::<f32>();
}

#[uzu_test]
fn test_query_only_f32() {
    test_query_only::<f32>(&get_test_data::<f32>(8, 2, 64, 64, 4, 512));
}

#[uzu_test]
fn test_query_only_partial_f32() {
    test_query_only::<f32>(&get_test_data::<f32>(8, 2, 256, 128, 4, 512));
}

#[uzu_test]
fn test_query_only_bf16() {
    test_query_only::<bf16>(&get_test_data::<bf16>(8, 2, 64, 64, 4, 512));
}

// f16 tests
#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_mha_f16() {
    test_mha::<f16>();
}

#[uzu_test]
fn test_single_token_f16() {
    test_single_token::<f16>();
}

#[uzu_test]
fn test_small_f16() {
    test_small::<f16>();
}

// bf16 tests
#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[uzu_test]
fn test_mha_bf16() {
    test_mha::<bf16>();
}

#[uzu_test]
fn test_single_token_bf16() {
    test_single_token::<bf16>();
}

#[uzu_test]
fn test_small_bf16() {
    test_small::<bf16>();
}

#[uzu_test]
fn test_partial_rope_basic_f32() {
    test_partial_rope_basic::<f32>();
}

#[uzu_test]
fn test_partial_rope_small_f32() {
    test_partial_rope_small::<f32>();
}
