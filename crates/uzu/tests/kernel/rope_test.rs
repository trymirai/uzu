use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels, kernel::RopeKernel,
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    qkv: Box<[T]>,
    cosines: Box<[T]>,
    sines: Box<[T]>,
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

    let mut cosines = vec![T::zero(); cos_sin_size];
    let mut sines = vec![T::zero(); cos_sin_size];
    for i in 0..cos_sin_size {
        cosines[i] = T::from(((i as f32) * 0.05).cos()).unwrap();
        sines[i] = T::from(((i as f32) * 0.05).sin()).unwrap();
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

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::RopeKernel::new(&context, T::data_type())
        .expect("Failed to create RopeKernel");

    let total_heads = (input.num_heads + 2 * input.num_groups) as usize;
    let qkv_len = input.suffix_length as usize * total_heads * input.head_dim as usize;
    let cos_sin_len = input.max_sequence_length as usize * input.rope_dim as usize;
    let queries_len = input.num_heads as usize * input.suffix_length as usize * input.head_dim as usize;
    let keys_len = input.num_groups as usize * input.suffix_length as usize * input.head_dim as usize;

    let qkv_array = context.create_array_from(&[qkv_len], &input.qkv, "");
    let cosines_array = context.create_array_from(&[cos_sin_len], &input.cosines, "");
    let sines_array = context.create_array_from(&[cos_sin_len], &input.sines, "");
    let token_positions_array = context.create_array_from(&[input.suffix_length as usize], &input.token_positions, "");
    let rotated_queries_array = context.create_array_uninitialized(&[queries_len], T::data_type(), "");
    let rotated_keys_array = context.create_array_uninitialized(&[keys_len], T::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        qkv_array.buffer().borrow().deref(),
        cosines_array.buffer().borrow().deref(),
        sines_array.buffer().borrow().deref(),
        token_positions_array.buffer().borrow().deref(),
        rotated_queries_array.buffer().borrow_mut().deref_mut(),
        rotated_keys_array.buffer().borrow_mut().deref_mut(),
        input.head_dim,
        input.rope_dim,
        input.num_heads,
        input.num_groups,
        input.suffix_length,
        input.max_sequence_length,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    (rotated_queries_array.as_slice().to_vec(), rotated_keys_array.as_slice().to_vec())
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(input: &Input<T>) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    let (expected_queries, expected_keys) = get_output::<T, Cpu>(input);

    for_each_non_cpu_backend!(|B| {
        let (queries, keys) = get_output::<T, B>(input);
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
    // Partial RoPE: only first 64 of 128 dims are rotated, rest pass through
    let input = get_test_data::<T>(32, 8, 128, 64, 4, 512);
    test_internal(&input);
}

fn test_partial_rope_small<T: ArrayElement + Float + Debug + Display>() {
    // Minimal partial RoPE: head_dim=8, rope_dim=4
    let input = get_test_data::<T>(2, 1, 8, 4, 1, 8);
    test_internal(&input);
}

fn test_nonzero_positions<T: ArrayElement + Float + Debug + Display>() {
    let mut input = get_test_data::<T>(4, 2, 8, 8, 3, 64);
    input.token_positions = vec![10, 11, 12].into_boxed_slice();
    test_internal(&input);
}

// f32 tests
#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[test]
fn test_mha_f32() {
    test_mha::<f32>();
}

#[test]
fn test_single_token_f32() {
    test_single_token::<f32>();
}

#[test]
fn test_small_f32() {
    test_small::<f32>();
}

#[test]
fn test_nonzero_positions_f32() {
    test_nonzero_positions::<f32>();
}

// f16 tests
#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_mha_f16() {
    test_mha::<f16>();
}

#[test]
fn test_single_token_f16() {
    test_single_token::<f16>();
}

#[test]
fn test_small_f16() {
    test_small::<f16>();
}

// bf16 tests
#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[test]
fn test_mha_bf16() {
    test_mha::<bf16>();
}

#[test]
fn test_single_token_bf16() {
    test_single_token::<bf16>();
}

#[test]
fn test_small_bf16() {
    test_small::<bf16>();
}

#[test]
fn test_partial_rope_basic_f32() {
    test_partial_rope_basic::<f32>();
}

#[test]
fn test_partial_rope_small_f32() {
    test_partial_rope_small::<f32>();
}
