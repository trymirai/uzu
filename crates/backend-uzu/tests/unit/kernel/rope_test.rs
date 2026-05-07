use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::RopeKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    qkv: Box<[T]>,
    token_positions: Box<[i32]>,
    head_dim: u32,
    rope_dim: u32,
    rotary_pair_stride: u32,
    rotary_frequency_dim: u32,
    rope_max_sequence_length: u32,
    rope_scaling_type: u32,
    rope_base: f32,
    rope_scaling_factor: f32,
    rope_original_context_length: u32,
    rope_low_frequency_factor: f32,
    rope_high_frequency_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
    rope_truncate: u32,
    rope_attention_scaling_factor: f32,
    num_heads: u32,
    num_groups: u32,
    suffix_length: u32,
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

    let mut qkv = vec![T::zero(); qkv_size];
    for i in 0..qkv_size {
        qkv[i] = T::from(((i as f32) * 0.1).sin() * 2.0).unwrap();
    }

    let mut token_positions = vec![0i32; suffix_length as usize];
    for i in 0..suffix_length as usize {
        token_positions[i] = i as i32;
    }

    Input {
        qkv: qkv.into_boxed_slice(),
        token_positions: token_positions.into_boxed_slice(),
        head_dim,
        rope_dim,
        rotary_pair_stride: rope_dim / 2,
        rotary_frequency_dim: rope_dim,
        rope_max_sequence_length: max_sequence_length,
        rope_scaling_type: 0,
        rope_base: 10000.0,
        rope_scaling_factor: 1.0,
        rope_original_context_length: max_sequence_length,
        rope_low_frequency_factor: 1.0,
        rope_high_frequency_factor: 1.0,
        rope_beta_fast: 1.0,
        rope_beta_slow: 1.0,
        rope_truncate: 0,
        rope_attention_scaling_factor: 1.0,
        num_heads,
        num_groups,
        suffix_length,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::RopeKernel::new(&context, T::data_type())
        .expect("Failed to create RopeKernel");

    let total_heads = (input.num_heads + 2 * input.num_groups) as usize;
    let qkv_len = input.suffix_length as usize * total_heads * input.head_dim as usize;
    let queries_len = input.num_heads as usize * input.suffix_length as usize * input.head_dim as usize;
    let keys_len = input.num_groups as usize * input.suffix_length as usize * input.head_dim as usize;

    let qkv_array = context.create_array_from(&[qkv_len], &input.qkv, "");
    let token_positions_array = context.create_array_from(&[input.suffix_length as usize], &input.token_positions, "");
    let rotated_queries_array = context.create_array_uninitialized(&[queries_len], T::data_type(), "");
    let rotated_keys_array = context.create_array_uninitialized(&[keys_len], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        qkv_array.buffer().borrow().deref(),
        token_positions_array.buffer().borrow().deref(),
        rotated_queries_array.buffer().borrow_mut().deref_mut(),
        rotated_keys_array.buffer().borrow_mut().deref_mut(),
        input.head_dim,
        input.rope_dim,
        input.rotary_pair_stride,
        input.rotary_frequency_dim,
        input.rope_max_sequence_length,
        input.rope_scaling_type,
        input.rope_base,
        input.rope_scaling_factor,
        input.rope_original_context_length,
        input.rope_low_frequency_factor,
        input.rope_high_frequency_factor,
        input.rope_beta_fast,
        input.rope_beta_slow,
        input.rope_truncate,
        input.rope_attention_scaling_factor,
        input.num_heads,
        input.num_groups,
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

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
    let input = get_test_data::<T>(32, 8, 128, 64, 4, 512);
    test_internal(&input);
}

fn test_partial_rope_small<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data::<T>(2, 1, 8, 4, 1, 8);
    test_internal(&input);
}

fn test_partial_rope_proportional_layout() {
    let input = Input {
        qkv: (0..32).map(|value| value as f32).collect::<Box<_>>(),
        token_positions: [1].into(),
        head_dim: 8,
        rope_dim: 4,
        rotary_pair_stride: 4,
        rotary_frequency_dim: 8,
        rope_max_sequence_length: 8,
        rope_scaling_type: 0,
        rope_base: 10000.0,
        rope_scaling_factor: 1.0,
        rope_original_context_length: 8,
        rope_low_frequency_factor: 1.0,
        rope_high_frequency_factor: 1.0,
        rope_beta_fast: 1.0,
        rope_beta_slow: 1.0,
        rope_truncate: 0,
        rope_attention_scaling_factor: 1.0,
        num_heads: 2,
        num_groups: 1,
        suffix_length: 1,
    };
    let cos_0 = 1.0f32.cos();
    let sin_0 = 1.0f32.sin();
    let cos_1 = 0.1f32.cos();
    let sin_1 = 0.1f32.sin();

    let rotate_head = |head_values: &[f32]| {
        [
            head_values[0] * cos_0 - head_values[4] * sin_0,
            head_values[1] * cos_1 - head_values[5] * sin_1,
            head_values[2],
            head_values[3],
            head_values[4] * cos_0 + head_values[0] * sin_0,
            head_values[5] * cos_1 + head_values[1] * sin_1,
            head_values[6],
            head_values[7],
        ]
    };

    let expected_queries =
        rotate_head(&input.qkv[0..8]).into_iter().chain(rotate_head(&input.qkv[8..16])).collect::<Vec<_>>();
    let expected_keys = rotate_head(&input.qkv[16..24]).to_vec();
    let (cpu_queries, cpu_keys) = get_output::<f32, Cpu>(&input);
    assert_eq_float::<f32>(&expected_queries, &cpu_queries, 1e-5, "CPU proportional RoPE queries");
    assert_eq_float::<f32>(&expected_keys, &cpu_keys, 1e-5, "CPU proportional RoPE keys");

    for_each_non_cpu_backend!(|B| {
        let (queries, keys) = get_output::<f32, B>(&input);
        let msg = format!("Proportional RoPE queries test failed with backend={}", std::any::type_name::<B>(),);
        assert_eq_float::<f32>(&expected_queries, &queries, 1e-5, &msg);

        let msg = format!("Proportional RoPE keys test failed with backend={}", std::any::type_name::<B>(),);
        assert_eq_float::<f32>(&expected_keys, &keys, 1e-5, &msg);
    });
}

fn test_nonzero_positions<T: ArrayElement + Float + Debug + Display>() {
    let mut input = get_test_data::<T>(4, 2, 8, 8, 3, 64);
    input.token_positions = vec![10, 11, 12].into_boxed_slice();
    test_internal(&input);
}

fn test_out_of_range_positions<T: ArrayElement + Float + Debug + Display>() {
    let mut input = get_test_data::<T>(4, 2, 8, 8, 3, 64);
    input.token_positions = vec![63, 64, 65].into_boxed_slice();
    test_internal(&input);
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
fn test_out_of_range_positions_f32() {
    test_out_of_range_positions::<f32>();
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

#[uzu_test]
fn test_partial_rope_proportional_layout_f32() {
    test_partial_rope_proportional_layout();
}
