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

    let qkv = (0..qkv_size)
        .map(|index| T::from(((index as f32) * 0.1).sin() * 2.0).unwrap())
        .collect::<Box<_>>();

    let token_positions = (0..suffix_length).map(|index| index as i32).collect::<Box<_>>();

    Input {
        qkv,
        token_positions,
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
    let input = get_test_data::<T>(32, 8, 128, 128, 4, 512);
    test_internal(&input);
}

fn test_mha<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data::<T>(8, 8, 64, 64, 2, 256);
    test_internal(&input);
}

fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    let input = get_test_data::<T>(16, 4, 64, 64, 1, 1024);
    test_internal(&input);
}

fn test_small<T: ArrayElement + Float + Debug + Display>() {
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

fn fixed_fixture_input(
    rope_scaling_type: u32,
    rope_scaling_factor: f32,
    rope_original_context_length: u32,
    rope_low_frequency_factor: f32,
    rope_high_frequency_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
    rope_truncate: u32,
    rope_attention_scaling_factor: f32,
) -> Input<f32> {
    Input {
        qkv: (0..12).map(|value| value as f32).collect::<Box<_>>(),
        token_positions: [1].into(),
        head_dim: 4,
        rope_dim: 4,
        rotary_pair_stride: 2,
        rotary_frequency_dim: 4,
        rope_max_sequence_length: 8,
        rope_scaling_type,
        rope_base: 10000.0,
        rope_scaling_factor,
        rope_original_context_length,
        rope_low_frequency_factor,
        rope_high_frequency_factor,
        rope_beta_fast,
        rope_beta_slow,
        rope_truncate,
        rope_attention_scaling_factor,
        num_heads: 1,
        num_groups: 1,
        suffix_length: 1,
    }
}

fn assert_fixed_fixture(
    input: Input<f32>,
    expected_queries: &[f32],
    expected_keys: &[f32],
    fixture_name: &str,
) {
    let (cpu_queries, cpu_keys) = get_output::<f32, Cpu>(&input);
    assert_eq_float::<f32>(expected_queries, &cpu_queries, 1e-5, &format!("{fixture_name} CPU queries"));
    assert_eq_float::<f32>(expected_keys, &cpu_keys, 1e-5, &format!("{fixture_name} CPU keys"));

    for_each_non_cpu_backend!(|B| {
        let (queries, keys) = get_output::<f32, B>(&input);
        let msg = format!("{fixture_name} queries test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<f32>(expected_queries, &queries, 1e-4, &msg);

        let msg = format!("{fixture_name} keys test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<f32>(expected_keys, &keys, 1e-4, &msg);
    });
}

fn test_unscaled_fixed_fixture() {
    assert_fixed_fixture(
        fixed_fixture_input(0, 1.0, 8, 1.0, 4.0, 32.0, 1.0, 0, 1.0),
        &[-1.682941970, 0.969950500, 1.080604612, 3.009849835],
        &[-2.887616685, 4.929751169, 6.607697774, 7.049649170],
        "unscaled RoPE fixed fixture",
    );
}

fn test_linear_fixed_fixture() {
    assert_fixed_fixture(
        fixed_fixture_input(1, 2.0, 8, 1.0, 4.0, 32.0, 1.0, 0, 1.0),
        &[-0.958851077, 0.984987563, 1.755165124, 3.004962479],
        &[0.633777016, 4.964937646, 7.183197526, 7.024912396],
        "linear RoPE fixed fixture",
    );
}

fn test_llama_fixed_fixture() {
    assert_fixed_fixture(
        fixed_fixture_input(2, 8.0, 8, 1.0, 4.0, 32.0, 1.0, 0, 1.0),
        &[-0.406536814, 0.996249220, 1.958246108, 3.001247656],
        &[2.696881775, 4.991246096, 6.687811951, 7.006244530],
        "llama RoPE fixed fixture",
    );
}

fn test_yarn_fixed_fixture() {
    assert_fixed_fixture(
        fixed_fixture_input(3, 4.0, 8, 1.0, 4.0, 32.0, 1.0, 0, 1.138629436),
        &[-1.916247266, 1.130086166, 1.230408220, 3.418724204],
        &[-3.287925358, 5.673203395, 7.523719191, 7.984613998],
        "yarn RoPE fixed fixture",
    );
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

#[uzu_test]
fn test_unscaled_fixed_fixture_f32() {
    test_unscaled_fixed_fixture();
}

#[uzu_test]
fn test_linear_fixed_fixture_f32() {
    test_linear_fixed_fixture();
}

#[uzu_test]
fn test_llama_fixed_fixture_f32() {
    test_llama_fixed_fixture();
}

#[uzu_test]
fn test_yarn_fixed_fixture_f32() {
    test_yarn_fixed_fixture();
}
