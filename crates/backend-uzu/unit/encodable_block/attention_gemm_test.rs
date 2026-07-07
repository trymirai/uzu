use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use super::AttentionCore;
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder},
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::mixer::attention::{
        core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
        state::AttentionStateType,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
) -> (Input<T>, Vec<T>) {
    let q_size = num_heads * suffix_length * head_dim;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    let k_size = sequence_length * num_kv_heads * head_dim;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    let v_size = sequence_length * num_kv_heads * head_dim;
    let mut values = vec![T::zero(); v_size];
    for i in 0..v_size {
        values[i] = T::from((i as f32 * 0.11 + 2.0).sin() * 0.5).unwrap();
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        num_kv_heads,
        sequence_length,
        suffix_length,
        head_dim,
        scale,
        do_causal,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let core = AttentionCore::<B>::new(
        AttentionCoreNewArguments {
            head_dim: input.head_dim,
            num_groups: input.num_kv_heads,
            num_q_heads: input.num_heads,
            has_sinks: false,
            is_kv_cache_ring: false,
            is_causal: input.do_causal,
            is_trie: false,
            sliding_window_size: None,
            scale: Some(input.scale),
            data_type: T::data_type(),
        },
        context.as_ref(),
    )
    .expect("Failed to create AttentionCore");

    let queries_allocation = alloc_allocation_with_data::<B, T>(context.as_ref(), &input.queries);
    let keys_allocation = alloc_allocation_with_data::<B, T>(context.as_ref(), &input.keys);
    let values_allocation = alloc_allocation_with_data::<B, T>(context.as_ref(), &input.values);

    let segment_prefix_length = input.sequence_length - input.suffix_length;
    let state_type = AttentionStateType::Full {
        length: segment_prefix_length,
    };

    let args = AttentionCoreEncodeArguments {
        queries: &queries_allocation,
        keys: &keys_allocation,
        values: &values_allocation,
        suffix_length: input.suffix_length,
        trie: None,
        sinks: None,
        state_type: &state_type,
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let pooled_output = core.encode(args, &mut encoder).expect("Failed to encode AttentionGemm");
    let mut output_allocation =
        alloc_allocation::<B, T>(context.as_ref(), input.suffix_length * input.num_heads * input.head_dim);
    encoder.encode_copy(&pooled_output, .., &mut output_allocation, ..);
    let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
    drop(pooled_output);
    drop(completed);

    allocation_to_vec::<B, T>(&output_allocation)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(input);
        let msg = format!(
            "AttentionGemm failed (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={}, causal={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.num_kv_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
            input.do_causal,
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 16, 9, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 16, 16, 64, false);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 16, 9, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 16, 16, 64, true);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(8, 2, 16, 9, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(8, 2, 16, 16, 64, true);
    test_internal(&input, &expected);
}

fn test_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: usize) {
    let (input, expected) = get_test_data::<T>(4, 4, 16, 9, head_dim, true);
    test_internal(&input, &expected);
}

fn test_unaligned<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 40, 11, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 21, 13, 64, false);
    test_internal(&input, &expected);
}

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

#[uzu_test]
fn test_causal_f32() {
    test_causal::<f32>();
}

#[uzu_test]
fn test_causal_f16() {
    test_causal::<f16>();
}

#[uzu_test]
fn test_causal_bf16() {
    test_causal::<bf16>();
}

#[uzu_test]
fn test_gqa_f32() {
    test_gqa::<f32>();
}

#[uzu_test]
fn test_gqa_f16() {
    test_gqa::<f16>();
}

#[uzu_test]
fn test_gqa_bf16() {
    test_gqa::<bf16>();
}

#[uzu_test]
fn test_head_dim_128_f32() {
    test_head_dim::<f32>(128);
}

#[uzu_test]
fn test_head_dim_128_f16() {
    test_head_dim::<f16>(128);
}

#[uzu_test]
fn test_head_dim_128_bf16() {
    test_head_dim::<bf16>(128);
}

#[uzu_test]
fn test_unaligned_f32() {
    test_unaligned::<f32>();
}

#[uzu_test]
fn test_unaligned_f16() {
    test_unaligned::<f16>();
}

#[uzu_test]
fn test_unaligned_bf16() {
    test_unaligned::<bf16>();
}

#[uzu_test]
fn test_prefill_mxu() {
    let (input, expected) = get_test_data::<f16>(4, 4, 128, 128, 64, true);
    test_internal(&input, &expected);
    let (input, expected) = get_test_data::<bf16>(8, 2, 100, 96, 128, true);
    test_internal(&input, &expected);
}
