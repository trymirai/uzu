use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use super::super::FullPrecisionInput;
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig},
        },
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataTypeRepr,
    encodable_block::mixer::attention::KVCacheView,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
    },
};

struct Input<T: DataTypeRepr + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
}

fn attention_config<T: DataTypeRepr + Float>(input: &Input<T>) -> AttentionKernelConfig {
    super::default_attention_config::<T>(input.head_dim, input.num_heads, input.num_kv_heads, input.do_causal)
}

fn get_test_data<T: DataTypeRepr + Float>(
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

    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        num_kv_heads,
        sequence_length,
        suffix_length,
        head_dim,
        do_causal,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: DataTypeRepr + Float, B: Backend>(fixture: &Input<T>) -> Vec<T> {
    let context = block_on(B::Context::new()).expect("Failed to create Context");

    let config = attention_config(fixture);

    let queries_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &fixture.queries);
    let keys_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &fixture.keys);
    let values_buffer = create_buffer_with_data::<B, T>(context.as_ref(), &fixture.values);

    let segment_prefix_length = fixture.sequence_length - fixture.suffix_length;
    let input = AttentionInput::FullPrecision(FullPrecisionInput {
        queries: &queries_buffer,
        keys: &keys_buffer,
        values: &values_buffer,
    });
    let args = AttentionArguments {
        suffix_length: fixture.suffix_length as u32,
        trie: None::<&B::GlobalGpuBuffer>,
        sinks: None,
        cache: KVCacheView::full(segment_prefix_length as u32),
        gate: None::<&B::GlobalGpuBuffer>,
    };

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    let kernel = <B::Kernels as Kernels>::AttentionKernel::new(context.as_ref(), config)
        .expect("Failed to create attention kernel");
    let pooled_output = kernel.encode(input, args, &mut command_buffer).expect("Failed to encode attention");
    let mut output_buffer =
        create_buffer::<B, T>(context.as_ref(), fixture.suffix_length * fixture.num_heads * fixture.head_dim);
    command_buffer.encode_copy(&pooled_output, &mut output_buffer);
    drop(pooled_output);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    buffer_to_vec::<B, T>(&context, &output_buffer)
}

fn get_gemm_output<T: DataTypeRepr + Float>(input: &Input<T>) -> Vec<T> {
    let context = block_on(<Metal as Backend>::Context::new()).expect("Failed to create Metal context");
    let config = attention_config(input);
    let queries = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.queries);
    let keys = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.keys);
    let values = create_buffer_with_data::<Metal, T>(context.as_ref(), &input.values);
    let cache = KVCacheView::full((input.sequence_length - input.suffix_length) as u32);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    let pooled = super::gemm::AttentionGemm::new(&config)
        .encode(
            FullPrecisionInput {
                queries: &queries,
                keys: &keys,
                values: &values,
            },
            AttentionArguments {
                suffix_length: input.suffix_length as u32,
                trie: None::<&<Metal as Backend>::GlobalGpuBuffer>,
                sinks: None,
                cache,
                gate: None::<&<Metal as Backend>::GlobalGpuBuffer>,
            },
            &mut command_buffer,
        )
        .expect("Failed to encode AttentionGemm");
    let mut output =
        create_buffer::<Metal, T>(context.as_ref(), input.suffix_length * input.num_heads * input.head_dim);
    command_buffer.encode_copy(&pooled, &mut output);
    drop(pooled);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<Metal, T>(&context, &output)
}

fn test_internal<T: DataTypeRepr + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let eps = 1e-2;

    let output = get_gemm_output(input);
    let msg = format!(
        "AttentionGemm failed (heads={}, kv_heads={}, seq={}, suffix={}, head_dim={}, causal={})",
        input.num_heads,
        input.num_kv_heads,
        input.sequence_length,
        input.suffix_length,
        input.head_dim,
        input.do_causal,
    );
    assert_eq_float::<T>(expected, &output, eps, &msg);
}

fn test_basic<T: DataTypeRepr + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 1, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, false);
    test_internal(&input, &expected);
}

fn test_causal<T: DataTypeRepr + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 16, 1, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_gqa<T: DataTypeRepr + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(8, 2, 8, 1, 64, false);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(8, 2, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_head_dim<T: DataTypeRepr + Float + Debug + Display>(head_dim: usize) {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 2, head_dim, true);
    test_internal(&input, &expected);
}

fn test_unaligned<T: DataTypeRepr + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 40, 7, 64, true);
    test_internal(&input, &expected);

    let (input, expected) = get_test_data::<T>(4, 4, 13, 4, 64, false);
    test_internal(&input, &expected);
}

#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[uzu_test]
fn test_causal_bf16() {
    test_causal::<bf16>();
}

#[uzu_test]
fn test_gqa_bf16() {
    test_gqa::<bf16>();
}

#[uzu_test]
fn test_head_dim_128_bf16() {
    test_head_dim::<bf16>(128);
}

#[uzu_test]
fn test_unaligned_bf16() {
    test_unaligned::<bf16>();
}

#[uzu_test]
fn test_prefill_mxu() {
    let (input, expected) = get_test_data::<bf16>(4, 4, 128, 128, 64, true);
    test_internal(&input, &expected);
    let (input, expected) = get_test_data::<bf16>(8, 2, 100, 96, 128, true);
    test_internal(&input, &expected);
}
