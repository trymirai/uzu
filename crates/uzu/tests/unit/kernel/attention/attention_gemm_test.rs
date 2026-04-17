use std::{
    fmt::{Debug, Display},
    mem::size_of,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    allocation_as_slice,
    backends::{
        common::{
            Allocation, AllocationType, Backend, Buffer, Context, Encoder,
            kernel::{
                attention::{AttentionGemmArguments, AttentionGemmBlock},
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
                ManualKernels,
            },
        },
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

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

fn allocation_from_slice<T: ArrayElement, B: Backend>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let allocation = context
        .create_allocation(data.len() * size_of::<T>(), AllocationType::Global)
        .expect("Failed to create allocation");
    let (buffer, range) = allocation.as_buffer_range();
    let bytes = bytemuck::cast_slice(data);
    unsafe {
        let dst = (buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
    allocation
}

fn get_test_data<T: ArrayElement + Float>(
    num_heads: usize,
    num_kv_heads: usize,
    sequence_length: usize,
    suffix_length: usize,
    head_dim: usize,
    do_causal: bool,
) -> (Input<T>, Vec<T>) {
    // queries: [num_heads, suffix_length, head_dim] (contiguous, row-major)
    let q_size = num_heads * suffix_length * head_dim;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    // keys: [num_kv_heads, sequence_length, head_dim]
    let k_size = num_kv_heads * sequence_length * head_dim;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    // values: [num_kv_heads, sequence_length, head_dim]
    let v_size = num_kv_heads * sequence_length * head_dim;
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

    let block = uzu::backends::common::kernel::attention::AttentionGemmBlock::<B>::new(T::data_type());

    let queries_allocation = allocation_from_slice(context.as_ref(), &input.queries);
    let keys_allocation = allocation_from_slice(context.as_ref(), &input.keys);
    let values_allocation = allocation_from_slice(context.as_ref(), &input.values);

    let output_size = input.suffix_length * input.num_heads * input.head_dim * size_of::<T>();
    let mut output_allocation = context
        .create_allocation(output_size, AllocationType::Global)
        .expect("Failed to create output allocation");

    let segment_prefix_length = input.sequence_length - input.suffix_length;

    let args = AttentionGemmArguments {
        queries_buffer: &queries_allocation,
        keys_buffer: &keys_allocation,
        values_buffer: &values_allocation,
        output_buffer: &mut output_allocation,
        trie_buffer: None,
        sinks_buffer: None,
        num_heads: input.num_heads,
        num_groups: input.num_kv_heads,
        suffix_length: input.suffix_length,
        sequence_length: input.sequence_length,
        segment_prefix_length,
        max_sequence_length: input.sequence_length,
        ring_params: None,
        head_dim: input.head_dim,
        sliding_window_size: None,
        is_causal: input.do_causal,
        scale: input.scale,
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    block.encode(&context, &mut encoder, args).expect("Failed to encode AttentionGemm");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_as_slice::<T, B>(&output_allocation).to_vec()
}

fn get_output_with_padded_kv_cache<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    max_sequence_length: usize,
) -> Vec<T> {
    assert!(max_sequence_length >= input.sequence_length);

    let context = B::Context::new().expect("Failed to create Context");
    let block = uzu::backends::common::kernel::attention::AttentionGemmBlock::<B>::new(T::data_type());

    let queries_allocation = allocation_from_slice(context.as_ref(), &input.queries);

    let padded_kv_len = input.num_kv_heads * max_sequence_length * input.head_dim;
    let mut padded_keys = vec![T::zero(); padded_kv_len];
    let mut padded_values = vec![T::zero(); padded_kv_len];
    let head_span = max_sequence_length * input.head_dim;
    let source_head_span = input.sequence_length * input.head_dim;

    for head in 0..input.num_kv_heads {
        let src_start = head * source_head_span;
        let dst_start = head * head_span;
        padded_keys[dst_start..dst_start + source_head_span]
            .copy_from_slice(&input.keys[src_start..src_start + source_head_span]);
        padded_values[dst_start..dst_start + source_head_span]
            .copy_from_slice(&input.values[src_start..src_start + source_head_span]);
    }

    let keys_allocation = allocation_from_slice(context.as_ref(), &padded_keys);
    let values_allocation = allocation_from_slice(context.as_ref(), &padded_values);

    let output_size = input.suffix_length * input.num_heads * input.head_dim * size_of::<T>();
    let mut output_allocation = context
        .create_allocation(output_size, AllocationType::Global)
        .expect("Failed to create output allocation");

    let segment_prefix_length = input.sequence_length - input.suffix_length;

    let args = AttentionGemmArguments {
        queries_buffer: &queries_allocation,
        keys_buffer: &keys_allocation,
        values_buffer: &values_allocation,
        output_buffer: &mut output_allocation,
        trie_buffer: None,
        sinks_buffer: None,
        num_heads: input.num_heads,
        num_groups: input.num_kv_heads,
        suffix_length: input.suffix_length,
        sequence_length: input.sequence_length,
        segment_prefix_length,
        max_sequence_length,
        ring_params: None,
        head_dim: input.head_dim,
        sliding_window_size: None,
        is_causal: input.do_causal,
        scale: input.scale,
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    block.encode(&context, &mut encoder, args).expect("Failed to encode padded AttentionGemm");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_as_slice::<T, B>(&output_allocation).to_vec()
}

fn get_output_with_pooled_scratch_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let block = uzu::backends::common::kernel::attention::AttentionGemmBlock::<B>::new(T::data_type());

    let queries_allocation = allocation_from_slice(context.as_ref(), &input.queries);
    let keys_allocation = allocation_from_slice(context.as_ref(), &input.keys);
    let values_allocation = allocation_from_slice(context.as_ref(), &input.values);

    let output_size = input.suffix_length * input.num_heads * input.head_dim * size_of::<T>();
    let segment_prefix_length = input.sequence_length - input.suffix_length;

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    {
        let dirty = encoder.allocate_scratch(output_size).expect("Failed to allocate dirty scratch");
        let (buffer, range) = dirty.as_buffer_range();
        encoder.encode_fill(buffer, range, 0x7f);
    }

    let mut output_allocation = encoder.allocate_scratch(output_size).expect("Failed to allocate pooled output");
    let output_copy = context
        .create_allocation(output_size, AllocationType::Global)
        .expect("Failed to allocate global output copy");

    let args = AttentionGemmArguments {
        queries_buffer: &queries_allocation,
        keys_buffer: &keys_allocation,
        values_buffer: &values_allocation,
        output_buffer: &mut output_allocation,
        trie_buffer: None,
        sinks_buffer: None,
        num_heads: input.num_heads,
        num_groups: input.num_kv_heads,
        suffix_length: input.suffix_length,
        sequence_length: input.sequence_length,
        segment_prefix_length,
        max_sequence_length: input.sequence_length,
        ring_params: None,
        head_dim: input.head_dim,
        sliding_window_size: None,
        is_causal: input.do_causal,
        scale: input.scale,
    };

    block.encode(&context, &mut encoder, args).expect("Failed to encode pooled AttentionGemm");
    {
        let (src_buffer, src_range) = output_allocation.as_buffer_range();
        let (dst_buffer, dst_range) = output_copy.as_buffer_range();
        encoder.encode_copy(src_buffer, src_range, dst_buffer, dst_range);
    }
    drop(output_allocation);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_as_slice::<T, B>(&output_copy).to_vec()
}

fn get_output_with_followup_matmul<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    output_dim: usize,
) -> Vec<T>
where
    B::Kernels: ManualKernels,
{
    let context = B::Context::new().expect("Failed to create Context");
    let attention = AttentionGemmBlock::<B>::new(T::data_type());
    let mut matmul =
        <B::Kernels as ManualKernels>::MatmulKernel::new(context.as_ref(), T::data_type()).expect("Failed to create matmul");

    let queries_allocation = allocation_from_slice(context.as_ref(), &input.queries);
    let keys_allocation = allocation_from_slice(context.as_ref(), &input.keys);
    let values_allocation = allocation_from_slice(context.as_ref(), &input.values);

    let input_dim = input.num_heads * input.head_dim;
    let weights = (0..output_dim * input_dim)
        .map(|i| T::from((i as f32 * 0.017 + 0.25).cos() * 0.5).unwrap())
        .collect::<Vec<_>>();
    let weights_allocation = allocation_from_slice::<T, B>(context.as_ref(), &weights);

    let attention_output_size = input.suffix_length * input_dim * size_of::<T>();
    let matmul_output_size = input.suffix_length * output_dim * size_of::<T>();
    let segment_prefix_length = input.sequence_length - input.suffix_length;

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    {
        let dirty = encoder
            .allocate_scratch(attention_output_size)
            .expect("Failed to allocate dirty scratch");
        let (buffer, range) = dirty.as_buffer_range();
        encoder.encode_fill(buffer, range, 0x5a);
    }

    let mut attention_output = encoder
        .allocate_scratch(attention_output_size)
        .expect("Failed to allocate attention scratch output");
    let mut matmul_output = context
        .create_allocation(matmul_output_size, AllocationType::Global)
        .expect("Failed to create matmul output allocation");

    attention
        .encode(
            &context,
            &mut encoder,
            AttentionGemmArguments {
                queries_buffer: &queries_allocation,
                keys_buffer: &keys_allocation,
                values_buffer: &values_allocation,
                output_buffer: &mut attention_output,
                trie_buffer: None,
                sinks_buffer: None,
                num_heads: input.num_heads,
                num_groups: input.num_kv_heads,
                suffix_length: input.suffix_length,
                sequence_length: input.sequence_length,
                segment_prefix_length,
                max_sequence_length: input.sequence_length,
                ring_params: None,
                head_dim: input.head_dim,
                sliding_window_size: None,
                is_causal: input.do_causal,
                scale: input.scale,
            },
        )
        .expect("Failed to encode chained AttentionGemm");

    matmul.encode(
        context.as_ref(),
        MatmulArguments {
            a: &attention_output,
            b: &weights_allocation,
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: &mut matmul_output,
            batch_dim: input.suffix_length as u32,
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
        },
        &mut encoder,
    );

    drop(attention_output);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_as_slice::<T, B>(&matmul_output).to_vec()
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
    // Non-causal, single token
    let (input, expected) = get_test_data::<T>(4, 4, 8, 1, 64, false);
    test_internal(&input, &expected);

    // Non-causal, multiple tokens
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, false);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    // Causal, single token decode
    let (input, expected) = get_test_data::<T>(4, 4, 16, 1, 64, true);
    test_internal(&input, &expected);

    // Causal, multi-token prefill
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    // GQA: 8 query heads, 2 kv heads
    let (input, expected) = get_test_data::<T>(8, 2, 8, 1, 64, false);
    test_internal(&input, &expected);

    // GQA causal
    let (input, expected) = get_test_data::<T>(8, 2, 8, 4, 64, true);
    test_internal(&input, &expected);
}

fn test_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: usize) {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 2, head_dim, true);
    test_internal(&input, &expected);
}

fn test_unaligned<T: ArrayElement + Float + Debug + Display>() {
    // suffix_length not aligned to BQ=32
    let (input, expected) = get_test_data::<T>(4, 4, 40, 7, 64, true);
    test_internal(&input, &expected);

    // sequence_length not aligned to BK
    let (input, expected) = get_test_data::<T>(4, 4, 13, 4, 64, false);
    test_internal(&input, &expected);
}

fn test_model_like_prefill<T: ArrayElement + Float + Debug + Display>() {
    // Llama-3.2-1B-Instruct style attention shape.
    // This matches the prompt-sized causal prefill regime that the text session exercises.
    let (input, expected) = get_test_data::<T>(32, 8, 38, 38, 64, true);
    test_internal(&input, &expected);
}

fn test_model_like_prefill_with_padded_kv_cache<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(32, 8, 38, 38, 64, true);
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output_with_padded_kv_cache::<T, B>(&input, 256);
        let msg = format!(
            "AttentionGemm failed with padded KV cache (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.num_kv_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
        );
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

fn test_model_like_prefill_with_pooled_scratch_output<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(32, 8, 38, 38, 64, true);
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output_with_pooled_scratch_output::<T, B>(&input);
        let msg = format!(
            "AttentionGemm failed with pooled scratch output (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.num_kv_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
        );
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

fn test_model_like_prefill_followed_by_matmul<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected_attention) = get_test_data::<T>(32, 8, 38, 38, 64, true);
    let input_dim = input.num_heads * input.head_dim;
    let output_dim = 256usize;
    let weights = (0..output_dim * input_dim)
        .map(|i| T::from((i as f32 * 0.017 + 0.25).cos() * 0.5).unwrap())
        .collect::<Vec<_>>();

    let mut expected = vec![T::zero(); input.suffix_length * output_dim];
    for row in 0..input.suffix_length {
        for out_col in 0..output_dim {
            let mut acc = 0.0f32;
            for k in 0..input_dim {
                let a = expected_attention[row * input_dim + k].to_f32().unwrap();
                let b = weights[out_col * input_dim + k].to_f32().unwrap();
                acc += a * b;
            }
            expected[row * output_dim + out_col] = T::from(acc).unwrap();
        }
    }

    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2
    } else {
        1e-4
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output_with_followup_matmul::<T, B>(&input, output_dim);
        let msg = format!(
            "AttentionGemm followed by Matmul failed (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.num_kv_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
        );
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
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

// Causal tests
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

// GQA tests
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

// Head dim 128
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

// Unaligned tests
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
fn test_model_like_prefill_f32() {
    test_model_like_prefill::<f32>();
}

#[uzu_test]
fn test_model_like_prefill_f16() {
    test_model_like_prefill::<f16>();
}

#[uzu_test]
fn test_model_like_prefill_bf16() {
    test_model_like_prefill::<bf16>();
}

#[uzu_test]
fn test_model_like_prefill_with_padded_kv_cache_f32() {
    test_model_like_prefill_with_padded_kv_cache::<f32>();
}

#[uzu_test]
fn test_model_like_prefill_with_padded_kv_cache_f16() {
    test_model_like_prefill_with_padded_kv_cache::<f16>();
}

#[uzu_test]
fn test_model_like_prefill_with_padded_kv_cache_bf16() {
    test_model_like_prefill_with_padded_kv_cache::<bf16>();
}

#[uzu_test]
fn test_model_like_prefill_with_pooled_scratch_output_f32() {
    test_model_like_prefill_with_pooled_scratch_output::<f32>();
}

#[uzu_test]
fn test_model_like_prefill_with_pooled_scratch_output_f16() {
    test_model_like_prefill_with_pooled_scratch_output::<f16>();
}

#[uzu_test]
fn test_model_like_prefill_with_pooled_scratch_output_bf16() {
    test_model_like_prefill_with_pooled_scratch_output::<bf16>();
}

#[uzu_test]
fn test_model_like_prefill_followed_by_matmul_f32() {
    test_model_like_prefill_followed_by_matmul::<f32>();
}

#[uzu_test]
fn test_model_like_prefill_followed_by_matmul_f16() {
    test_model_like_prefill_followed_by_matmul::<f16>();
}

#[uzu_test]
fn test_model_like_prefill_followed_by_matmul_bf16() {
    test_model_like_prefill_followed_by_matmul::<bf16>();
}
