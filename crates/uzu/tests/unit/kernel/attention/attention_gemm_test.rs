use std::{fmt::{Debug, Display}, mem::size_of};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Allocation, AllocationType, Backend, Buffer, Context, Encoder, kernel::attention::AttentionGemmBlock},
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

    let block = AttentionGemmBlock::<B>::new(T::data_type());

    let queries = allocation_from_slice::<T, B>(context.as_ref(), &input.queries);
    let keys = allocation_from_slice::<T, B>(context.as_ref(), &input.keys);
    let values = allocation_from_slice::<T, B>(context.as_ref(), &input.values);

    let output_size = input.suffix_length * input.num_heads * input.head_dim;
    let mut output = context
        .create_allocation(output_size * size_of::<T>(), AllocationType::Global)
        .expect("Failed to create output allocation");

    let segment_prefix_length = input.sequence_length - input.suffix_length;
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    block
        .encode(
            &context,
            &mut encoder,
            &queries,
            &keys,
            &values,
            &mut output,
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            input.num_heads,
            input.num_kv_heads,
            input.suffix_length,
            input.sequence_length,
            segment_prefix_length,
            input.sequence_length,
            None,
            input.head_dim,
            None,
            input.do_causal,
            input.scale,
        )
        .expect("Failed to encode AttentionGemm");
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<T, B>(&output)
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

fn allocation_to_vec<T: ArrayElement, B: Backend>(allocation: &Allocation<B>) -> Vec<T> {
    let (buffer, range) = allocation.as_buffer_range();
    let byte_len = range.end - range.start;
    let element_count = byte_len / size_of::<T>();
    unsafe {
        let src = (buffer.cpu_ptr().as_ptr() as *const u8).add(range.start);
        let bytes = std::slice::from_raw_parts(src, byte_len);
        bytemuck::cast_slice(bytes)[..element_count].to_vec()
    }
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
