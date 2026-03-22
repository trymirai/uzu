use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, kernel::attention::AttentionGemmArguments},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    mask: Option<Box<[T]>>,
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
    with_mask: bool,
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

    // mask: [suffix_length, sequence_length] (broadcast over heads)
    let mask = if with_mask {
        let m_size = suffix_length * sequence_length;
        let mut m = vec![T::zero(); m_size];
        for q in 0..suffix_length {
            for k in 0..sequence_length {
                let idx = q * sequence_length + k;
                let diff = (k as i32) - (sequence_length as i32 - suffix_length as i32 + q as i32);
                if diff.abs() > 3 {
                    m[idx] = T::from(-1e9f32).unwrap();
                }
            }
        }
        Some(m.into_boxed_slice())
    } else {
        None
    };

    let scale = 1.0 / (head_dim as f32).sqrt();

    let input = Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        mask,
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

    let queries_array = context.create_array_from(&[input.queries.len()], &input.queries, "");
    let keys_array = context.create_array_from(&[input.keys.len()], &input.keys, "");
    let values_array = context.create_array_from(&[input.values.len()], &input.values, "");

    let output_size = input.suffix_length * input.num_heads * input.head_dim;
    let output_array = context.create_array_uninitialized(&[output_size], T::data_type(), "");

    let mask_array = input.mask.as_ref().map(|m| context.create_array_from(&[m.len()], m, ""));

    let segment_prefix_length = input.sequence_length - input.suffix_length;

    {
        let q_buf_rc = queries_array.buffer();
        let q_buf = q_buf_rc.borrow();
        let k_buf_rc = keys_array.buffer();
        let k_buf = k_buf_rc.borrow();
        let v_buf_rc = values_array.buffer();
        let v_buf = v_buf_rc.borrow();
        let o_buf_rc = output_array.buffer();
        let mut o_buf = o_buf_rc.borrow_mut();

        let mask_buf_rc = mask_array.as_ref().map(|a| a.buffer());
        let mask_buf_borrow = mask_buf_rc.as_ref().map(|rc| rc.borrow());
        let mask_buffer: Option<&B::Buffer> = mask_buf_borrow.as_ref().map(|b| b.deref());

        let args = AttentionGemmArguments::<B> {
            queries_buffer: q_buf.deref(),
            keys_buffer: k_buf.deref(),
            values_buffer: v_buf.deref(),
            output_buffer: o_buf.deref_mut(),
            mask_buffer,
            sinks_buffer: None,
            num_heads: input.num_heads,
            num_groups: input.num_kv_heads,
            suffix_length: input.suffix_length,
            sequence_length: input.sequence_length,
            segment_prefix_length,
            max_sequence_length: input.sequence_length,
            head_dim: input.head_dim,
            is_causal: input.do_causal,
            scale: input.scale,
        };

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        block.encode(&context, &mut encoder, args).expect("Failed to encode AttentionGemm");
        encoder.end_encoding().submit().wait_until_completed().unwrap();
    }

    output_array.as_slice().to_vec()
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
            "AttentionGemm failed (backend={}, heads={}, kv_heads={}, seq={}, suffix={}, head_dim={}, causal={}, mask={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.num_kv_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
            input.do_causal,
            input.mask.is_some(),
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // Non-causal, no mask, single token
    let (input, expected) = get_test_data::<T>(4, 4, 8, 1, 64, false, false);
    test_internal(&input, &expected);

    // Non-causal, no mask, multiple tokens
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, false, false);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    // Causal, single token decode
    let (input, expected) = get_test_data::<T>(4, 4, 16, 1, 64, true, false);
    test_internal(&input, &expected);

    // Causal, multi-token prefill
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, true, false);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    // GQA: 8 query heads, 2 kv heads
    let (input, expected) = get_test_data::<T>(8, 2, 8, 1, 64, false, false);
    test_internal(&input, &expected);

    // GQA causal
    let (input, expected) = get_test_data::<T>(8, 2, 8, 4, 64, true, false);
    test_internal(&input, &expected);
}

fn test_mask<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, false, true);
    test_internal(&input, &expected);

    // Causal + mask
    let (input, expected) = get_test_data::<T>(4, 4, 8, 4, 64, true, true);
    test_internal(&input, &expected);
}

fn test_head_dim<T: ArrayElement + Float + Debug + Display>(head_dim: usize) {
    let (input, expected) = get_test_data::<T>(4, 4, 8, 2, head_dim, true, false);
    test_internal(&input, &expected);
}

fn test_unaligned<T: ArrayElement + Float + Debug + Display>() {
    // suffix_length not aligned to BQ=32
    let (input, expected) = get_test_data::<T>(4, 4, 40, 7, 64, true, false);
    test_internal(&input, &expected);

    // sequence_length not aligned to BK
    let (input, expected) = get_test_data::<T>(4, 4, 13, 4, 64, false, false);
    test_internal(&input, &expected);
}

// Basic tests
#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

// Causal tests
#[test]
fn test_causal_f32() {
    test_causal::<f32>();
}

#[test]
fn test_causal_f16() {
    test_causal::<f16>();
}

#[test]
fn test_causal_bf16() {
    test_causal::<bf16>();
}

// GQA tests
#[test]
fn test_gqa_f32() {
    test_gqa::<f32>();
}

#[test]
fn test_gqa_f16() {
    test_gqa::<f16>();
}

#[test]
fn test_gqa_bf16() {
    test_gqa::<bf16>();
}

// Mask tests
#[test]
fn test_mask_f32() {
    test_mask::<f32>();
}

#[test]
fn test_mask_f16() {
    test_mask::<f16>();
}

#[test]
fn test_mask_bf16() {
    test_mask::<bf16>();
}

// Head dim 128
#[test]
fn test_head_dim_128_f32() {
    test_head_dim::<f32>(128);
}

#[test]
fn test_head_dim_128_f16() {
    test_head_dim::<f16>(128);
}

#[test]
fn test_head_dim_128_bf16() {
    test_head_dim::<bf16>(128);
}

// Unaligned tests
#[test]
fn test_unaligned_f32() {
    test_unaligned::<f32>();
}

#[test]
fn test_unaligned_f16() {
    test_unaligned::<f16>();
}

#[test]
fn test_unaligned_bf16() {
    test_unaligned::<bf16>();
}
