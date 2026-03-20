use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
            Context, Kernels, kernel::AttentionSinglePassKernel,
        },
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    mask: Option<Box<[T]>>,
    num_heads: u32,
    gqa_factor: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    scale: f32,
    do_causal: bool,
    has_mask: bool,
}

fn get_input<T: ArrayElement + Float>(
    num_heads: u32,
    num_kv_heads: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    do_causal: bool,
    with_mask: bool,
) -> Input<T> {
    let gqa_factor = num_heads / num_kv_heads;

    // queries: [num_heads * suffix_length, head_dim]
    let q_size = (num_heads * suffix_length * head_dim) as usize;
    let mut queries = vec![T::zero(); q_size];
    for i in 0..q_size {
        queries[i] = T::from((i as f32 * 0.13 + 0.5).sin() * 0.5).unwrap();
    }

    // keys: [num_kv_heads, sequence_length, head_dim]
    let k_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut keys = vec![T::zero(); k_size];
    for i in 0..k_size {
        keys[i] = T::from((i as f32 * 0.07 + 1.0).cos() * 0.5).unwrap();
    }

    // values: [num_kv_heads, sequence_length, head_dim]
    let v_size = (num_kv_heads * sequence_length * head_dim) as usize;
    let mut values = vec![T::zero(); v_size];
    for i in 0..v_size {
        values[i] = T::from((i as f32 * 0.11 + 2.0).sin() * 0.5).unwrap();
    }

    // mask: [sequence_length * suffix_length] (broadcast over heads with stride 0)
    let mask = if with_mask {
        let m_size = (suffix_length * sequence_length) as usize;
        let mut m = vec![T::zero(); m_size];
        for q in 0..suffix_length {
            for k in 0..sequence_length {
                let idx = (q * sequence_length + k) as usize;
                // Create a sliding window mask: allow only nearby keys
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

    Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        mask,
        num_heads,
        gqa_factor,
        sequence_length,
        suffix_length,
        head_dim,
        scale,
        do_causal,
        has_mask: with_mask,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let float_mask = input.has_mask;
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        T::data_type(),
        input.head_dim,
        float_mask,
        input.has_mask,
        false,
        input.do_causal,
    )
    .expect("Failed to create AttentionSinglePassKernel");

    let queries_array = context.create_array_from(&[input.queries.len()], &input.queries, "");
    let keys_array = context.create_array_from(&[input.keys.len()], &input.keys, "");
    let values_array = context.create_array_from(&[input.values.len()], &input.values, "");

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let output_array = context.create_array_uninitialized(&[output_size], T::data_type(), "");

    let mask_array = input.mask.as_ref().map(|m| context.create_array_from(&[m.len()], m, ""));
    let mask_buf_rc = mask_array.as_ref().map(|a| a.buffer());
    let mask_buf_borrow = mask_buf_rc.as_ref().map(|rc| rc.borrow());
    let mask_buffer: Option<&B::Buffer> = mask_buf_borrow.as_ref().map(|b| b.deref());

    let mask_kv_seq_stride: Option<u32> = input.has_mask.then_some(1);
    let mask_q_seq_stride: Option<u32> = input.has_mask.then_some(input.sequence_length);
    let mask_head_stride: Option<u32> = input.has_mask.then_some(0);

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer").start_encoding();
    kernel.encode(
        queries_array.buffer().borrow().deref(),
        keys_array.buffer().borrow().deref(),
        values_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.gqa_factor,
        input.sequence_length,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.sequence_length * input.head_dim,
        input.head_dim,
        input.scale,
        mask_buffer,
        mask_kv_seq_stride,
        mask_q_seq_stride,
        mask_head_stride,
        None::<&B::Buffer>,
        input.num_heads,
        input.suffix_length,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

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
            "AttentionSinglePass failed (backend={}, heads={}, seq={}, suffix={}, head_dim={}, causal={}, mask={})",
            std::any::type_name::<B>(),
            input.num_heads,
            input.sequence_length,
            input.suffix_length,
            input.head_dim,
            input.do_causal,
            input.has_mask,
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // Non-causal, no mask, single token
    let input = get_input::<T>(4, 4, 8, 1, 64, false, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Non-causal, no mask, multiple tokens
    let input = get_input::<T>(4, 4, 8, 4, 64, false, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    // Causal, single token decode
    let input = get_input::<T>(4, 4, 16, 1, 64, true, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Causal, multi-token prefill
    let input = get_input::<T>(4, 4, 8, 4, 64, true, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    // GQA: 8 query heads, 2 kv heads
    let input = get_input::<T>(8, 2, 8, 1, 64, false, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // GQA causal
    let input = get_input::<T>(8, 2, 8, 4, 64, true, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_mask<T: ArrayElement + Float + Debug + Display>() {
    let input = get_input::<T>(4, 4, 8, 4, 64, false, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Causal + mask
    let input = get_input::<T>(4, 4, 8, 4, 64, true, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_head_dim_128<T: ArrayElement + Float + Debug + Display>() {
    let input = get_input::<T>(4, 4, 8, 2, 128, true, false);
    let expected = get_output::<T, Cpu>(&input);
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
    test_head_dim_128::<f32>();
}

#[test]
fn test_head_dim_128_f16() {
    test_head_dim_128::<f16>();
}

#[test]
fn test_head_dim_128_bf16() {
    test_head_dim_128::<bf16>();
}
