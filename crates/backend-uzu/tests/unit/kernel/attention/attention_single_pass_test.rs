use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::AttentionSinglePassKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    queries: Box<[T]>,
    keys: Box<[T]>,
    values: Box<[T]>,
    num_heads: u32,
    gqa_factor: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    scale: f32,
    do_causal: bool,
}

fn get_input<T: ArrayElement + Float>(
    num_heads: u32,
    num_kv_heads: u32,
    sequence_length: u32,
    suffix_length: u32,
    head_dim: u32,
    do_causal: bool,
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

    let scale = 1.0 / (head_dim as f32).sqrt();

    Input {
        queries: queries.into_boxed_slice(),
        keys: keys.into_boxed_slice(),
        values: values.into_boxed_slice(),
        num_heads,
        gqa_factor,
        sequence_length,
        suffix_length,
        head_dim,
        scale,
        do_causal,
    }
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionSinglePassKernel::new(
        &context,
        T::data_type(),
        input.head_dim,
        false,
        false,
        input.do_causal,
        false,
        false,
    )
    .expect("Failed to create AttentionSinglePassKernel");

    let queries_array = context.create_array_from(&[input.queries.len()], &input.queries, "");
    let keys_array = context.create_array_from(&[input.keys.len()], &input.keys, "");
    let values_array = context.create_array_from(&[input.values.len()], &input.values, "");

    let output_size = (input.suffix_length * input.num_heads * input.head_dim) as usize;
    let output_array = context.create_array_uninitialized(&[output_size], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
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
        None,
        input.scale,
        None::<&B::DenseBuffer>,
        None,
        None::<&B::DenseBuffer>,
        input.num_heads,
        input.suffix_length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

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
            "AttentionSinglePass failed (backend={}, heads={}, seq={}, suffix={}, head_dim={}, causal={})",
            std::any::type_name::<B>(),
            input.num_heads,
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
    let input = get_input::<T>(4, 4, 8, 1, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Non-causal, multiple tokens
    let input = get_input::<T>(4, 4, 8, 4, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_causal<T: ArrayElement + Float + Debug + Display>() {
    // Causal, single token decode
    let input = get_input::<T>(4, 4, 16, 1, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // Causal, multi-token prefill
    let input = get_input::<T>(4, 4, 8, 4, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_gqa<T: ArrayElement + Float + Debug + Display>() {
    // GQA: 8 query heads, 2 kv heads
    let input = get_input::<T>(8, 2, 8, 1, 64, false);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);

    // GQA causal
    let input = get_input::<T>(8, 2, 8, 4, 64, true);
    let expected = get_output::<T, Cpu>(&input);
    test_internal(&input, &expected);
}

fn test_head_dim_128<T: ArrayElement + Float + Debug + Display>() {
    let input = get_input::<T>(4, 4, 8, 2, 128, true);
    let expected = get_output::<T, Cpu>(&input);
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
    test_head_dim_128::<f32>();
}

#[uzu_test]
fn test_head_dim_128_f16() {
    test_head_dim_128::<f16>();
}

#[uzu_test]
fn test_head_dim_128_bf16() {
    test_head_dim_128::<bf16>();
}
