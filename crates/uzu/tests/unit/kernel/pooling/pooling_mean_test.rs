use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::PoolingMeanKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    input: Box<[T]>,
    seq_len: u32,
    hidden_dim: u32,
    batch_size: u32,
}

fn get_test_data<T: ArrayElement + Float>(
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
) -> (Input<T>, Vec<T>) {
    let len = (batch_size * seq_len * hidden_dim) as usize;
    let input: Vec<T> = (0..len).map(|i| T::from((i as f32 * 0.1).sin() * 2.0).unwrap()).collect();

    let input = Input {
        input: input.into_boxed_slice(),
        seq_len,
        hidden_dim,
        batch_size,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::PoolingMeanKernel::new(&context, T::data_type())
        .expect("Failed to create PoolingMeanKernel");

    let input_len = (input.batch_size * input.seq_len * input.hidden_dim) as usize;
    let output_len = (input.batch_size * input.hidden_dim) as usize;

    let input_array = context.create_array_from(&[input_len], &input.input, "");
    let output_array = context.create_array_uninitialized(&[output_len], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.seq_len,
        input.hidden_dim,
        input.batch_size,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    batch_size: u32,
    seq_len: u32,
    hidden_dim: u32,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    let (input, expected) = get_test_data::<T>(batch_size, seq_len, hidden_dim);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        let msg = format!("PoolingMean test failed for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &msg);
    });
}

// Basic tests
#[test]
fn test_basic_f32() {
    test_internal::<f32>(2, 16, 64);
}

#[test]
fn test_basic_f16() {
    test_internal::<f16>(2, 16, 64);
}

#[test]
fn test_basic_bf16() {
    test_internal::<bf16>(2, 16, 64);
}

// Single batch
#[test]
fn test_single_batch_f32() {
    test_internal::<f32>(1, 32, 128);
}

#[test]
fn test_single_batch_f16() {
    test_internal::<f16>(1, 32, 128);
}

#[test]
fn test_single_batch_bf16() {
    test_internal::<bf16>(1, 32, 128);
}

// Single token sequence — mean of one element equals that element
#[test]
fn test_single_token_f32() {
    test_internal::<f32>(2, 1, 64);
}

#[test]
fn test_single_token_f16() {
    test_internal::<f16>(2, 1, 64);
}

#[test]
fn test_single_token_bf16() {
    test_internal::<bf16>(2, 1, 64);
}

// Large hidden dim
#[test]
fn test_large_hidden_dim_f32() {
    test_internal::<f32>(2, 8, 512);
}

// Long sequence — more accumulation, tests precision
#[test]
fn test_long_seq_f32() {
    test_internal::<f32>(1, 256, 64);
}

// Non-aligned hidden dim
#[test]
fn test_non_aligned_f32() {
    test_internal::<f32>(3, 10, 100);
}

#[test]
fn test_non_aligned_f16() {
    test_internal::<f16>(3, 10, 100);
}

#[test]
fn test_non_aligned_bf16() {
    test_internal::<bf16>(3, 10, 100);
}
