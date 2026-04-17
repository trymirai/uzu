use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::Conv1dPackKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    state_in: Box<[T]>,
    x: Box<[T]>,
    state_stride: u32,
    row_stride: u32,
    suffix_len: u32,
    num_channels: u32,
}

fn get_input<T: ArrayElement + Float>(
    state_stride: u32,
    row_stride: u32,
    suffix_len: u32,
    num_channels: u32,
) -> Input<T> {
    let state_size = num_channels as usize * state_stride as usize;
    let x_size = suffix_len as usize * row_stride as usize;

    let state_in: Vec<T> = (0..state_size).map(|i| T::from(0.1 * (i as f64 + 1.0)).unwrap()).collect();
    let x: Vec<T> = (0..x_size).map(|i| T::from(-0.05 * (i as f64 + 1.0)).unwrap()).collect();

    Input {
        state_in: state_in.into_boxed_slice(),
        x: x.into_boxed_slice(),
        state_stride,
        row_stride,
        suffix_len,
        num_channels,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::Conv1dPackKernel::new(&context, T::data_type())
        .expect("Failed to create Conv1dPackKernel");

    let state_size = input.state_in.len();
    let x_size = input.x.len();
    let total_rows = input.state_stride as usize + input.suffix_len as usize;
    let padded_size = total_rows * input.row_stride as usize;

    let state_array = context.create_array_from(&[state_size], &input.state_in, "state_in");
    let x_array = context.create_array_from(&[x_size], &input.x, "x");
    let padded_array = context.create_array_uninitialized(&[padded_size], T::data_type(), "padded");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        state_array.buffer().borrow().deref(),
        x_array.buffer().borrow().deref(),
        padded_array.buffer().borrow_mut().deref_mut(),
        input.state_stride,
        input.row_stride,
        input.suffix_len,
        input.num_channels,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    padded_array.as_slice().to_vec()
}

fn get_test_data<T: ArrayElement + Float>(
    state_stride: u32,
    row_stride: u32,
    suffix_len: u32,
    num_channels: u32,
) -> (Input<T>, Vec<T>) {
    let input = get_input::<T>(state_stride, row_stride, suffix_len, num_channels);
    let expected = get_output::<Cpu, T>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3f32
    } else {
        1e-6
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let msg = format!("Conv1dPack {} {} (type={})", std::any::type_name::<B>(), label, std::any::type_name::<T>(),);
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    // 4 channels, state_stride=3 (kernel_size=4), 2 tokens
    let (input, expected) = get_test_data::<T>(3, 4, 2, 4);
    test_internal(&input, &expected, "basic");
}

fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    // 8 channels, state_stride=1, 1 token
    let (input, expected) = get_test_data::<T>(1, 8, 1, 8);
    test_internal(&input, &expected, "single_token");
}

fn test_many_tokens<T: ArrayElement + Float + Debug + Display>() {
    // Minimal state (kernel_size=2), many tokens
    let (input, expected) = get_test_data::<T>(1, 4, 8, 4);
    test_internal(&input, &expected, "many_tokens");
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    // Larger dimensions closer to real usage
    let (input, expected) = get_test_data::<T>(3, 128, 16, 128);
    test_internal(&input, &expected, "large");
}

fn test_channels_lt_row_stride<T: ArrayElement + Float + Debug + Display>() {
    // num_channels < row_stride (padded channels that kernel doesn't write)
    let (input, expected) = get_test_data::<T>(2, 8, 3, 6);
    test_internal(&input, &expected, "channels_lt_row_stride");
}

// f32
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_single_token_f32() {
    test_single_token::<f32>();
}

#[uzu_test]
fn test_many_tokens_f32() {
    test_many_tokens::<f32>();
}

#[uzu_test]
fn test_large_f32() {
    test_large::<f32>();
}

#[uzu_test]
fn test_channels_lt_row_stride_f32() {
    test_channels_lt_row_stride::<f32>();
}

// f16
#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_single_token_f16() {
    test_single_token::<f16>();
}

#[uzu_test]
fn test_many_tokens_f16() {
    test_many_tokens::<f16>();
}

#[uzu_test]
fn test_large_f16() {
    test_large::<f16>();
}

// bf16
#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[uzu_test]
fn test_single_token_bf16() {
    test_single_token::<bf16>();
}

#[uzu_test]
fn test_many_tokens_bf16() {
    test_many_tokens::<bf16>();
}

#[uzu_test]
fn test_large_bf16() {
    test_large::<bf16>();
}
