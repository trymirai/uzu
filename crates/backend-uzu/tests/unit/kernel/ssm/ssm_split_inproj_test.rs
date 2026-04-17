use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SplitInProjKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    input: Box<[T]>,
    z_bias: Box<[T]>,
    suffix_length: u32,
    total_dim: u32,
    conv_dim: u32,
    inner_dim: u32,
    num_heads: u32,
}

struct Output<T: ArrayElement + Float> {
    conv_out: Vec<T>,
    z_out: Vec<T>,
    dt_out: Vec<T>,
}

fn get_input<T: ArrayElement + Float>(
    suffix_length: u32,
    conv_dim: u32,
    inner_dim: u32,
    num_heads: u32,
) -> Input<T> {
    let total_dim = conv_dim + inner_dim + num_heads;
    let input_size = suffix_length as usize * total_dim as usize;

    let input: Vec<T> = (0..input_size).map(|i| T::from(0.1 * ((i as f64 % 13.0) - 6.0)).unwrap()).collect();
    let z_bias: Vec<T> = (0..inner_dim as usize).map(|i| T::from(0.05 * ((i as f64 % 7.0) - 3.0)).unwrap()).collect();

    Input {
        input: input.into_boxed_slice(),
        z_bias: z_bias.into_boxed_slice(),
        suffix_length,
        total_dim,
        conv_dim,
        inner_dim,
        num_heads,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SplitInProjKernel::new(&context, T::data_type())
        .expect("Failed to create SplitInProjKernel");

    let conv_out_size = input.suffix_length as usize * input.conv_dim as usize;
    let z_out_size = input.suffix_length as usize * input.inner_dim as usize;
    let dt_out_size = input.suffix_length as usize * input.num_heads as usize;

    let input_array = context.create_array_from(&[input.input.len()], &input.input, "input");
    let z_bias_array = context.create_array_from(&[input.z_bias.len()], &input.z_bias, "z_bias");

    let conv_out_array = context.create_array_uninitialized(&[conv_out_size], T::data_type(), "conv_out");
    let z_out_array = context.create_array_uninitialized(&[z_out_size], T::data_type(), "z_out");
    let dt_out_array = context.create_array_uninitialized(&[dt_out_size], T::data_type(), "dt_out");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_array.buffer().borrow().deref(),
        conv_out_array.buffer().borrow_mut().deref_mut(),
        z_out_array.buffer().borrow_mut().deref_mut(),
        dt_out_array.buffer().borrow_mut().deref_mut(),
        z_bias_array.buffer().borrow().deref(),
        input.suffix_length,
        input.total_dim,
        input.conv_dim,
        input.inner_dim,
        input.num_heads,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        conv_out: conv_out_array.as_slice().to_vec(),
        z_out: z_out_array.as_slice().to_vec(),
        dt_out: dt_out_array.as_slice().to_vec(),
    }
}

fn get_test_data<T: ArrayElement + Float>(
    suffix_length: u32,
    conv_dim: u32,
    inner_dim: u32,
    num_heads: u32,
) -> (Input<T>, Output<T>) {
    let input = get_input::<T>(suffix_length, conv_dim, inner_dim, num_heads);
    let expected = get_output::<Cpu, T>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &Output<T>,
    label: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let backend_name = std::any::type_name::<B>();
        let type_name = std::any::type_name::<T>();

        assert_eq_float::<T>(
            &expected.conv_out,
            &output.conv_out,
            eps,
            &format!("SplitInProj conv_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.z_out,
            &output.z_out,
            eps,
            &format!("SplitInProj z_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.dt_out,
            &output.dt_out,
            eps,
            &format!("SplitInProj dt_out {backend_name} {label} (type={type_name})"),
        );
    });
}

// conv_dim=8, inner_dim=4, num_heads=2, suffix_length=2
fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(2, 8, 4, 2);
    test_internal(&input, &expected, "basic");
}

// Single token
fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 8, 4, 2);
    test_internal(&input, &expected, "single_token");
}

// Many tokens
fn test_many_tokens<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(16, 8, 4, 2);
    test_internal(&input, &expected, "many_tokens");
}

// Larger dimensions closer to real usage
fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(8, 128, 64, 32);
    test_internal(&input, &expected, "large");
}

// Minimal dimensions
fn test_minimal<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(1, 1, 1, 1);
    test_internal(&input, &expected, "minimal");
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
fn test_minimal_f32() {
    test_minimal::<f32>();
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
