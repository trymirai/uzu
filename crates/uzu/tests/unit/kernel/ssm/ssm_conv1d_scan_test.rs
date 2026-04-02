use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::Conv1dScanKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    padded: Box<[T]>,
    w: Box<[T]>,
    b: Option<Box<[T]>>,
    suffix_len: u32,
    kernel_size: u32,
    row_stride: u32,
    state_stride: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    activation_type: ActivationType,
}

struct Output<T: ArrayElement + Float> {
    x_out: Vec<T>,
    b_out: Vec<T>,
    c_out: Vec<T>,
    state_out: Vec<T>,
}

fn get_input<T: ArrayElement + Float>(
    kernel_size: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    suffix_len: u32,
    has_bias: bool,
    activation_type: ActivationType,
) -> Input<T> {
    let row_stride = num_channels;
    let state_stride = kernel_size.saturating_sub(1);
    let total_rows = (suffix_len + state_stride) as usize;
    let padded_size = total_rows * row_stride as usize;
    let w_size = num_channels as usize * kernel_size as usize;

    let padded: Vec<T> = (0..padded_size).map(|i| T::from(0.1 * ((i as f64 % 11.0) - 5.0)).unwrap()).collect();
    let w: Vec<T> = (0..w_size).map(|i| T::from(0.05 * ((i as f64 % 5.0) - 2.0)).unwrap()).collect();
    let b = if has_bias {
        Some(
            (0..num_channels as usize)
                .map(|i| T::from(0.01 * (i as f64 + 1.0)).unwrap())
                .collect::<Vec<T>>()
                .into_boxed_slice(),
        )
    } else {
        None
    };

    Input {
        padded: padded.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        suffix_len,
        kernel_size,
        row_stride,
        state_stride,
        num_channels,
        inner_dim,
        proj_dim,
        activation_type,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let has_bias = input.b.is_some();
    let kernel = <<B as Backend>::Kernels as Kernels>::Conv1dScanKernel::new(&context, T::data_type(), has_bias)
        .expect("Failed to create Conv1dScanKernel");

    let x_out_size = input.suffix_len as usize * input.inner_dim as usize;
    let b_out_size = input.suffix_len as usize * input.proj_dim as usize;
    let c_out_size = input.suffix_len as usize * input.proj_dim as usize;
    let state_size = input.num_channels as usize * input.state_stride as usize;

    let padded_array = context.create_array_from(&[input.padded.len()], &input.padded, "padded");
    let w_array = context.create_array_from(&[input.w.len()], &input.w, "w");
    let b_array = input.b.as_ref().map(|b| context.create_array_from(&[b.len()], b, "b"));

    let x_out_array = context.create_array_uninitialized(&[x_out_size], T::data_type(), "x_out");
    let b_out_array = context.create_array_uninitialized(&[b_out_size], T::data_type(), "b_out");
    let c_out_array = context.create_array_uninitialized(&[c_out_size], T::data_type(), "c_out");
    let state_out_array = context.create_array_uninitialized(&[state_size], T::data_type(), "state_out");

    let b_buf = b_array.as_ref().map(|a| a.buffer());
    let b_borrow = b_buf.as_ref().map(|rc| rc.borrow());
    let b_deref: Option<&B::Buffer> = b_borrow.as_ref().map(|b| b.deref());

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        padded_array.buffer().borrow().deref(),
        w_array.buffer().borrow().deref(),
        b_deref,
        x_out_array.buffer().borrow_mut().deref_mut(),
        b_out_array.buffer().borrow_mut().deref_mut(),
        c_out_array.buffer().borrow_mut().deref_mut(),
        state_out_array.buffer().borrow_mut().deref_mut(),
        input.suffix_len,
        input.kernel_size,
        input.row_stride,
        input.state_stride,
        input.num_channels,
        input.inner_dim,
        input.proj_dim,
        input.activation_type,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        x_out: x_out_array.as_slice().to_vec(),
        b_out: b_out_array.as_slice().to_vec(),
        c_out: c_out_array.as_slice().to_vec(),
        state_out: state_out_array.as_slice().to_vec(),
    }
}

fn get_test_data<T: ArrayElement + Float>(
    kernel_size: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    suffix_len: u32,
    has_bias: bool,
    activation_type: ActivationType,
) -> (Input<T>, Output<T>) {
    let input = get_input::<T>(kernel_size, num_channels, inner_dim, proj_dim, suffix_len, has_bias, activation_type);
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
            &expected.x_out,
            &output.x_out,
            eps,
            &format!("Conv1dScan x_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.b_out,
            &output.b_out,
            eps,
            &format!("Conv1dScan b_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.c_out,
            &output.c_out,
            eps,
            &format!("Conv1dScan c_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.state_out,
            &output.state_out,
            eps,
            &format!("Conv1dScan state_out {backend_name} {label} (type={type_name})"),
        );
    });
}

// inner_dim=4, proj_dim=2 -> num_channels = 4 + 2*2 = 8, kernel_size=4, suffix_len=2
fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [true, false] {
        let label = format!("basic(bias={has_bias})");
        let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 2, has_bias, ActivationType::SILU);
        test_internal(&input, &expected, &label);
    }
}

// kernel_size=2 (state_stride=1, minimal state)
fn test_small_kernel<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(2, 6, 4, 1, 3, true, ActivationType::SILU);
    test_internal(&input, &expected, "small_kernel");
}

// Single token suffix
fn test_single_token<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 1, true, ActivationType::SILU);
    test_internal(&input, &expected, "single_token");
}

// Many tokens
fn test_many_tokens<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 16, true, ActivationType::SILU);
    test_internal(&input, &expected, "many_tokens");
}

// Larger dimensions closer to real usage
fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 128, 64, 32, 8, true, ActivationType::SILU);
    test_internal(&input, &expected, "large");
}

fn test_identity_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 2, false, ActivationType::IDENTITY);
    test_internal(&input, &expected, "identity_activation");
}

fn test_gelu_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 2, true, ActivationType::GELU);
    test_internal(&input, &expected, "gelu_activation");
}

fn test_tanh_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 2, true, ActivationType::TANH);
    test_internal(&input, &expected, "tanh_activation");
}

// f32
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_small_kernel_f32() {
    test_small_kernel::<f32>();
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
fn test_identity_activation_f32() {
    test_identity_activation::<f32>();
}

#[uzu_test]
fn test_gelu_activation_f32() {
    test_gelu_activation::<f32>();
}

#[uzu_test]
fn test_tanh_activation_f32() {
    test_tanh_activation::<f32>();
}

// f16
#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_small_kernel_f16() {
    test_small_kernel::<f16>();
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
fn test_small_kernel_bf16() {
    test_small_kernel::<bf16>();
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
