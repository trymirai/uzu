use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::Conv1dDecodeKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    w: Box<[T]>,
    b: Option<Box<[T]>>,
    state: Box<[T]>,
    kernel_size: u32,
    row_stride: u32,
    state_stride: u32,
    num_channels: u32,
    suffix_len: u32,
    inner_dim: u32,
    proj_dim: u32,
    activation_type: ActivationType,
    state_in_place: bool,
}

struct Output<T: ArrayElement + Float> {
    x_out: Vec<T>,
    b_out: Vec<T>,
    c_out: Vec<T>,
    next_state: Vec<T>,
}

fn get_input<T: ArrayElement + Float>(
    kernel_size: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    suffix_len: u32,
    has_bias: bool,
    state_in_place: bool,
    activation_type: ActivationType,
) -> Input<T> {
    let row_stride = num_channels;
    let state_stride = kernel_size.saturating_sub(1);
    let state_size = num_channels as usize * state_stride as usize;
    let x_size = suffix_len as usize * row_stride as usize;
    let w_size = num_channels as usize * kernel_size as usize;

    let x: Vec<T> = (0..x_size).map(|i| T::from(0.1 * ((i as f64 % 7.0) - 3.0)).unwrap()).collect();
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
    let state: Vec<T> = (0..state_size).map(|i| T::from(0.02 * ((i as f64 % 9.0) - 4.0)).unwrap()).collect();

    Input {
        x: x.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        state: state.into_boxed_slice(),
        kernel_size,
        row_stride,
        state_stride,
        num_channels,
        suffix_len,
        inner_dim,
        proj_dim,
        activation_type,
        state_in_place,
    }
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Output<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let has_bias = input.b.is_some();
    let kernel = <<B as Backend>::Kernels as Kernels>::Conv1dDecodeKernel::new(
        &context,
        T::data_type(),
        has_bias,
        input.state_in_place,
    )
    .expect("Failed to create Conv1dDecodeKernel");

    let x_out_size = input.suffix_len as usize * input.inner_dim as usize;
    let b_out_size = input.suffix_len as usize * input.proj_dim as usize;
    let c_out_size = input.suffix_len as usize * input.proj_dim as usize;
    let state_size = input.num_channels as usize * input.state_stride as usize;

    let x_array = context.create_array_from(&[input.x.len()], &input.x, "x");
    let w_array = context.create_array_from(&[input.w.len()], &input.w, "w");
    let b_array = input.b.as_ref().map(|b| context.create_array_from(&[b.len()], b, "b"));

    let x_out_array = context.create_array_uninitialized(&[x_out_size], T::data_type(), "x_out");
    let b_out_array = context.create_array_uninitialized(&[b_out_size], T::data_type(), "b_out");
    let c_out_array = context.create_array_uninitialized(&[c_out_size], T::data_type(), "c_out");

    let b_buf = b_array.as_ref().map(|a| a.buffer());
    let b_borrow = b_buf.as_ref().map(|rc| rc.borrow());
    let b_deref: Option<&B::Buffer> = b_borrow.as_ref().map(|b| b.deref());

    if input.state_in_place {
        let next_state_array = context.create_array_from(&[state_size], &input.state, "next_state");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel.encode(
            x_array.buffer().borrow().deref(),
            w_array.buffer().borrow().deref(),
            b_deref,
            None::<&B::Buffer>,
            x_out_array.buffer().borrow_mut().deref_mut(),
            b_out_array.buffer().borrow_mut().deref_mut(),
            c_out_array.buffer().borrow_mut().deref_mut(),
            next_state_array.buffer().borrow_mut().deref_mut(),
            input.kernel_size,
            input.row_stride,
            input.state_stride,
            input.num_channels,
            input.suffix_len,
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
            next_state: next_state_array.as_slice().to_vec(),
        }
    } else {
        let state_array = context.create_array_from(&[state_size], &input.state, "state");
        let next_state_array = context.create_array_uninitialized(&[state_size], T::data_type(), "next_state");

        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
        kernel.encode(
            x_array.buffer().borrow().deref(),
            w_array.buffer().borrow().deref(),
            b_deref,
            Some(state_array.buffer().borrow().deref()),
            x_out_array.buffer().borrow_mut().deref_mut(),
            b_out_array.buffer().borrow_mut().deref_mut(),
            c_out_array.buffer().borrow_mut().deref_mut(),
            next_state_array.buffer().borrow_mut().deref_mut(),
            input.kernel_size,
            input.row_stride,
            input.state_stride,
            input.num_channels,
            input.suffix_len,
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
            next_state: next_state_array.as_slice().to_vec(),
        }
    }
}

fn get_test_data<T: ArrayElement + Float>(
    kernel_size: u32,
    num_channels: u32,
    inner_dim: u32,
    proj_dim: u32,
    suffix_len: u32,
    has_bias: bool,
    state_in_place: bool,
    activation_type: ActivationType,
) -> (Input<T>, Output<T>) {
    let input = get_input::<T>(
        kernel_size,
        num_channels,
        inner_dim,
        proj_dim,
        suffix_len,
        has_bias,
        state_in_place,
        activation_type,
    );
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
            &format!("Conv1dDecode x_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.b_out,
            &output.b_out,
            eps,
            &format!("Conv1dDecode b_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.c_out,
            &output.c_out,
            eps,
            &format!("Conv1dDecode c_out {backend_name} {label} (type={type_name})"),
        );
        assert_eq_float::<T>(
            &expected.next_state,
            &output.next_state,
            eps,
            &format!("Conv1dDecode next_state {backend_name} {label} (type={type_name})"),
        );
    });
}

// inner_dim=4, proj_dim=2 -> num_channels = 4 + 2*2 = 8, kernel_size=4
fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    for state_in_place in [true, false] {
        for has_bias in [true, false] {
            let label = format!("basic(bias={has_bias},in_place={state_in_place})");
            let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 1, has_bias, state_in_place, ActivationType::SILU);
            test_internal(&input, &expected, &label);
        }
    }
}

// kernel_size=2 (state_stride=1, minimal state)
fn test_small_kernel<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(2, 6, 4, 1, 1, true, true, ActivationType::SILU);
    test_internal(&input, &expected, "small_kernel");
}

// Larger dimensions closer to real usage
fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 128, 64, 32, 1, true, true, ActivationType::SILU);
    test_internal(&input, &expected, "large");
}

fn test_identity_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 1, false, true, ActivationType::IDENTITY);
    test_internal(&input, &expected, "identity_activation");
}

fn test_gelu_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 1, true, true, ActivationType::GELU);
    test_internal(&input, &expected, "gelu_activation");
}

fn test_tanh_activation<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data::<T>(4, 8, 4, 2, 1, true, true, ActivationType::TANH);
    test_internal(&input, &expected, "tanh_activation");
}

// f32
#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[test]
fn test_small_kernel_f32() {
    test_small_kernel::<f32>();
}

#[test]
fn test_large_f32() {
    test_large::<f32>();
}

#[test]
fn test_identity_activation_f32() {
    test_identity_activation::<f32>();
}

#[test]
fn test_gelu_activation_f32() {
    test_gelu_activation::<f32>();
}

#[test]
fn test_tanh_activation_f32() {
    test_tanh_activation::<f32>();
}

// f16
#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_small_kernel_f16() {
    test_small_kernel::<f16>();
}

#[test]
fn test_large_f16() {
    test_large::<f16>();
}

// bf16
#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[test]
fn test_small_kernel_bf16() {
    test_small_kernel::<bf16>();
}

#[test]
fn test_large_bf16() {
    test_large::<bf16>();
}
