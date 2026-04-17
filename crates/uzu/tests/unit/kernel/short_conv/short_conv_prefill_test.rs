use std::{
    fmt::{Debug, Display},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::ShortConvPrefillKernel},
        cpu::Cpu,
    },
};

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    padded: Box<[T]>,
    in_proj: Box<[T]>,
    w: Box<[T]>,
    b: Option<Box<[T]>>,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let has_bias = input.b.is_some();
    let kernel = <<B as Backend>::Kernels as Kernels>::ShortConvPrefillKernel::new(&context, T::data_type(), has_bias)
        .expect("Failed to create ShortConvPrefillKernel");

    let padded_array = context.create_array_from(&[input.padded.len()], &input.padded, "");
    let in_proj_array = context.create_array_from(&[input.in_proj.len()], &input.in_proj, "");
    let w_array = context.create_array_from(&[input.w.len()], &input.w, "");
    let b_array = input.b.as_ref().map(|b| context.create_array_from(&[b.len()], b, ""));

    let out_size = input.suffix_len as usize * input.model_dim as usize;
    let mut out = context
        .create_array_uninitialized(&[out_size], T::data_type(), "")
        .into_allocation();

    let state_out_size = input.model_dim as usize * input.state_stride as usize;
    let mut state_out = context
        .create_array_uninitialized(&[state_out_size], T::data_type(), "")
        .into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        padded_array.allocation(),
        in_proj_array.allocation(),
        w_array.allocation(),
        b_array.as_ref().map(|bias| bias.allocation()),
        &mut out,
        &mut state_out,
        input.suffix_len,
        input.kernel_size,
        input.in_proj_stride,
        input.state_stride,
        input.model_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        crate::common::helpers::allocation_to_vec(&out),
        crate::common::helpers::allocation_to_vec(&state_out),
    )
}

fn get_test_data_basic<T: ArrayElement + Float>(
    model_dim: usize,
    suffix_len: usize,
    kernel_size: usize,
    has_bias: bool,
) -> (Input<T>, Vec<T>, Vec<T>) {
    let state_stride = kernel_size.saturating_sub(1);
    let in_proj_stride = model_dim * 3;
    let padded_rows = state_stride + suffix_len;

    // padded[row * model_dim + channel]
    let mut padded = vec![T::zero(); padded_rows * model_dim];
    for row in 0..padded_rows {
        for ch in 0..model_dim {
            let val = 0.01 * (ch as f32) + 0.01 * (row as f32) + 0.5;
            padded[row * model_dim + ch] = T::from(val).unwrap();
        }
    }

    // in_proj[token * in_proj_stride + col]
    // post_conv_gate at offset [model_dim..2*model_dim)
    let mut in_proj = vec![T::zero(); suffix_len * in_proj_stride];
    for token in 0..suffix_len {
        for ch in 0..model_dim {
            let post_gate = 0.01 * (ch as f32) - 0.02 * (token as f32) + 1.0;
            in_proj[token * in_proj_stride + model_dim + ch] = T::from(post_gate).unwrap();
        }
    }

    // w[channel * kernel_size + tap]
    let mut w = vec![T::zero(); model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.1 * (tap as f32) - 0.01 * (ch as f32) + 0.5;
            w[ch * kernel_size + tap] = T::from(val).unwrap();
        }
    }

    let b = if has_bias {
        let mut bias = vec![T::zero(); model_dim];
        for ch in 0..model_dim {
            bias[ch] = T::from(0.01 * (ch as f32) + 0.1).unwrap();
        }
        Some(bias.into_boxed_slice())
    } else {
        None
    };

    let input = Input {
        padded: padded.into_boxed_slice(),
        in_proj: in_proj.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        suffix_len: suffix_len as u32,
        kernel_size: kernel_size as u32,
        in_proj_stride: in_proj_stride as u32,
        state_stride: state_stride as u32,
        model_dim: model_dim as u32,
    };

    let (out, state_out) = get_output::<T, Cpu>(&input);
    (input, out, state_out)
}

fn get_test_data_edge<T: ArrayElement + Float>(
    model_dim: usize,
    suffix_len: usize,
    kernel_size: usize,
    has_bias: bool,
) -> (Input<T>, Vec<T>, Vec<T>) {
    let state_stride = kernel_size.saturating_sub(1);
    let in_proj_stride = model_dim * 3;
    let padded_rows = state_stride + suffix_len;

    let mut padded = vec![T::zero(); padded_rows * model_dim];
    for row in 0..padded_rows {
        for ch in 0..model_dim {
            let val = 1e-3 * (ch as f32) + 1e-3 * (row as f32) + 0.01;
            padded[row * model_dim + ch] = T::from(val).unwrap();
        }
    }

    let mut in_proj = vec![T::zero(); suffix_len * in_proj_stride];
    for token in 0..suffix_len {
        for ch in 0..model_dim {
            let post_gate = 0.5 + 0.1 * (ch as f32);
            in_proj[token * in_proj_stride + model_dim + ch] = T::from(post_gate).unwrap();
        }
    }

    let mut w = vec![T::zero(); model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.25 * (tap as f32) + 0.1;
            w[ch * kernel_size + tap] = T::from(val).unwrap();
        }
    }

    let b = if has_bias {
        let mut bias = vec![T::zero(); model_dim];
        for ch in 0..model_dim {
            bias[ch] = T::from(0.005 * (ch as f32) + 0.01).unwrap();
        }
        Some(bias.into_boxed_slice())
    } else {
        None
    };

    let input = Input {
        padded: padded.into_boxed_slice(),
        in_proj: in_proj.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        suffix_len: suffix_len as u32,
        kernel_size: kernel_size as u32,
        in_proj_stride: in_proj_stride as u32,
        state_stride: state_stride as u32,
        model_dim: model_dim as u32,
    };

    let (out, state_out) = get_output::<T, Cpu>(&input);
    (input, out, state_out)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected_out: &[T],
    expected_state: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-4
    };

    for_each_non_cpu_backend!(|B| {
        let (out, state_out) = get_output::<T, B>(input);
        let msg = format!(
            "ShortConvPrefill out failed with backend={}, has_bias={}, suffix_len={}, kernel_size={}, model_dim={}",
            std::any::type_name::<B>(),
            input.b.is_some(),
            input.suffix_len,
            input.kernel_size,
            input.model_dim,
        );
        assert_eq_float::<T>(expected_out, &out, eps, &msg);

        let state_msg = format!(
            "ShortConvPrefill state failed with backend={}, has_bias={}, suffix_len={}, kernel_size={}, model_dim={}",
            std::any::type_name::<B>(),
            input.b.is_some(),
            input.suffix_len,
            input.kernel_size,
            input.model_dim,
        );
        assert_eq_float::<T>(expected_state, &state_out, eps, &state_msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        let (input, expected_out, expected_state) = get_test_data_basic::<T>(64, 4, 4, has_bias);
        test_internal(&input, &expected_out, &expected_state);
    }
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        let (input, expected_out, expected_state) = get_test_data_basic::<T>(256, 16, 4, has_bias);
        test_internal(&input, &expected_out, &expected_state);
    }
}

fn test_edge_single_token<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        let (input, expected_out, expected_state) = get_test_data_edge::<T>(4, 1, 4, has_bias);
        test_internal(&input, &expected_out, &expected_state);
    }
}

fn test_edge_small<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        let (input, expected_out, expected_state) = get_test_data_edge::<T>(4, 2, 2, has_bias);
        test_internal(&input, &expected_out, &expected_state);
    }
}

// basic tests
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

// large tests
#[uzu_test]
fn test_large_f32() {
    test_large::<f32>();
}

#[uzu_test]
fn test_large_f16() {
    test_large::<f16>();
}

#[uzu_test]
fn test_large_bf16() {
    test_large::<bf16>();
}

// edge: single token
#[uzu_test]
fn test_edge_single_token_f32() {
    test_edge_single_token::<f32>();
}

#[uzu_test]
fn test_edge_single_token_f16() {
    test_edge_single_token::<f16>();
}

#[uzu_test]
fn test_edge_single_token_bf16() {
    test_edge_single_token::<bf16>();
}

// edge: small
#[uzu_test]
fn test_edge_small_f32() {
    test_edge_small::<f32>();
}

#[uzu_test]
fn test_edge_small_f16() {
    test_edge_small::<f16>();
}

#[uzu_test]
fn test_edge_small_bf16() {
    test_edge_small::<bf16>();
}
