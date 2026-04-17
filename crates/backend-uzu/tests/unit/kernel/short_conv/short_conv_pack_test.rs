use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::ShortConvPackKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    state_in: Box<[T]>,
    in_proj: Box<[T]>,
    state_stride: u32,
    suffix_len: u32,
    in_proj_stride: u32,
    model_dim: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::ShortConvPackKernel::new(&context, T::data_type())
        .expect("Failed to create ShortConvPackKernel");

    let state_in_array = context.create_array_from(&[input.state_in.len()], &input.state_in, "");
    let in_proj_array = context.create_array_from(&[input.in_proj.len()], &input.in_proj, "");

    let padded_rows = (input.state_stride + input.suffix_len) as usize;
    let padded_size = padded_rows * input.model_dim as usize;
    let padded_array = context.create_array_uninitialized(&[padded_size], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        state_in_array.buffer().borrow().deref(),
        in_proj_array.buffer().borrow().deref(),
        padded_array.buffer().borrow_mut().deref_mut(),
        input.state_stride,
        input.suffix_len,
        input.in_proj_stride,
        input.model_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    padded_array.as_slice().to_vec()
}

/// Build test input for ShortConvPack.
///
/// Layout:
///   state_in: [model_dim, state_stride] stored as state_in[channel * state_stride + row]
///   in_proj:  [suffix_len, in_proj_stride] where in_proj_stride = model_dim * 3
///     - columns [0..model_dim)          = pre_gate
///     - columns [2*model_dim..3*model_dim) = x_in
///   padded (output): [(state_stride + suffix_len), model_dim]
fn get_input<T: ArrayElement + Float>(
    model_dim: usize,
    state_stride: usize,
    suffix_len: usize,
) -> (Input<T>, Vec<T>) {
    let in_proj_stride = model_dim * 3;

    // state_in[channel_idx * state_stride + row_idx]
    let mut state_in = vec![T::zero(); model_dim * state_stride];
    for ch in 0..model_dim {
        for row in 0..state_stride {
            let val = 0.1 * (ch as f32) + 0.01 * (row as f32) + 0.5;
            state_in[ch * state_stride + row] = T::from(val).unwrap();
        }
    }

    // in_proj[token * in_proj_stride + col]
    let mut in_proj = vec![T::zero(); suffix_len * in_proj_stride];
    for token in 0..suffix_len {
        for ch in 0..model_dim {
            let pre_gate = 0.3 * (ch as f32) - 0.1 * (token as f32) + 1.0;
            let x_in = 0.2 * (token as f32) + 0.05 * (ch as f32) + 0.7;
            in_proj[token * in_proj_stride + ch] = T::from(pre_gate).unwrap();
            in_proj[token * in_proj_stride + 2 * model_dim + ch] = T::from(x_in).unwrap();
        }
    }

    let input = Input {
        state_in: state_in.into_boxed_slice(),
        in_proj: in_proj.into_boxed_slice(),
        state_stride: state_stride as u32,
        suffix_len: suffix_len as u32,
        in_proj_stride: in_proj_stride as u32,
        model_dim: model_dim as u32,
    };

    let output = get_output::<T, Cpu>(&input);
    (input, output)
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let (input, expected) = get_input::<T>(64, 3, 4);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("ShortConvPack basic test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let (input, expected) = get_input::<T>(256, 3, 16);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("ShortConvPack large test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_edge_single_token<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let (input, expected) = get_input::<T>(4, 3, 1);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("ShortConvPack single-token edge test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_edge_small<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let (input, expected) = get_input::<T>(4, 1, 2);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("ShortConvPack no-state edge test failed for backend {}", std::any::type_name::<B>()),
        );
    });
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

// edge tests: single token
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

// edge tests: no state (state_stride = 0)
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
