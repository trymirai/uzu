use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::ShortConvDecodeKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    in_proj: Box<[T]>,
    w: Box<[T]>,
    b: Option<Box<[T]>>,
    state: Box<[T]>,
    suffix_len: u32,
    kernel_size: u32,
    in_proj_stride: u32,
    state_stride: u32,
    model_dim: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    state_in_place: bool,
) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let has_bias = input.b.is_some();
    let kernel = <<B as Backend>::Kernels as Kernels>::ShortConvDecodeKernel::new(
        &context,
        T::data_type(),
        has_bias,
        state_in_place,
    )
    .expect("Failed to create ShortConvDecodeKernel");

    let in_proj_array = context.create_array_from(&[input.in_proj.len()], &input.in_proj, "");
    let w_array = context.create_array_from(&[input.w.len()], &input.w, "");
    let b_array = input.b.as_ref().map(|b| context.create_array_from(&[b.len()], b, ""));

    let out_size = input.suffix_len as usize * input.model_dim as usize;
    let out_array = context.create_array_uninitialized(&[out_size], T::data_type(), "");

    let state_size = input.model_dim as usize * input.state_stride as usize;
    let next_state_array = if state_in_place {
        if state_size > 0 {
            context.create_array_from(&[state_size], &input.state, "")
        } else {
            context.create_array_uninitialized(&[1], T::data_type(), "")
        }
    } else {
        context.create_array_uninitialized(&[state_size.max(1)], T::data_type(), "")
    };

    let state_array = if state_in_place {
        None
    } else if state_size > 0 {
        Some(context.create_array_from(&[state_size], &input.state, ""))
    } else {
        Some(context.create_array_uninitialized(&[1], T::data_type(), ""))
    };

    let bias_buf_rc = b_array.as_ref().map(|b| b.buffer());
    let bias_buf_borrow = bias_buf_rc.as_ref().map(|rc| rc.borrow());

    let state_buf_rc = state_array.as_ref().map(|s| s.buffer());
    let state_buf_borrow = state_buf_rc.as_ref().map(|rc| rc.borrow());

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        in_proj_array.buffer().borrow().deref(),
        w_array.buffer().borrow().deref(),
        bias_buf_borrow.as_deref(),
        state_buf_borrow.as_deref(),
        out_array.buffer().borrow_mut().deref_mut(),
        next_state_array.buffer().borrow_mut().deref_mut(),
        input.suffix_len,
        input.kernel_size,
        input.in_proj_stride,
        input.state_stride,
        input.model_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (out_array.as_slice().to_vec(), next_state_array.as_slice().to_vec())
}

fn get_test_data_basic<T: ArrayElement + Float>(
    model_dim: usize,
    kernel_size: usize,
    has_bias: bool,
) -> (Input<T>, Vec<T>, Vec<T>) {
    let state_stride = kernel_size.saturating_sub(1);
    let in_proj_stride = model_dim * 3;
    let suffix_len = 1;

    // in_proj[token * in_proj_stride + col]
    //   [0..model_dim)             = pre_conv_gate
    //   [model_dim..2*model_dim)   = post_conv_gate
    //   [2*model_dim..3*model_dim) = x_in
    let mut in_proj = vec![T::zero(); suffix_len * in_proj_stride];
    for ch in 0..model_dim {
        let pre_gate = 0.3 * (ch as f32) + 1.0;
        let post_gate = 0.01 * (ch as f32) + 1.0;
        let x_in = 0.05 * (ch as f32) + 0.7;
        in_proj[ch] = T::from(pre_gate).unwrap();
        in_proj[model_dim + ch] = T::from(post_gate).unwrap();
        in_proj[2 * model_dim + ch] = T::from(x_in).unwrap();
    }

    // w[channel * kernel_size + tap]
    let mut w = vec![T::zero(); model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.1 * (tap as f32) - 0.01 * (ch as f32) + 0.5;
            w[ch * kernel_size + tap] = T::from(val).unwrap();
        }
    }

    // state[channel * state_stride + tap]
    let mut state = vec![T::zero(); model_dim * state_stride];
    for ch in 0..model_dim {
        for tap in 0..state_stride {
            let val = 0.1 * (ch as f32) + 0.01 * (tap as f32) + 0.5;
            state[ch * state_stride + tap] = T::from(val).unwrap();
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
        in_proj: in_proj.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        state: state.into_boxed_slice(),
        suffix_len: suffix_len as u32,
        kernel_size: kernel_size as u32,
        in_proj_stride: in_proj_stride as u32,
        state_stride: state_stride as u32,
        model_dim: model_dim as u32,
    };

    let (out, next_state) = get_output::<T, Cpu>(&input, false);
    (input, out, next_state)
}

fn get_test_data_edge<T: ArrayElement + Float>(
    model_dim: usize,
    kernel_size: usize,
    has_bias: bool,
) -> (Input<T>, Vec<T>, Vec<T>) {
    let state_stride = kernel_size.saturating_sub(1);
    let in_proj_stride = model_dim * 3;
    let suffix_len = 1;

    let mut in_proj = vec![T::zero(); suffix_len * in_proj_stride];
    for ch in 0..model_dim {
        let pre_gate = 0.5 + 0.1 * (ch as f32);
        let post_gate = 0.5 + 0.1 * (ch as f32);
        let x_in = 1e-3 * (ch as f32) + 0.01;
        in_proj[ch] = T::from(pre_gate).unwrap();
        in_proj[model_dim + ch] = T::from(post_gate).unwrap();
        in_proj[2 * model_dim + ch] = T::from(x_in).unwrap();
    }

    let mut w = vec![T::zero(); model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.25 * (tap as f32) + 0.1;
            w[ch * kernel_size + tap] = T::from(val).unwrap();
        }
    }

    let mut state = vec![T::zero(); model_dim * state_stride];
    for ch in 0..model_dim {
        for tap in 0..state_stride {
            let val = 1e-3 * (ch as f32) + 1e-3 * (tap as f32) + 0.01;
            state[ch * state_stride + tap] = T::from(val).unwrap();
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
        in_proj: in_proj.into_boxed_slice(),
        w: w.into_boxed_slice(),
        b,
        state: state.into_boxed_slice(),
        suffix_len: suffix_len as u32,
        kernel_size: kernel_size as u32,
        in_proj_stride: in_proj_stride as u32,
        state_stride: state_stride as u32,
        model_dim: model_dim as u32,
    };

    let (out, next_state) = get_output::<T, Cpu>(&input, false);
    (input, out, next_state)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected_out: &[T],
    expected_state: &[T],
    state_in_place: bool,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-3
    };

    for_each_non_cpu_backend!(|B| {
        let (out, next_state) = get_output::<T, B>(input, state_in_place);
        let msg = format!(
            "ShortConvDecode out failed with backend={}, has_bias={}, state_in_place={}, kernel_size={}, model_dim={}",
            std::any::type_name::<B>(),
            input.b.is_some(),
            state_in_place,
            input.kernel_size,
            input.model_dim,
        );
        assert_eq_float::<T>(expected_out, &out, eps, &msg);

        let state_msg = format!(
            "ShortConvDecode state failed with backend={}, has_bias={}, state_in_place={}, kernel_size={}, model_dim={}",
            std::any::type_name::<B>(),
            input.b.is_some(),
            state_in_place,
            input.kernel_size,
            input.model_dim,
        );
        assert_eq_float::<T>(expected_state, &next_state, eps, &state_msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        for state_in_place in [false, true] {
            let (input, expected_out, expected_state) = get_test_data_basic::<T>(64, 4, has_bias);
            test_internal(&input, &expected_out, &expected_state, state_in_place);
        }
    }
}

fn test_large<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        for state_in_place in [false, true] {
            let (input, expected_out, expected_state) = get_test_data_basic::<T>(256, 4, has_bias);
            test_internal(&input, &expected_out, &expected_state, state_in_place);
        }
    }
}

fn test_edge_small<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        for state_in_place in [false, true] {
            let (input, expected_out, expected_state) = get_test_data_edge::<T>(4, 2, has_bias);
            test_internal(&input, &expected_out, &expected_state, state_in_place);
        }
    }
}

fn test_edge_kernel<T: ArrayElement + Float + Debug + Display>() {
    for has_bias in [false, true] {
        for state_in_place in [false, true] {
            let (input, expected_out, expected_state) = get_test_data_edge::<T>(4, 1, has_bias);
            test_internal(&input, &expected_out, &expected_state, state_in_place);
        }
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

// edge: small dimensions
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

// edge: kernel_size=1 (no state taps)
#[uzu_test]
fn test_edge_kernel_f32() {
    test_edge_kernel::<f32>();
}

#[uzu_test]
fn test_edge_kernel_f16() {
    test_edge_kernel::<f16>();
}

#[uzu_test]
fn test_edge_kernel_bf16() {
    test_edge_kernel::<bf16>();
}
