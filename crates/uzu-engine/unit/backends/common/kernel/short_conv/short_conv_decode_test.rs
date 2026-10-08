use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::ShortConvDecodeKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

struct Input<T: ArrayElement + Float> {
    in_proj: Box<[T]>,
    w: Box<[f32]>,
    b: Option<Box<[f32]>>,
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
        DataType::F32,
        has_bias,
        state_in_place,
    )
    .expect("Failed to create ShortConvDecodeKernel");

    let in_proj = create_buffer_with_data::<B, T>(&context, &input.in_proj);
    let w = create_buffer_with_data::<B, f32>(&context, &input.w);
    let b = input.b.as_ref().map(|b| create_buffer_with_data::<B, f32>(&context, b));

    let out_size = input.suffix_len as usize * input.model_dim as usize;
    let mut out = create_buffer::<B, T>(&context, out_size);

    let state_size = input.model_dim as usize * input.state_stride as usize;
    let state_buffer_size = state_size.max(1);
    let state_data: Vec<T> =
        input.state.iter().copied().chain(std::iter::repeat(T::zero())).take(state_buffer_size).collect();

    let mut next_state = if state_in_place {
        create_buffer_with_data::<B, T>(&context, &state_data)
    } else {
        create_buffer::<B, T>(&context, state_buffer_size)
    };

    let state = (!state_in_place).then(|| create_buffer_with_data::<B, T>(&context, &state_data));

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &in_proj,
        &w,
        b.as_ref(),
        state.as_ref(),
        &mut out,
        &mut next_state,
        input.suffix_len,
        input.kernel_size,
        input.in_proj_stride,
        input.state_stride,
        input.model_dim,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    (buffer_to_vec(&out), buffer_to_vec(&next_state))
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
    let mut w = vec![0.0f32; model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.1 * (tap as f32) - 0.01 * (ch as f32) + 0.5;
            w[ch * kernel_size + tap] = val;
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
        let mut bias = vec![0.0f32; model_dim];
        for ch in 0..model_dim {
            bias[ch] = 0.01 * (ch as f32) + 0.1;
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

    let mut w = vec![0.0f32; model_dim * kernel_size];
    for ch in 0..model_dim {
        for tap in 0..kernel_size {
            let val = 0.25 * (tap as f32) + 0.1;
            w[ch * kernel_size + tap] = val;
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
        let mut bias = vec![0.0f32; model_dim];
        for ch in 0..model_dim {
            bias[ch] = 0.005 * (ch as f32) + 0.01;
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

/// Multi-token decode runs tokens in order, each after the first continuing the state the previous one stored, in place
/// or not. The CPU state follows the independent trajectory of shifts and T(pre_gate · input) bit for bit, and so does
/// every backend's. Outputs of two backends differ by at most twice the FP32 error γ(max(K + 2, 4)) of the absolute
/// terms (the longest weighted path: pre_gate · input, its weight, the last accumulation and the gate) plus both
/// roundings to T. Repeated to expose races between channels or tokens.
fn test_sequential_tokens<T: ArrayElement + Float + Debug + Display>() {
    let (model_dim, kernel_size, in_proj_stride) = (300, 4, 905);
    let tap_count = kernel_size - 1;
    let value = |i: usize| {
        let magnitude = 0.125 + ((i * 7919) % 1021) as f32 / 1020.0 * 1.875;
        if (i * 7919 / 1021).is_multiple_of(3) {
            -magnitude
        } else {
            magnitude
        }
    };
    let n = (kernel_size + 2).max(4) as f64 * f64::from(f32::EPSILON) / 2.0;
    let gamma = n / (1.0 - n);
    let unit = match T::data_type() {
        DataType::F32 => 0.0,
        _ => T::epsilon().to_f64().unwrap() / 2.0,
    };
    let tiny = unit * T::min_positive_value().to_f64().unwrap();
    let f64_of = |value: T| value.to_f64().unwrap();
    let bits = |values: &[T]| bytemuck::cast_slice::<T, u8>(values).to_vec();
    for (suffix_len, has_bias) in [(3, false), (17, true)] {
        let input = Input::<T> {
            in_proj: (0..suffix_len * in_proj_stride).map(|i| T::from(value(i)).unwrap()).collect(),
            w: (0..model_dim * kernel_size).map(|i| value(i + 3)).collect(),
            b: has_bias.then(|| (0..model_dim).map(|i| value(i + 7)).collect()),
            state: (0..model_dim * tap_count).map(|i| T::from(value(i + 11)).unwrap()).collect(),
            suffix_len: suffix_len as u32,
            kernel_size: kernel_size as u32,
            in_proj_stride: in_proj_stride as u32,
            state_stride: tap_count as u32,
            model_dim: model_dim as u32,
        };
        let mut state = input.state.to_vec();
        let mut bounds = vec![0.0; suffix_len * model_dim];
        for token in 0..suffix_len {
            for channel in 0..model_dim {
                let row = token * in_proj_stride + channel;
                let x = f64_of(input.in_proj[row]) * f64_of(input.in_proj[row + 2 * model_dim]);
                let samples = state[channel * tap_count..(channel + 1) * tap_count].iter().map(|&s| f64_of(s));
                let weights = &input.w[channel * kernel_size..(channel + 1) * kernel_size];
                let bias = input.b.as_ref().map_or(0.0, |b| f64::from(b[channel]));
                let magnitude =
                    samples.chain([x]).zip(weights).map(|(s, &w)| (s * f64::from(w)).abs()).sum::<f64>() + bias.abs();
                let gated = magnitude * f64_of(input.in_proj[row + model_dim]).abs();
                let accumulation = gamma * gated;
                bounds[token * model_dim + channel] =
                    2.0 * accumulation + 2.0 * unit * (gated + accumulation) + 2.0 * tiny;
                state.copy_within(channel * tap_count + 1..(channel + 1) * tap_count, channel * tap_count);
                state[(channel + 1) * tap_count - 1] = T::from(x as f32).unwrap();
            }
        }
        for state_in_place in [false, true] {
            let case =
                format!("{:?} suffix_len={suffix_len} has_bias={has_bias} in_place={state_in_place}", T::data_type());
            let (cpu_out, cpu_state) = get_output::<T, Cpu>(&input, state_in_place);
            assert_eq!(bits(&cpu_state), bits(&state), "CPU state trajectory, {case}");
            for _ in 0..3 {
                for_each_non_cpu_backend!(|B| {
                    let (out, next_state) = get_output::<T, B>(&input, state_in_place);
                    let backend = std::any::type_name::<B>();
                    assert_eq!(bits(&next_state), bits(&cpu_state), "{backend} state, {case}");
                    for (index, ((&expected, &actual), &bound)) in cpu_out.iter().zip(&out).zip(&bounds).enumerate() {
                        let error = (f64_of(actual) - f64_of(expected)).abs();
                        assert!(
                            error <= bound,
                            "{backend} out {index}: CPU {expected}, {actual}, bound {bound:e}, {case}"
                        );
                    }
                });
            }
        }
    }
}

#[uzu_test]
fn test_sequential_tokens_all_types() {
    test_sequential_tokens::<f32>();
    test_sequential_tokens::<f16>();
    test_sequential_tokens::<bf16>();
}
