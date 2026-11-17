use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            kernel::{NGramConvInjectKernel, NGramGateKernel},
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend,
            submit_command_buffer,
        },
    },
};

fn values<T: DataTypeRepr + Float>(
    count: usize,
    seed: f32,
) -> Vec<T> {
    (0..count).map(|i| T::from(((i as f32 * 0.37 + seed).sin()) * 1.5).unwrap()).collect()
}

fn run_gate<T: DataTypeRepr + Float, B: Backend>(
    num_tokens: u32,
    stream_count: u32,
    model_dim: u32,
) -> (Vec<T>, Vec<T>) {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::NGramGateKernel::new(&context, T::data_type())
        .expect("Failed to create NGramGateKernel");
    let rows = (num_tokens * stream_count) as usize;
    let dim = model_dim as usize;

    let keys = create_buffer_with_data::<B, T>(&context, &values::<T>(rows * dim, 0.1));
    let stream = create_buffer_with_data::<B, T>(&context, &values::<T>(rows * dim, 1.3));
    let value_rows = create_buffer_with_data::<B, T>(&context, &values::<T>(num_tokens as usize * dim, 2.7));
    let scales = |seed| create_buffer_with_data::<B, f32>(&context, &values::<f32>(stream_count as usize * dim, seed));
    let (key_scales, query_scales, conv_scales) = (scales(0.5), scales(0.9), scales(1.7));
    let mut gated = create_buffer::<B, T>(&context, rows * dim);
    let mut conv_input = create_buffer::<B, T>(&context, rows * dim);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &keys,
        &stream,
        &value_rows,
        &key_scales,
        &query_scales,
        &conv_scales,
        &mut gated,
        &mut conv_input,
        num_tokens,
        stream_count,
        model_dim,
        1e-6,
        0.0,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<B, T>(&context, &gated), buffer_to_vec::<B, T>(&context, &conv_input))
}

fn run_conv<T: DataTypeRepr + Float, B: Backend>(
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
) -> (Vec<T>, Vec<T>) {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::NGramConvInjectKernel::new(&context, T::data_type())
        .expect("Failed to create NGramConvInjectKernel");
    let tokens = num_tokens as usize;
    let channels_count = channels as usize;
    let state_rows = kernel_size as usize - 1;

    let conv_input = create_buffer_with_data::<B, T>(&context, &values::<T>(tokens * channels_count, 0.2));
    let state = create_buffer_with_data::<B, T>(&context, &values::<T>(state_rows * channels_count, 3.1));
    let weights =
        create_buffer_with_data::<B, f32>(&context, &values::<f32>(channels_count * kernel_size as usize, 0.8));
    let gated = create_buffer_with_data::<B, T>(&context, &values::<T>(tokens * channels_count, 4.4));
    let mut stream = create_buffer_with_data::<B, T>(&context, &values::<T>(tokens * channels_count, 5.5));
    let mut next_state = create_buffer::<B, T>(&context, state_rows * channels_count);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &conv_input,
        &state,
        &weights,
        &gated,
        &mut stream,
        &mut next_state,
        num_tokens,
        channels,
        kernel_size,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<B, T>(&context, &stream), buffer_to_vec::<B, T>(&context, &next_state))
}

fn tolerance<T: DataTypeRepr>() -> f32 {
    match T::data_type() {
        DataType::F32 => 1e-4,
        _ => 0.05,
    }
}

fn test_ngram_gate<T: DataTypeRepr + Float + Debug + Display>() {
    for (num_tokens, stream_count, model_dim) in [(1, 4, 64), (5, 2, 1000)] {
        let (expected_gated, expected_conv_input) = run_gate::<T, Cpu>(num_tokens, stream_count, model_dim);
        for_each_non_cpu_backend!(|B| {
            let (gated, conv_input) = run_gate::<T, B>(num_tokens, stream_count, model_dim);
            let message = format!("NGramGate failed with backend={}", std::any::type_name::<B>());
            assert_eq_float::<T>(&expected_gated, &gated, tolerance::<T>(), &format!("{message} (gated)"));
            assert_eq_float::<T>(
                &expected_conv_input,
                &conv_input,
                tolerance::<T>(),
                &format!("{message} (conv input)"),
            );
        });
    }
}

fn test_ngram_conv_inject<T: DataTypeRepr + Float + Debug + Display>() {
    for (num_tokens, channels, kernel_size) in [(1, 256, 10), (3, 300, 10), (17, 64, 4)] {
        let (expected_stream, expected_state) = run_conv::<T, Cpu>(num_tokens, channels, kernel_size);
        for_each_non_cpu_backend!(|B| {
            let (stream, next_state) = run_conv::<T, B>(num_tokens, channels, kernel_size);
            let message = format!("NGramConvInject failed with backend={}", std::any::type_name::<B>());
            assert_eq_float::<T>(&expected_stream, &stream, tolerance::<T>(), &format!("{message} (stream)"));
            assert_eq_float::<T>(&expected_state, &next_state, tolerance::<T>(), &format!("{message} (state)"));
        });
    }
}

#[uzu_test]
fn test_ngram_gate_f32() {
    test_ngram_gate::<f32>();
}

#[uzu_test]
fn test_ngram_gate_bf16() {
    test_ngram_gate::<bf16>();
}

#[uzu_test]
fn test_ngram_conv_inject_f32() {
    test_ngram_conv_inject::<f32>();
}

#[uzu_test]
fn test_ngram_conv_inject_bf16() {
    test_ngram_conv_inject::<bf16>();
}
