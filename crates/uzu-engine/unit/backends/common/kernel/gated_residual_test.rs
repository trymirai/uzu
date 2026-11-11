use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        kernel::{
            GatedResidualExpandKernel, GatedResidualLowRankActivationKernel, GatedResidualMixKernel,
            GatedResidualUpdateKernel,
        },
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
    },
};

const ROWS: usize = 3;
const STREAM_COUNT: usize = 4;
const MODEL_DIM: usize = 40;
const LOW_RANK_DIM: usize = 24;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn values<T: DataTypeRepr + Float>(
    count: usize,
    seed: usize,
) -> Vec<T> {
    (0..count).map(|index| T::from(((index * 37 + seed * 11) % 97) as f32 * 0.04 - 1.9).unwrap()).collect()
}

fn to_f32<T: DataTypeRepr + Float>(values: &[T]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32().unwrap()).collect()
}

fn eps<T: DataTypeRepr>() -> f32 {
    if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        2e-2
    } else {
        1e-5
    }
}

fn run<B: Backend>(
    context: &B::Context,
    encode: impl FnOnce(&mut <B::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding),
) {
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    encode(&mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");
}

fn test_expand<T: DataTypeRepr + Float + Debug + Display>() {
    let input = values::<T>(ROWS * MODEL_DIM, 0);
    let expected: Vec<T> = (0..ROWS * STREAM_COUNT * MODEL_DIM)
        .map(|position| input[(position / (STREAM_COUNT * MODEL_DIM)) * MODEL_DIM + position % MODEL_DIM])
        .collect();

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::GatedResidualExpandKernel::new(&context, T::data_type())
            .expect("Failed to create GatedResidualExpandKernel");
        let input_buffer = create_buffer_with_data::<B, T>(&context, &input);
        let mut output_buffer = create_buffer::<B, T>(&context, expected.len());
        run::<B>(&context, |command_buffer| {
            kernel.encode(
                &input_buffer,
                &mut output_buffer,
                MODEL_DIM as u32,
                STREAM_COUNT as u32,
                expected.len() as u32,
                command_buffer,
            )
        });
        let actual = buffer_to_vec::<B, T>(&context, &output_buffer);
        let message = format!("GatedResidualExpand failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, 0.0, &message);
    });
}

fn test_low_rank_activation<T: DataTypeRepr + Float + Debug + Display>() {
    let input = values::<T>(ROWS * LOW_RANK_DIM, 6);
    let expected: Vec<T> = to_f32(&input)
        .into_iter()
        .map(|value| {
            let x = value / STREAM_COUNT as f32;
            T::from(x * sigmoid(x)).unwrap()
        })
        .collect();

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel =
            <<B as Backend>::Kernels as Kernels>::GatedResidualLowRankActivationKernel::new(&context, T::data_type())
                .expect("Failed to create GatedResidualLowRankActivationKernel");
        let mut values_buffer = create_buffer_with_data::<B, T>(&context, &input);
        run::<B>(&context, |command_buffer| {
            kernel.encode(&mut values_buffer, STREAM_COUNT as u32, input.len() as u32, command_buffer)
        });
        let actual = buffer_to_vec::<B, T>(&context, &values_buffer);
        let message = format!("GatedResidualLowRankActivation failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps::<T>(), &message);
    });
}

fn test_mix<T: DataTypeRepr + Float + Debug + Display>() {
    let streams = values::<T>(ROWS * STREAM_COUNT * MODEL_DIM, 1);
    let logits = values::<T>(ROWS * STREAM_COUNT * MODEL_DIM, 2);
    let (streams_f32, logits_f32) = (to_f32(&streams), to_f32(&logits));
    let expected: Vec<T> = (0..ROWS * MODEL_DIM)
        .map(|position| {
            let (row, channel) = (position / MODEL_DIM, position % MODEL_DIM);
            let sum: f32 = (0..STREAM_COUNT)
                .map(|stream| {
                    let index = (row * STREAM_COUNT + stream) * MODEL_DIM + channel;
                    sigmoid(logits_f32[index]) * streams_f32[index]
                })
                .sum();
            T::from(sum / STREAM_COUNT as f32).unwrap()
        })
        .collect();

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::GatedResidualMixKernel::new(&context, T::data_type())
            .expect("Failed to create GatedResidualMixKernel");
        let streams_buffer = create_buffer_with_data::<B, T>(&context, &streams);
        let logits_buffer = create_buffer_with_data::<B, T>(&context, &logits);
        let mut output_buffer = create_buffer::<B, T>(&context, expected.len());
        run::<B>(&context, |command_buffer| {
            kernel.encode(
                &streams_buffer,
                &logits_buffer,
                &mut output_buffer,
                MODEL_DIM as u32,
                STREAM_COUNT as u32,
                expected.len() as u32,
                command_buffer,
            )
        });
        let actual = buffer_to_vec::<B, T>(&context, &output_buffer);
        let message = format!("GatedResidualMix failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps::<T>(), &message);
    });
}

fn test_update<T: DataTypeRepr + Float + Debug + Display>() {
    let streams = values::<T>(ROWS * STREAM_COUNT * MODEL_DIM, 3);
    let injection_logits = values::<T>(ROWS * STREAM_COUNT, 4);
    let update = values::<T>(ROWS * MODEL_DIM, 5);
    let (streams_f32, injection_f32, update_f32) = (to_f32(&streams), to_f32(&injection_logits), to_f32(&update));
    let expected: Vec<T> = (0..ROWS * STREAM_COUNT * MODEL_DIM)
        .map(|position| {
            let row_stream = position / MODEL_DIM;
            let (row, channel) = (row_stream / STREAM_COUNT, position % MODEL_DIM);
            let weight = 2.0 * sigmoid(injection_f32[row_stream] / STREAM_COUNT as f32);
            T::from(streams_f32[position] + weight * update_f32[row * MODEL_DIM + channel]).unwrap()
        })
        .collect();

    for_each_backend!(|B| {
        let context = block_on(<B as Backend>::Context::new()).expect("Failed to create Context");
        let kernel = <<B as Backend>::Kernels as Kernels>::GatedResidualUpdateKernel::new(&context, T::data_type())
            .expect("Failed to create GatedResidualUpdateKernel");
        let streams_buffer = create_buffer_with_data::<B, T>(&context, &streams);
        let injection_buffer = create_buffer_with_data::<B, T>(&context, &injection_logits);
        let update_buffer = create_buffer_with_data::<B, T>(&context, &update);
        let mut output_buffer = create_buffer::<B, T>(&context, expected.len());
        run::<B>(&context, |command_buffer| {
            kernel.encode(
                &streams_buffer,
                &injection_buffer,
                &update_buffer,
                &mut output_buffer,
                MODEL_DIM as u32,
                STREAM_COUNT as u32,
                expected.len() as u32,
                command_buffer,
            )
        });
        let actual = buffer_to_vec::<B, T>(&context, &output_buffer);
        let message = format!("GatedResidualUpdate failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps::<T>(), &message);
    });
}

#[uzu_test]
fn test_gated_residual_expand_f32() {
    test_expand::<f32>();
}

#[uzu_test]
fn test_gated_residual_expand_bf16() {
    test_expand::<bf16>();
}

#[uzu_test]
fn test_gated_residual_low_rank_activation_f32() {
    test_low_rank_activation::<f32>();
}

#[uzu_test]
fn test_gated_residual_low_rank_activation_bf16() {
    test_low_rank_activation::<bf16>();
}

#[uzu_test]
fn test_gated_residual_mix_f32() {
    test_mix::<f32>();
}

#[uzu_test]
fn test_gated_residual_mix_bf16() {
    test_mix::<bf16>();
}

#[uzu_test]
fn test_gated_residual_update_f32() {
    test_update::<f32>();
}

#[uzu_test]
fn test_gated_residual_update_bf16() {
    test_update::<bf16>();
}
