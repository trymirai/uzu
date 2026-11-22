use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{DepthAttentionKernel, MessageGateApplyKernel, MessageGateDownKernel},
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_parity,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const MODEL_DIM: u32 = 288;
const EPSILON: f32 = 1e-6;

fn value(index: usize, salt: f32) -> f32 {
    (index as f32 * 0.379 + salt).sin() * (1.0 + salt.cos())
}

fn run_depth_attention<B: Backend>(
    query_count: u32,
    source_slots: &[u32],
) -> Vec<bf16> {
    const SLOTS: u32 = 6;
    const STREAM_TOKENS: u32 = 7;
    const TOKEN_OFFSET: u32 = 2;
    const TOKENS: u32 = 4;
    let context = block_on(B::Context::new()).expect("create context");
    let kernel = <B::Kernels as Kernels>::DepthAttentionKernel::new(&context, bf16::data_type(), MODEL_DIM, query_count)
        .expect("create depth attention kernel");
    let stream = (0..(SLOTS * STREAM_TOKENS * MODEL_DIM) as usize)
        .map(|i| bf16::from_f32(value(i, (i / (STREAM_TOKENS * MODEL_DIM) as usize) as f32)))
        .collect::<Vec<_>>();
    let queries = (0..(query_count * MODEL_DIM) as usize).map(|i| 3.0 * value(i, 0.7)).collect::<Vec<f32>>();
    let stream = create_buffer_with_data::<B, bf16>(&context, &stream);
    let slots = create_buffer_with_data::<B, u32>(&context, source_slots);
    let queries = create_buffer_with_data::<B, f32>(&context, &queries);
    let mut output = create_buffer::<B, bf16>(&context, (query_count * TOKENS * MODEL_DIM) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &stream,
        &slots,
        &queries,
        &mut output,
        TOKENS,
        STREAM_TOKENS,
        TOKEN_OFFSET,
        source_slots.len() as u32,
        EPSILON,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run depth attention");
    buffer_to_vec::<B, bf16>(&context, &output)
}

#[uzu_test]
fn test_depth_attention_matches_cpu() {
    for query_count in [1, 3, 4] {
        for source_slots in [&[5][..], &[0, 5][..], &[3, 1, 4, 0, 2, 5][..]] {
            let expected = run_depth_attention::<Cpu>(query_count, source_slots);
            for_each_non_cpu_backend!(|B| {
                let actual = run_depth_attention::<B>(query_count, source_slots);
                assert_parity(
                    &format!("depth attention queries={query_count} sources={source_slots:?}"),
                    &expected,
                    &actual,
                    1e-2,
                    1e-2,
                );
            });
        }
    }
}

fn run_message_gates<B: Backend>(shared_input: bool) -> (Vec<f32>, Vec<bf16>) {
    const TOKENS: u32 = 3;
    const RANK: u32 = 64;
    const SLOTS: u32 = 3;
    let context = block_on(B::Context::new()).expect("create context");
    let down_kernel =
        <B::Kernels as Kernels>::MessageGateDownKernel::new(&context, bf16::data_type(), MODEL_DIM, RANK, SLOTS)
            .expect("create message gate down kernel");
    let apply_kernel =
        <B::Kernels as Kernels>::MessageGateApplyKernel::new(&context, bf16::data_type(), MODEL_DIM, RANK, SLOTS)
            .expect("create message gate kernel");
    let size = (TOKENS * MODEL_DIM) as usize;
    let input_slots = if shared_input {
        1
    } else {
        SLOTS as usize
    };
    let inputs = (0..input_slots * size).map(|i| bf16::from_f32(value(i, 0.1))).collect::<Vec<_>>();
    let messages = (0..SLOTS as usize * size).map(|i| bf16::from_f32(2.0 * value(i, 2.3))).collect::<Vec<_>>();
    let down = (0..(SLOTS * RANK * 2 * MODEL_DIM) as usize)
        .map(|i| bf16::from_f32(0.05 * value(i, 1.7)))
        .collect::<Vec<_>>();
    let forget = (0..(SLOTS * MODEL_DIM * RANK) as usize).map(|i| 0.1 * value(i, 0.4)).collect::<Vec<f32>>();
    let gain = (0..(SLOTS * MODEL_DIM * RANK) as usize).map(|i| 0.1 * value(i, 3.1)).collect::<Vec<f32>>();
    let inputs = create_buffer_with_data::<B, bf16>(&context, &inputs);
    let messages = create_buffer_with_data::<B, bf16>(&context, &messages);
    let down = create_buffer_with_data::<B, bf16>(&context, &down);
    let forget = create_buffer_with_data::<B, f32>(&context, &forget);
    let gain = create_buffer_with_data::<B, f32>(&context, &gain);
    let mut low_rank = create_buffer::<B, f32>(&context, (SLOTS * TOKENS * RANK) as usize);
    let mut output = create_buffer::<B, bf16>(&context, SLOTS as usize * size);
    let input_slot_stride = if shared_input {
        0
    } else {
        size as u32
    };
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    down_kernel.encode(&inputs, &messages, &down, &mut low_rank, TOKENS, input_slot_stride, EPSILON, &mut command_buffer);
    apply_kernel.encode(
        &inputs,
        &messages,
        &low_rank,
        &forget,
        &gain,
        &mut output,
        TOKENS,
        input_slot_stride,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run message gates");
    (buffer_to_vec::<B, f32>(&context, &low_rank), buffer_to_vec::<B, bf16>(&context, &output))
}

#[uzu_test]
fn test_message_gate_matches_cpu() {
    for shared_input in [false, true] {
        let (expected_low_rank, expected_output) = run_message_gates::<Cpu>(shared_input);
        for_each_non_cpu_backend!(|B| {
            let (low_rank, output) = run_message_gates::<B>(shared_input);
            assert_parity(&format!("message gate low rank shared={shared_input}"), &expected_low_rank, &low_rank, 1e-3, 1e-3);
            assert_parity(&format!("message gate output shared={shared_input}"), &expected_output, &output, 1e-2, 1e-2);
        });
    }
}
