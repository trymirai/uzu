use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::MoeCombineKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend,
            supported_kernel,
        },
    },
};

const MOE_COMBINE_BLOCK_DIM: u32 = 256;
// Qwen3.5-35B-A3B: 8 routed experts plus the gated shared expert, model_dim 2048.
const MODEL_DIM: u32 = 2048;
const ACTIVE_EXPERTS: u32 = 8;

fn run_moe_combine<B: Backend>(
    routed_outputs: &[bf16],
    expert_weights: &[f32],
    shared_output: &[bf16],
    shared_gate: &[bf16],
    batch_dim: u32,
) -> Option<Vec<bf16>> {
    let context = create_context::<B>();
    let routed_buf = create_buffer_with_data::<B, bf16>(&context, routed_outputs);
    let weights_buf = create_buffer_with_data::<B, f32>(&context, expert_weights);
    let shared_out_buf = create_buffer_with_data::<B, bf16>(&context, shared_output);
    let shared_gate_buf = create_buffer_with_data::<B, bf16>(&context, shared_gate);
    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * MODEL_DIM) as usize);

    let kernel = supported_kernel::<B, _>(<B::Kernels as Kernels>::MoeCombineKernel::new(
        &context,
        DataType::BF16,
        MOE_COMBINE_BLOCK_DIM,
    ))?;
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &routed_buf,
        &weights_buf,
        &shared_out_buf,
        &shared_gate_buf,
        &mut output_buf,
        MODEL_DIM,
        ACTIVE_EXPERTS,
        batch_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    Some(buffer_to_vec::<B, _>(&context, &output_buf))
}

#[uzu_test]
fn moe_combine_matches_cpu() {
    for batch_dim in [1u32, 2, 16] {
        let routed_outputs: Vec<bf16> = (0..(batch_dim * ACTIVE_EXPERTS * MODEL_DIM) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 3) % 71) as f32 - 35.0) * 0.05))
            .collect();
        // Each pick gets a different weight, in a different order for each token, and a token's weights sum to 1
        // like the router's softmax.
        let expert_weights: Vec<f32> = (0..(batch_dim * ACTIVE_EXPERTS) as usize)
            .map(|i| ((i + i / ACTIVE_EXPERTS as usize) % ACTIVE_EXPERTS as usize + 1) as f32 / 36.0)
            .collect();
        let shared_output: Vec<bf16> = (0..(batch_dim * MODEL_DIM) as usize)
            .map(|i| bf16::from_f32((((i * 17 + 11) % 43) as f32 - 21.0) * 0.05))
            .collect();
        let shared_gate: Vec<bf16> = (0..batch_dim as usize).map(|i| bf16::from_f32((i as f32 - 1.5) * 0.7)).collect();

        let expected =
            run_moe_combine::<Cpu>(&routed_outputs, &expert_weights, &shared_output, &shared_gate, batch_dim).unwrap();
        for_each_non_cpu_backend!(|B| {
            if let Some(actual) =
                run_moe_combine::<B>(&routed_outputs, &expert_weights, &shared_output, &shared_gate, batch_dim)
            {
                assert_eq_float(&expected, &actual, 1e-2, &format!("MoeCombine ({batch_dim} tokens)"));
            }
        });
    }
}
