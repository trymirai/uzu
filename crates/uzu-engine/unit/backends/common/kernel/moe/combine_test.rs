use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::assert_vec_close;
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::MoeCombineKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const MOE_COMBINE_BLOCK_DIM: u32 = 256;

fn run_moe_combine<B: Backend>(
    routed_outputs: &[bf16],
    expert_weights: &[f32],
    shared_output: Option<&[bf16]>,
    shared_gate: Option<&[f32]>,
    model_dim: u32,
    active_experts_per_token: u32,
    batch_dim: u32,
    gate_stride: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let routed_buf = create_buffer_with_data::<B, bf16>(&context, routed_outputs);
    let weights_buf = create_buffer_with_data::<B, f32>(&context, expert_weights);
    let shared_out_buf = shared_output.map(|s| create_buffer_with_data::<B, bf16>(&context, s));
    let shared_gate_buf = shared_gate.map(|g| create_buffer_with_data::<B, f32>(&context, g));
    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * model_dim) as usize);

    let kernel = <B::Kernels as Kernels>::MoeCombineKernel::new(
        &context,
        DataType::BF16,
        MOE_COMBINE_BLOCK_DIM,
        shared_output.is_some(),
        shared_gate.is_some(),
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &routed_buf,
        &weights_buf,
        shared_out_buf.as_ref(),
        shared_gate_buf.as_ref(),
        &mut output_buf,
        model_dim,
        active_experts_per_token,
        batch_dim,
        gate_stride,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output_buf)
}

#[uzu_test]
fn moe_combine_matches_cpu() {
    let (model_dim, active) = (128u32, 8u32);

    for (batch_dim, has_shared, has_gate, gate_stride) in [
        (1u32, false, false, 1u32),
        (2u32, true, false, 1u32),
        (4u32, true, true, 1u32),
        (16u32, true, true, 1u32),
        (2u32, true, true, 257u32),
    ] {
        let routed_outputs: Vec<bf16> = (0..(batch_dim * active * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 3) % 71) as f32 - 35.0) * 0.05))
            .collect();
        let expert_weights: Vec<f32> = (0..(batch_dim * active) as usize).map(|_| 1.0 / (active as f32)).collect();
        let shared_output: Option<Vec<bf16>> = has_shared.then(|| {
            (0..(batch_dim * model_dim) as usize)
                .map(|i| bf16::from_f32((((i * 17 + 11) % 43) as f32 - 21.0) * 0.05))
                .collect()
        });
        let shared_gate: Option<Vec<f32>> = has_gate.then(|| {
            let total_len = (batch_dim * gate_stride) as usize;
            let mut v = vec![0.0f32; total_len];
            for i in 0..batch_dim as usize {
                v[i * gate_stride as usize] = (i as f32 - 1.5) * 0.7;
            }
            v
        });

        let expected = run_moe_combine::<Cpu>(
            &routed_outputs,
            &expert_weights,
            shared_output.as_deref(),
            shared_gate.as_deref(),
            model_dim,
            active,
            batch_dim,
            gate_stride,
        );

        for_each_non_cpu_backend!(|B| {
            let actual = run_moe_combine::<B>(
                &routed_outputs,
                &expert_weights,
                shared_output.as_deref(),
                shared_gate.as_deref(),
                model_dim,
                active,
                batch_dim,
                gate_stride,
            );
            assert_vec_close(
                &actual,
                &expected,
                1e-2,
                &format!("MoeCombine (batch={batch_dim}, shared={has_shared}, gate={has_gate}, stride={gate_stride})"),
            );
        });
    }
}
