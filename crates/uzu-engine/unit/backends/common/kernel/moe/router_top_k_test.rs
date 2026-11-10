use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::moe, kernel::MoeRouterTopKKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;

fn run_router_top_k<B: Backend>(
    logits: &[f32],
    batch_dim: u32,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let logits_buf = create_buffer_with_data::<B, f32>(&context, logits);
    let mut expert_ids = create_buffer::<B, u32>(&context, (batch_dim * TOP_K) as usize);
    let mut expert_weights = create_buffer::<B, f32>(&context, (batch_dim * TOP_K) as usize);

    let kernel =
        <B::Kernels as Kernels>::MoeRouterTopKKernel::new(&context, DataType::F32, moe::ROUTER_MAX_TOP_K).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &logits_buf,
        &mut expert_ids,
        &mut expert_weights,
        NUM_EXPERTS,
        TOP_K,
        batch_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (buffer_to_vec::<B, _>(&context, &expert_ids), buffer_to_vec::<B, _>(&context, &expert_weights))
}

#[uzu_test]
fn moe_router_top_k_matches_cpu() {
    for batch_dim in [1u32, 2, 16] {
        let mut logits: Vec<f32> =
            (0..(batch_dim * NUM_EXPERTS) as usize).map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.125).collect();
        logits[3] = 10.0;
        logits[11] = 10.0;

        let expected = run_router_top_k::<Cpu>(&logits, batch_dim);

        for_each_non_cpu_backend!(|B| {
            let actual = run_router_top_k::<B>(&logits, batch_dim);
            assert_eq!(actual.0, expected.0, "expert_ids mismatch for {batch_dim} tokens");
            assert_eq_float(&expected.1, &actual.1, 1e-5, &format!("expert_weights mismatch for {batch_dim} tokens"));
        });
    }
}
