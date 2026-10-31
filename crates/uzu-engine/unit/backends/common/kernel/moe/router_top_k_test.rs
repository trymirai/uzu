use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::MoeRouterTopKKernel,
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

const MOE_ROUTER_MAX_TOP_K: u32 = 8;
// Qwen3.5-35B-A3B routes each token to 8 of 256 experts.
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;

fn run_router_top_k<B: Backend>(
    logits: &[f32],
    batch_dim: u32,
) -> Option<(Vec<u32>, Vec<f32>)> {
    let context = create_context::<B>();
    let logits_buf = create_buffer_with_data::<B, f32>(&context, logits);
    let mut expert_ids = create_buffer::<B, u32>(&context, (batch_dim * TOP_K) as usize);
    let mut expert_weights = create_buffer::<B, f32>(&context, (batch_dim * TOP_K) as usize);

    let kernel = supported_kernel::<B, _>(<B::Kernels as Kernels>::MoeRouterTopKKernel::new(
        &context,
        DataType::F32,
        MOE_ROUTER_MAX_TOP_K,
    ))?;
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
    Some((buffer_to_vec::<B, _>(&context, &expert_ids), buffer_to_vec::<B, _>(&context, &expert_weights)))
}

#[uzu_test]
fn moe_router_top_k_matches_cpu() {
    for batch_dim in [1u32, 2, 16] {
        let mut logits: Vec<f32> =
            (0..(batch_dim * NUM_EXPERTS) as usize).map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.125).collect();
        // Inject an exact tie to verify deterministic lowest-index tie-breaking.
        logits[3] = 10.0;
        logits[11] = 10.0;

        let expected = run_router_top_k::<Cpu>(&logits, batch_dim).unwrap();
        let sums: Vec<f32> =
            expected.1.as_chunks::<{ TOP_K as usize }>().0.iter().map(|row| row.iter().sum()).collect();
        assert_eq_float(&vec![1.0; sums.len()], &sums, 1e-5, "softmax weights should sum to 1.0");

        for_each_non_cpu_backend!(|B| {
            if let Some(actual) = run_router_top_k::<B>(&logits, batch_dim) {
                assert_eq!(actual.0, expected.0, "expert_ids mismatch for {batch_dim} tokens");
                let message = format!("expert_weights mismatch for {batch_dim} tokens");
                assert_eq_float(&expected.1, &actual.1, 1e-5, &message);
            }
        });
    }
}

#[uzu_test]
fn moe_router_top_k_nan_safety() {
    let batch_dim = 3u32;
    // Row 0: every third logit is NaN. Row 1: all NaN. Row 2: three finite logits, fewer than k.
    let mut logits = vec![f32::NAN; (batch_dim * NUM_EXPERTS) as usize];
    for i in (0..NUM_EXPERTS as usize).filter(|i| i % 3 != 0) {
        logits[i] = (i as f32) * 0.1;
    }
    for i in [5, 17, 40] {
        logits[2 * NUM_EXPERTS as usize + i] = (i as f32) * 0.1;
    }

    let expected = run_router_top_k::<Cpu>(&logits, batch_dim).unwrap();
    assert_eq!(
        expected.0[2 * TOP_K as usize..],
        [40, 17, 5, 0, 1, 2, 3, 4],
        "NaN logits rank below every finite logit"
    );
    for_each_non_cpu_backend!(|B| {
        if let Some(actual) = run_router_top_k::<B>(&logits, batch_dim) {
            assert_eq!(actual.0, expected.0, "expert_ids mismatch under NaN inputs");
            assert_eq_float(&expected.1, &actual.1, 1e-5, "expert_weights mismatch under NaN inputs");
        }
    });
}

#[uzu_test]
fn moe_router_top_k_jax_ties() {
    // As in jax.lax.top_k: a tie for the 8th pick goes to the lower index, and +0.0 ranks above -0.0.
    let top_seven = [200u32, 150, 100, 90, 60, 50, 40];
    for (logit_7, logit_230, eighth) in [(1.0f32, 1.0f32, 7u32), (-0.0, 0.0, 230)] {
        let mut logits = vec![-1.0f32; NUM_EXPERTS as usize];
        for (rank, &expert) in top_seven.iter().enumerate() {
            logits[expert as usize] = 10.0 - rank as f32;
        }
        logits[7] = logit_7;
        logits[230] = logit_230;
        let expected_ids: Vec<u32> = top_seven.into_iter().chain([eighth]).collect();
        assert_eq!(
            run_router_top_k::<Cpu>(&logits, 1).unwrap().0,
            expected_ids,
            "CPU picks with {logit_7} and {logit_230} tied"
        );
        for_each_non_cpu_backend!(|B| {
            if let Some((ids, _)) = run_router_top_k::<B>(&logits, 1) {
                assert_eq!(ids, expected_ids, "Backend picks with {logit_7} and {logit_230} tied");
            }
        });
    }
}
