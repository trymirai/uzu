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
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

fn run_router_top_k<B: Backend>(
    logits: &[f32],
    biases: Option<&[f32]>,
    num_experts: u32,
    k: u32,
    batch_dim: u32,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let logits_buf = create_buffer_with_data::<B, f32>(&context, logits);
    let biases_buf = biases.map(|b| create_buffer_with_data::<B, f32>(&context, b));
    let mut expert_ids = create_buffer::<B, u32>(&context, (batch_dim * k) as usize);
    let mut expert_weights = create_buffer::<B, f32>(&context, (batch_dim * k) as usize);

    assert!(k <= 8, "k must be <= 8 for MoeRouterTopK");
    let max_top_k = 8;
    let kernel =
        <B::Kernels as Kernels>::MoeRouterTopKKernel::new(&context, DataType::F32, max_top_k, biases.is_some())
            .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &logits_buf,
        biases_buf.as_ref(),
        &mut expert_ids,
        &mut expert_weights,
        num_experts,
        k,
        batch_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (buffer_to_vec::<B, _>(&context, &expert_ids), buffer_to_vec::<B, _>(&context, &expert_weights))
}

#[uzu_test]
fn moe_router_top_k_matches_cpu() {
    for (batch_dim, num_experts, k, has_biases) in [
        (1u32, 256u32, 8u32, false),
        (2u32, 256u32, 8u32, true),
        (16u32, 256u32, 8u32, false),
        (5u32, 64u32, 4u32, true),
        (1u32, 512u32, 1u32, false),
        (2u32, 512u32, 8u32, true),
    ] {
        let mut logits: Vec<f32> =
            (0..(batch_dim * num_experts) as usize).map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.125).collect();
        // Inject an exact tie to verify deterministic lowest-index tie-breaking
        if num_experts >= 16 {
            logits[3] = 10.0;
            logits[11] = 10.0;
        }
        let biases: Option<Vec<f32>> =
            has_biases.then(|| (0..num_experts as usize).map(|i| (((i * 19 + 7) % 31) as f32 - 15.0) * 0.05).collect());

        let expected = run_router_top_k::<Cpu>(&logits, biases.as_deref(), num_experts, k, batch_dim);
        for row in expected.1.chunks_exact(k as usize) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax weights should sum to 1.0, got {sum}");
        }

        for_each_non_cpu_backend!(|B| {
            let actual = run_router_top_k::<B>(&logits, biases.as_deref(), num_experts, k, batch_dim);
            assert_eq!(
                actual.0, expected.0,
                "expert_ids mismatch for shape=({batch_dim}, {num_experts}, {k}, has_biases={has_biases})"
            );
            for (idx, (&a, &e)) in actual.1.iter().zip(expected.1.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-5,
                    "expert_weights[{idx}] mismatch: actual={a}, expected={e} for shape=({batch_dim}, {num_experts}, {k})"
                );
            }
        });
    }
}

#[uzu_test]
fn moe_router_top_k_nan_safety() {
    let num_experts = 64u32;
    let k = 8u32;
    let batch_dim = 2u32;
    let mut logits = vec![0.0f32; (batch_dim * num_experts) as usize];
    for i in 0..num_experts as usize {
        logits[i] = (i as f32) * 0.1;
    }
    for i in (0..num_experts as usize).step_by(3) {
        logits[i] = f32::NAN;
    }
    for i in (num_experts as usize)..(2 * num_experts as usize) {
        logits[i] = f32::NAN;
    }

    let expected = run_router_top_k::<Cpu>(&logits, None, num_experts, k, batch_dim);
    for &id in &expected.0 {
        assert!(id < num_experts, "CPU produced out-of-range expert id {id} >= {num_experts}");
    }
    for_each_non_cpu_backend!(|B| {
        let actual = run_router_top_k::<B>(&logits, None, num_experts, k, batch_dim);
        for &id in &actual.0 {
            assert!(id < num_experts, "Backend produced out-of-range expert id {id} >= {num_experts}");
        }
        assert_eq!(&actual.0[..k as usize], &expected.0[..k as usize], "expert_ids mismatch under mixed NaN inputs");
    });
}

#[uzu_test]
fn moe_router_top_k_jax_ties() {
    let logits: Vec<f32> = vec![1.0, 3.0, 3.0, 2.0, 3.0];
    let num_experts = 5u32;
    let k = 2u32;
    let batch_dim = 1u32;
    let expected = run_router_top_k::<Cpu>(&logits, None, num_experts, k, batch_dim);
    assert_eq!(expected.0, vec![1, 2], "Tie breaking expects lowest indices first: [1, 2]");
    for_each_non_cpu_backend!(|B| {
        let actual = run_router_top_k::<B>(&logits, None, num_experts, k, batch_dim);
        assert_eq!(actual.0, vec![1, 2], "Backend tie breaking must match CPU and JAX: [1, 2]");
    });
}
