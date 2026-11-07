use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::MoeGemvKernel,
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

const MOE_GEMV_OUTPUT_ROW_TILE: u32 = 32;
const GROUP_SIZE: u32 = 64;
const NUM_EXPERTS: u32 = 16;
const ACTIVE_EXPERTS: u32 = 8;
// Qwen3.5-35B-A3B expert projections: up (gate and up together) reads one input row per token and down reads
// one row per route. Every projection rotates its output.
const PROJECTIONS: [(&str, u32, u32, bool); 2] = [("up", 2048, 1024, true), ("down", 512, 2048, false)];

struct MoeGemvCase<'a> {
    bits: u32,
    weights: &'a [u8],
    scales: &'a [bf16],
    zero_points: Option<&'a [u8]>,
    hadamard_factors: &'a [i32],
    input: &'a [bf16],
    expert_ids: &'a [u32],
    in_dim: u32,
    out_dim: u32,
    a_per_token: bool,
}

fn run_moe_gemv<B: Backend>(case: &MoeGemvCase) -> Option<Vec<bf16>> {
    let (in_dim, out_dim) = (case.in_dim, case.out_dim);
    let num_routes = case.expert_ids.len() as u32;
    let num_groups = in_dim / GROUP_SIZE;
    let context = create_context::<B>();
    let weights = create_buffer_with_data::<B, u8>(&context, case.weights);
    let scales = create_buffer_with_data::<B, bf16>(&context, case.scales);
    let zero_points = case.zero_points.map(|z| create_buffer_with_data::<B, u8>(&context, z));
    let input = create_buffer_with_data::<B, bf16>(&context, case.input);
    let expert_ids = create_buffer_with_data::<B, u32>(&context, case.expert_ids);
    let hadamard_factors = create_buffer_with_data::<B, i32>(&context, case.hadamard_factors);
    let mut output = create_buffer::<B, bf16>(&context, (num_routes * out_dim) as usize);

    let kernel = supported_kernel::<B, _>(<B::Kernels as Kernels>::MoeGemvKernel::new(
        &context,
        DataType::BF16,
        GROUP_SIZE,
        case.bits,
        case.a_per_token,
    ))?;
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &weights,
        &scales,
        zero_points.as_ref(),
        &input,
        &expert_ids,
        &mut output,
        &hadamard_factors,
        in_dim,
        out_dim,
        num_routes,
        ACTIVE_EXPERTS,
        out_dim / MOE_GEMV_OUTPUT_ROW_TILE,
        out_dim * in_dim * case.bits / 8,
        num_groups * out_dim,
        num_groups * (out_dim / 2),
        1,
        out_dim,
        1,
        out_dim,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    Some(buffer_to_vec::<B, _>(&context, &output))
}

// Runs each projection at 1, 2 and 16 tokens, with the CPU twin as the reference. The weight scales keep outputs
// near 1, as in a real model, so 2e-2 is at least two bf16 steps. Every pattern has a prime period, so each expert's
// weights, zero points and rotation signs differ.
fn compare_projections(
    bits: u32,
    weights_for: impl Fn(u32, u32) -> (Vec<u8>, Vec<bf16>, Option<Vec<u8>>),
) {
    for (name, in_dim, out_dim, a_per_token) in PROJECTIONS {
        let (weights, scales, zero_points) = weights_for(in_dim, out_dim);
        let hadamard_factors: Vec<i32> = (0..(NUM_EXPERTS * out_dim) as usize)
            .map(|i| {
                if (i * 7 + 3) % 17 < 8 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        for tokens in [1u32, 2, 16] {
            let num_routes = tokens * ACTIVE_EXPERTS;
            let mut expert_ids: Vec<u32> =
                (0..num_routes as usize).map(|i| ((i * 5 + 3) as u32) % NUM_EXPERTS).collect();
            expert_ids[0] = NUM_EXPERTS - 1;
            let input_rows = if a_per_token {
                tokens
            } else {
                num_routes
            };
            let input: Vec<bf16> = (0..(input_rows * in_dim) as usize)
                .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
                .collect();
            let case = MoeGemvCase {
                bits,
                weights: &weights,
                scales: &scales,
                zero_points: zero_points.as_deref(),
                hadamard_factors: &hadamard_factors,
                input: &input,
                expert_ids: &expert_ids,
                in_dim,
                out_dim,
                a_per_token,
            };
            let expected = run_moe_gemv::<Cpu>(&case).unwrap();
            for_each_non_cpu_backend!(|B| {
                if let Some(actual) = run_moe_gemv::<B>(&case) {
                    assert_eq_float(&expected, &actual, 2e-2, &format!("{bits} bit {name}, {tokens} tokens"));
                }
            });
        }
    }
}

#[uzu_test]
fn moe_gemv_u4_zero_point_matches_cpu() {
    // Qwen3.5-35B-A3B-M: 4-bit weights with zero points, groups of 64.
    compare_projections(4, |in_dim, out_dim| {
        let num_groups = in_dim / GROUP_SIZE;
        let weights =
            (0..(NUM_EXPERTS * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) % 251) as u8).collect();
        let scales = (0..(NUM_EXPERTS * num_groups * out_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.00015))
            .collect();
        let zero_points = (0..(NUM_EXPERTS * num_groups * (out_dim / 2)) as usize)
            .map(|i| (((i * 43 + 11) % 13) | (((i * 61 + 3) % 11) << 4)) as u8)
            .collect();
        (weights, scales, Some(zero_points))
    });
}

#[uzu_test]
fn moe_gemv_u8_symmetric_matches_cpu() {
    // Qwen3.5-35B-A3B-L: 8-bit symmetric weights, groups of 64.
    compare_projections(8, |in_dim, out_dim| {
        let num_groups = in_dim / GROUP_SIZE;
        let weights = (0..(NUM_EXPERTS * out_dim * in_dim) as usize).map(|i| ((i * 59 + 11) % 251) as u8).collect();
        let scales = (0..(NUM_EXPERTS * num_groups * out_dim) as usize)
            .map(|i| bf16::from_f32((((i * 23 + 7) % 37) as f32 + 1.0) * 0.000012))
            .collect();
        (weights, scales, None)
    });
}
