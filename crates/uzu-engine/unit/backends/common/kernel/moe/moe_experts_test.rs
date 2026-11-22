use futures::executor::block_on;
use half::{bf16, f16};
use uzu_engine_macros::uzu_test;

use super::moe_fixtures::{
    HostExperts, HostRoutes, INKLING_ACTIVATION, INKLING_HIDDEN_DIM, INKLING_MODEL_DIM, INKLING_TOP_K, random_bf16,
    random_f16,
};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::moe::{MoeExpertFormat, nvfp4_repack_word, nvfp4_repack_word_fast, nvfp4_repacked_code},
            kernel::{
                MoeExpertDownBatchKernel, MoeExpertDownKernel, MoeExpertUpBatchKernel, MoeExpertUpKernel,
                MoeSumRoutesKernel,
            },
        },
        cpu::Cpu,
    },
    data_type::{DataType, F8E4M3},
    encodable_block::mlp::moe::weights::{make_scales_non_negative, negate_groups, pack_affine_parameters},
    tests::{
        assert::assert_parity,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

struct RouteBuffers<B: Backend> {
    route_weights: B::GlobalGpuBuffer,
    expert_routes: B::GlobalGpuBuffer,
    active_experts: B::GlobalGpuBuffer,
}

impl<B: Backend> RouteBuffers<B> {
    fn upload(
        routes: &HostRoutes,
        context: &B::Context,
    ) -> Self {
        Self {
            route_weights: create_buffer_with_data::<B, f32>(context, &routes.route_weights),
            expert_routes: create_buffer_with_data::<B, u32>(context, &routes.expert_routes),
            active_experts: create_buffer_with_data::<B, u32>(context, &routes.active_experts),
        }
    }
}

fn run_up<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    input: &[bf16],
    max_tokens: u32,
) -> Vec<f16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.cols, experts.rows / 2);
    let weights = experts.upload::<B>(&context);
    let tables = RouteBuffers::<B>::upload(routes, &context);
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    // Rows the kernel must not touch keep the fill.
    let mut hidden =
        create_buffer_with_data::<B, f16>(&context, &vec![f16::from_f32(-7.0); routes_count * hidden_dim as usize]);
    let activation = INKLING_ACTIVATION;
    let kernel = <B::Kernels as Kernels>::MoeExpertUpKernel::new(
        &context,
        DataType::BF16,
        max_tokens,
        experts.format,
        activation.up_soft_cap.is_some(),
        activation.gate_soft_cap.is_some(),
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &input,
        &weights.codes,
        &weights.scales,
        weights.global_scales.as_ref(),
        &tables.expert_routes,
        &tables.active_experts,
        &tables.route_weights,
        &mut hidden,
        routes.tokens,
        model_dim,
        hidden_dim,
        routes.top_k,
        activation.up_soft_cap.unwrap(),
        activation.gate_soft_cap.unwrap(),
        activation.alpha,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &hidden)
}

fn run_down<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    hidden: &[f16],
    max_tokens: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.rows, experts.cols);
    let weights = experts.upload::<B>(&context);
    let tables = RouteBuffers::<B>::upload(routes, &context);
    let hidden = create_buffer_with_data::<B, f16>(&context, hidden);
    let mut output = create_buffer::<B, bf16>(&context, (routes.tokens * model_dim) as usize);
    let kernel =
        <B::Kernels as Kernels>::MoeExpertDownKernel::new(&context, DataType::BF16, max_tokens, experts.format)
            .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &hidden,
        &weights.codes,
        &weights.scales,
        weights.global_scales.as_ref(),
        &tables.expert_routes,
        &tables.active_experts,
        &mut output,
        routes.tokens,
        model_dim,
        hidden_dim,
        routes.top_k,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output)
}

// (model_dim, hidden_dim, experts, tokens): Inkling's expert shapes over few experts, and a narrow block over
// Inkling's 128 experts so that 32 tokens spread over many of them.
const SHAPES: [(u32, u32, u32, &[u32]); 2] =
    [(INKLING_MODEL_DIM, INKLING_HIDDEN_DIM, 16, &[1, 2, 3, 8]), (512, 256, 128, &[16, 32])];

fn compare_up(format: MoeExpertFormat) {
    for (model_dim, hidden_dim, experts, token_counts) in SHAPES {
        let weights = HostExperts::random(format, experts, 2 * hidden_dim, model_dim, 21);
        for &tokens in token_counts {
            // Correlated picks give experts several tokens, which exercises the paired-token path.
            let routes = HostRoutes::random(tokens, experts, INKLING_TOP_K, 0.6, 31 + u64::from(tokens));
            let input = random_bf16((tokens * model_dim) as usize, 41 + u64::from(tokens));
            let expected = run_up::<Cpu>(&weights, &routes, &input, 32);
            for max_tokens in [1u32, 32].into_iter().filter(|&max_tokens| tokens <= max_tokens) {
                for_each_non_cpu_backend!(|B| {
                    assert_parity(
                        &format!(
                            "{format:?} up, {model_dim}x{hidden_dim}, {experts} experts, {tokens} tokens, up to {max_tokens}"
                        ),
                        &expected,
                        &run_up::<B>(&weights, &routes, &input, max_tokens),
                        4e-3,
                        4e-3,
                    );
                });
            }
        }
    }
}

fn compare_down(format: MoeExpertFormat) {
    for (model_dim, hidden_dim, experts, token_counts) in SHAPES {
        let weights = HostExperts::random(format, experts, model_dim, hidden_dim, 22);
        for &tokens in token_counts {
            let routes = HostRoutes::random(tokens, experts, INKLING_TOP_K, 0.6, 32 + u64::from(tokens));
            let hidden = random_f16((tokens * INKLING_TOP_K * hidden_dim) as usize, 2.0, 42 + u64::from(tokens));
            let expected = run_down::<Cpu>(&weights, &routes, &hidden, 32);
            for max_tokens in [1u32, 8, 32].into_iter().filter(|&max_tokens| tokens <= max_tokens) {
                for_each_non_cpu_backend!(|B| {
                    assert_parity(
                        &format!(
                            "{format:?} down, {model_dim}x{hidden_dim}, {experts} experts, {tokens} tokens, up to {max_tokens}"
                        ),
                        &expected,
                        &run_down::<B>(&weights, &routes, &hidden, max_tokens),
                        1.6e-2,
                        2e-2,
                    );
                });
            }
        }
    }
}

#[uzu_test]
fn moe_expert_up_nvfp4_matches_cpu() {
    compare_up(MoeExpertFormat::Nvfp4);
}

#[uzu_test]
fn moe_expert_up_affine_matches_cpu() {
    compare_up(MoeExpertFormat::AffineU4);
}

#[uzu_test]
fn moe_expert_down_nvfp4_matches_cpu() {
    compare_down(MoeExpertFormat::Nvfp4);
}

#[uzu_test]
fn moe_expert_down_affine_matches_cpu() {
    compare_down(MoeExpertFormat::AffineU4);
}

fn run_up_batch<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    input: &[bf16],
    pass: u32,
) -> Vec<f16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.cols, experts.rows / 2);
    let weights = experts.upload::<B>(&context);
    let tables = RouteBuffers::<B>::upload(routes, &context);
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut hidden =
        create_buffer_with_data::<B, f16>(&context, &vec![f16::from_f32(-7.0); routes_count * hidden_dim as usize]);
    let activation = INKLING_ACTIVATION;
    let kernel = <B::Kernels as Kernels>::MoeExpertUpBatchKernel::new(
        &context,
        DataType::BF16,
        pass,
        activation.up_soft_cap.is_some(),
        activation.gate_soft_cap.is_some(),
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &input,
        &weights.codes,
        &weights.scales,
        &tables.expert_routes,
        &tables.active_experts,
        &tables.route_weights,
        &mut hidden,
        routes.tokens,
        model_dim,
        hidden_dim,
        experts.experts,
        routes.top_k,
        activation.up_soft_cap.unwrap(),
        activation.gate_soft_cap.unwrap(),
        activation.alpha,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &hidden)
}

/// The batch down kernel's route rows, then their per-token sums.
fn run_down_batch<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    hidden: &[f16],
    pass: u32,
) -> (Vec<f32>, Vec<bf16>) {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.rows, experts.cols);
    let weights = experts.upload::<B>(&context);
    let tables = RouteBuffers::<B>::upload(routes, &context);
    let hidden = create_buffer_with_data::<B, f16>(&context, hidden);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut route_outputs = create_buffer_with_data::<B, f32>(&context, &vec![0.0; routes_count * model_dim as usize]);
    let mut output = create_buffer::<B, bf16>(&context, (routes.tokens * model_dim) as usize);
    let down = <B::Kernels as Kernels>::MoeExpertDownBatchKernel::new(&context, pass).unwrap();
    let sum = <B::Kernels as Kernels>::MoeSumRoutesKernel::new(&context, DataType::BF16).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    down.encode(
        &hidden,
        &weights.codes,
        &weights.scales,
        &tables.expert_routes,
        &tables.active_experts,
        &mut route_outputs,
        routes.tokens,
        model_dim,
        hidden_dim,
        experts.experts,
        routes.top_k,
        &mut command_buffer,
    );
    sum.encode(&route_outputs, &mut output, routes.tokens, model_dim, routes.top_k, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (buffer_to_vec::<B, _>(&context, &route_outputs), buffer_to_vec::<B, _>(&context, &output))
}

// The batch kernels stream each expert's rows once for up to `pass` routes. Overlapping picks over few experts give
// experts up to every token, so that full passes, the shorter last pass and single routes all run.
#[uzu_test]
fn moe_expert_batch_matches_cpu() {
    let format = MoeExpertFormat::AffineU4;
    for (model_dim, hidden_dim, experts) in [(INKLING_MODEL_DIM, INKLING_HIDDEN_DIM, 16u32), (512, 256, 128)] {
        let up = HostExperts::random(format, experts, 2 * hidden_dim, model_dim, 23);
        let down = HostExperts::random(format, experts, model_dim, hidden_dim, 24);
        for tokens in [2u32, 3, 5, 8] {
            let routes = HostRoutes::random(tokens, experts, INKLING_TOP_K, 0.9, 33 + u64::from(tokens));
            let input = random_bf16((tokens * model_dim) as usize, 43 + u64::from(tokens));
            let hidden = random_f16((tokens * INKLING_TOP_K * hidden_dim) as usize, 2.0, 53 + u64::from(tokens));
            let expected_hidden = run_up::<Cpu>(&up, &routes, &input, 32);
            let expected_output = run_down::<Cpu>(&down, &routes, &hidden, 32);
            let (expected_routes, _) = run_down_batch::<Cpu>(&down, &routes, &hidden, 4);
            for pass in [2u32, 4] {
                let label = format!("batch, {model_dim}x{hidden_dim}, {experts} experts, {tokens} tokens, pass {pass}");
                assert_eq!(expected_hidden, run_up_batch::<Cpu>(&up, &routes, &input, pass), "{label}: CPU up twins");
                for_each_non_cpu_backend!(|B| {
                    assert_parity(
                        &format!("{label}: up"),
                        &expected_hidden,
                        &run_up_batch::<B>(&up, &routes, &input, pass),
                        4e-3,
                        4e-3,
                    );
                    let (route_outputs, output) = run_down_batch::<B>(&down, &routes, &hidden, pass);
                    assert_parity(&format!("{label}: down routes"), &expected_routes, &route_outputs, 1e-2, 1e-2);
                    assert_parity(&format!("{label}: down sums"), &expected_output, &output, 1.6e-2, 2e-2);
                });
            }
        }
    }
}

// The MLX export stores output-input parameters group-major with rows padded to a multiple of four.
#[uzu_test]
fn moe_affine_parameters_repack_to_rows() {
    let (experts, rows, padded_rows, groups) = (2u32, 6u32, 8u32, 3u32);
    let value = |expert: u32, group: u32, row: u32, bias: bool| {
        bf16::from_f32(
            (expert * 1000 + group * 100 + row) as f32
                + if bias {
                    0.5
                } else {
                    0.0
                },
        )
    };
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for expert in 0..experts {
        for group in 0..groups {
            for row in 0..padded_rows {
                scales.push(value(expert, group, row, false));
                biases.push(value(expert, group, row, true));
            }
        }
    }
    let packed = pack_affine_parameters(&scales, &biases, experts, rows, padded_rows, groups);
    for expert in 0..experts {
        for row in 0..rows {
            for group in 0..groups {
                let word = packed[((expert * rows + row) * groups + group) as usize];
                assert_eq!(bf16::from_bits(word as u16), value(expert, group, row, false));
                assert_eq!(bf16::from_bits((word >> 16) as u16), value(expert, group, row, true));
            }
        }
    }
}

// Every code of every word survives the NVFP4 repacking, the loader's branch-free rewrite matches the reference, and
// the kernels' decode pairs code j with code j + 4.
#[uzu_test]
fn moe_nvfp4_repack_round_trips() {
    let mut rng = super::moe_fixtures::Rng::new(3);
    for _ in 0..4096 {
        let word = rng.next_u64() as u32;
        assert_eq!(nvfp4_repack_word_fast(word), nvfp4_repack_word(word), "{word:#x}");
    }
    for word in [0u32, 0xFFFF_FFFF, 0x7654_3210, 0xFEDC_BA98, 0x8F3A_C501, 0x1234_5678] {
        let packed = nvfp4_repack_word(word);
        for index in 0..8 {
            assert_eq!(
                u32::from(nvfp4_repacked_code(packed, index)),
                (word >> (4 * index)) & 0xF,
                "{word:#x}[{index}]"
            );
        }
        // The decode of pair 0 is a mask: magnitude bits [9, 12) and the sign of codes 0 and 4.
        let pair = packed & 0x8E00_8E00;
        for (lane, index) in [(0u32, 0u32), (1, 4)] {
            let bits = (pair >> (16 * lane)) & 0xFFFF;
            let code = (word >> (4 * index)) & 0xF;
            assert_eq!(bits, ((code & 7) << 9) | ((code >> 3) << 15), "{word:#x} pair 0, lane {lane}");
        }
    }
}

// Negative E4M3 group scales lose their sign at load and their codes flip sign instead, which keeps every weight.
#[uzu_test]
fn moe_nvfp4_negative_scales_move_into_codes() {
    const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    let mut rng = super::moe_fixtures::Rng::new(9);
    let groups = 12usize;
    let mut codes = rng.bytes(groups * 8);
    let mut scales: Vec<F8E4M3> = (0..groups)
        .map(|group| {
            F8E4M3(
                (0x30 + group as u8)
                    | if group % 3 == 1 {
                        0x80
                    } else {
                        0
                    },
            )
        })
        .collect();
    let weights = |codes: &[u8], scales: &[F8E4M3]| -> Vec<f32> {
        (0..groups * 16)
            .map(|index| {
                let code = (codes[index / 2] >> (4 * (index % 2))) & 0xF;
                E2M1[usize::from(code)] * scales[index / 16].decode()
            })
            .collect()
    };
    let expected = weights(&codes, &scales);
    let negated = make_scales_non_negative(&mut scales);
    assert_eq!(negated, (0..groups).filter(|group| group % 3 == 1).collect::<Vec<_>>());
    assert!(scales.iter().all(|scale| scale.0 & 0x80 == 0));
    negate_groups(&mut codes, &negated);
    assert_eq!(weights(&codes, &scales), expected);
}
