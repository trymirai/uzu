use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{
    MOE_GEMV_OUTPUT_ROW_TILE, SharedGemv, assert_vec_close, combine_test::run_moe_combine, gemv_test::run_moe_gemv,
};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::MoeDownCombineKernel,
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

fn run_moe_down_combine<B: Backend, W: DataTypeRepr, Z: DataTypeRepr>(
    weights: &[W],
    scales: Option<&[bf16]>,
    zero_points: Option<&[Z]>,
    biases: Option<&[bf16]>,
    input: &[bf16],
    expert_ids: &[u32],
    expert_weights: &[f32],
    shared_gate: Option<&[f32]>,
    output_bias: Option<&[bf16]>,
    output_hadamard_factors: Option<&[i32]>,
    input_dim: u32,
    output_dim: u32,
    batch_dim: u32,
    active_experts_per_token: u32,
    gate_stride: u32,
    bits: u32,
    group_size: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    shared: Option<&SharedGemv<'_, W, Z>>,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let weights_buf = create_buffer_with_data::<B, W>(&context, weights);
    let scales_buf = scales.map(|s| create_buffer_with_data::<B, bf16>(&context, s));
    let zp_buf = zero_points.map(|z| create_buffer_with_data::<B, Z>(&context, z));
    let biases_buf = biases.map(|b| create_buffer_with_data::<B, bf16>(&context, b));
    let input_buf = create_buffer_with_data::<B, bf16>(&context, input);
    let expert_ids_buf = create_buffer_with_data::<B, u32>(&context, expert_ids);
    let expert_weights_buf = create_buffer_with_data::<B, f32>(&context, expert_weights);
    let shared_gate_buf = shared_gate.map(|g| {
        let gate: Vec<bf16> = g.iter().map(|&v| bf16::from_f32(v)).collect();
        create_buffer_with_data::<B, bf16>(&context, &gate)
    });
    let out_bias_buf = output_bias.map(|b| create_buffer_with_data::<B, bf16>(&context, b));
    let rht_buf = output_hadamard_factors.map(|f| create_buffer_with_data::<B, i32>(&context, f));

    let has_shared = shared.is_some();
    let has_gate = shared_gate.is_some();
    let shared_weights_buf = shared.as_ref().map(|s| create_buffer_with_data::<B, W>(&context, s.weights));
    let shared_scales_buf =
        shared.as_ref().and_then(|s| s.scales.map(|sc| create_buffer_with_data::<B, bf16>(&context, sc)));
    let shared_zp_buf =
        shared.as_ref().and_then(|s| s.zero_points.map(|z| create_buffer_with_data::<B, Z>(&context, z)));
    let shared_biases_buf =
        shared.as_ref().and_then(|s| s.biases.map(|b| create_buffer_with_data::<B, bf16>(&context, b)));
    let shared_out_bias_buf =
        shared.as_ref().and_then(|s| s.output_bias.map(|b| create_buffer_with_data::<B, bf16>(&context, b)));
    let shared_rht_buf =
        shared.as_ref().and_then(|s| s.hadamard.map(|f| create_buffer_with_data::<B, i32>(&context, f)));

    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * output_dim) as usize);

    let b_prologue = if bits == 0 {
        GemmBPrologueKind::FullPrecision
    } else if zero_points.is_some() {
        GemmBPrologueKind::ScaleZeroPointDequant
    } else if biases.is_some() {
        GemmBPrologueKind::ScaleBiasDequant
    } else {
        GemmBPrologueKind::ScaleSymmetricDequant
    };

    let mut output_transform = GemmDTransform::empty();
    if output_bias.is_some() {
        output_transform |= GemmDTransform::BIAS;
    }
    if output_hadamard_factors.is_some() {
        output_transform |= GemmDTransform::RHT;
    }

    let full_tile = output_dim.is_multiple_of(MOE_GEMV_OUTPUT_ROW_TILE);
    let block_size = if bits == 0 {
        128
    } else if bits == 4 {
        512
    } else {
        256
    };
    let input_aligned = input_dim.is_multiple_of(block_size);

    let kernel = <B::Kernels as Kernels>::MoeDownCombineKernel::new(
        &context,
        DataType::BF16,
        group_size,
        bits,
        b_prologue,
        output_transform,
        input_aligned,
        false,
        full_tile,
        has_shared,
        has_gate,
    )
    .unwrap();

    let group_count_x = output_dim.div_ceil(MOE_GEMV_OUTPUT_ROW_TILE);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &weights_buf,
        scales_buf.as_ref(),
        zp_buf.as_ref(),
        biases_buf.as_ref(),
        shared_weights_buf.as_ref(),
        shared_scales_buf.as_ref(),
        shared_zp_buf.as_ref(),
        shared_biases_buf.as_ref(),
        &input_buf,
        &expert_ids_buf,
        &expert_weights_buf,
        shared_gate_buf.as_ref(),
        &mut output_buf,
        out_bias_buf.as_ref(),
        rht_buf.as_ref(),
        shared_out_bias_buf.as_ref(),
        shared_rht_buf.as_ref(),
        input_dim,
        output_dim,
        active_experts_per_token,
        group_count_x,
        b_expert_stride_bytes,
        scale_expert_stride,
        zero_point_expert_stride,
        scale_output_stride,
        scale_group_stride,
        zero_point_output_stride,
        zero_point_group_stride,
        batch_dim,
        gate_stride,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output_buf)
}

#[uzu_test]
fn moe_down_combine_matches_unfused() {
    let (input_dim, output_dim, active) = (256u32, 128u32, 4u32);
    let num_experts = 8u32;

    for (batch_dim, has_shared, has_gate) in
        [(1u32, false, false), (2u32, false, false), (2u32, true, false), (4u32, true, true)]
    {
        let gate_stride = 1u32;
        let num_routed_routes = batch_dim * active;
        let num_routes = num_routed_routes
            + if has_shared {
                batch_dim
            } else {
                0
            };

        let weights: Vec<bf16> = (0..(num_experts * output_dim * input_dim) as usize)
            .map(|i| bf16::from_f32((((i * 13 + 7) % 37) as f32 - 18.0) * 0.02))
            .collect();
        let input: Vec<bf16> = (0..(num_routes * input_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 5) % 41) as f32 - 20.0) * 0.03))
            .collect();
        let expert_ids: Vec<u32> = (0..num_routed_routes as usize).map(|i| (i as u32 * 3 + 1) % num_experts).collect();
        let expert_weights: Vec<f32> = (0..num_routed_routes as usize).map(|_| 1.0 / (active as f32)).collect();

        let shared_weights: Vec<bf16> = (0..(output_dim * input_dim) as usize)
            .map(|i| bf16::from_f32((((i * 19 + 3) % 29) as f32 - 14.0) * 0.02))
            .collect();
        let shared: Option<SharedGemv<'_, bf16, u8>> = has_shared.then(|| SharedGemv {
            weights: &shared_weights,
            scales: None,
            zero_points: None,
            biases: None,
            output_bias: None,
            hadamard: None,
            num_routed_routes,
        });
        let shared_gate: Option<Vec<f32>> =
            has_gate.then(|| (0..batch_dim as usize).map(|i| (i as f32 - 0.5) * 0.4).collect());

        let b_stride = output_dim * input_dim * 2;

        let unfused_gemv = run_moe_gemv::<Cpu, bf16, u8>(
            &weights,
            None,
            None,
            None,
            &input,
            &expert_ids,
            None,
            None,
            input_dim,
            output_dim,
            num_routes,
            active,
            0,
            0,
            b_stride,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            shared.as_ref(),
        );

        let shared_output = has_shared.then(|| {
            let start = (num_routed_routes * output_dim) as usize;
            unfused_gemv[start..].to_vec()
        });

        let expected = run_moe_combine::<Cpu>(
            &unfused_gemv[..(num_routed_routes * output_dim) as usize],
            &expert_weights,
            shared_output.as_deref(),
            shared_gate.as_deref(),
            output_dim,
            active,
            batch_dim,
            gate_stride,
        );

        let cpu_down_combine = run_moe_down_combine::<Cpu, bf16, u8>(
            &weights,
            None,
            None,
            None,
            &input,
            &expert_ids,
            &expert_weights,
            shared_gate.as_deref(),
            None,
            None,
            input_dim,
            output_dim,
            batch_dim,
            active,
            gate_stride,
            0,
            0,
            b_stride,
            0,
            0,
            0,
            0,
            0,
            0,
            shared.as_ref(),
        );
        assert_vec_close(
            &cpu_down_combine,
            &expected,
            2e-2,
            &format!("Cpu MoeDownCombine vs Unfused (batch={batch_dim}, shared={has_shared}, gate={has_gate})"),
        );

        for_each_non_cpu_backend!(|B| {
            let actual = run_moe_down_combine::<B, bf16, u8>(
                &weights,
                None,
                None,
                None,
                &input,
                &expert_ids,
                &expert_weights,
                shared_gate.as_deref(),
                None,
                None,
                input_dim,
                output_dim,
                batch_dim,
                active,
                gate_stride,
                0,
                0,
                b_stride,
                0,
                0,
                0,
                0,
                0,
                0,
                shared.as_ref(),
            );
            assert_vec_close(
                &actual,
                &expected,
                2e-2,
                &format!("Metal MoeDownCombine vs Unfused (batch={batch_dim}, shared={has_shared}, gate={has_gate})"),
            );
        });
    }
}
