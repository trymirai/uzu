use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{SharedGemv, assert_vec_close};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::MoeGemvKernel,
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const MOE_GEMV_OUTPUT_ROW_TILE: u32 = 32;

fn run_moe_gemv<B: Backend, W: DataTypeRepr, Z: DataTypeRepr>(
    weights: &[W],
    scales: Option<&[bf16]>,
    zero_points: Option<&[Z]>,
    biases: Option<&[bf16]>,
    input: &[bf16],
    expert_ids: &[u32],
    output_bias: Option<&[bf16]>,
    output_hadamard_factors: Option<&[i32]>,
    input_dim: u32,
    output_dim: u32,
    num_routes: u32,
    active_experts_per_token: u32,
    bits: u32,
    group_size: u32,
    b_expert_stride_bytes: u32,
    scale_expert_stride: u32,
    zero_point_expert_stride: u32,
    scale_output_stride: u32,
    scale_group_stride: u32,
    zero_point_output_stride: u32,
    zero_point_group_stride: u32,
    a_per_token: bool,
    shared: Option<&SharedGemv<'_, W, Z>>,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let weights_buf = create_buffer_with_data::<B, W>(&context, weights);
    let scales_buf = scales.map(|s| create_buffer_with_data::<B, bf16>(&context, s));
    let zp_buf = zero_points.map(|z| create_buffer_with_data::<B, Z>(&context, z));
    let biases_buf = biases.map(|b| create_buffer_with_data::<B, bf16>(&context, b));
    let input_buf = create_buffer_with_data::<B, bf16>(&context, input);
    let expert_ids_buf = create_buffer_with_data::<B, u32>(&context, expert_ids);
    let out_bias_buf = output_bias.map(|b| create_buffer_with_data::<B, bf16>(&context, b));
    let rht_buf = output_hadamard_factors.map(|f| create_buffer_with_data::<B, i32>(&context, f));

    let has_shared = shared.is_some();
    let num_routed_routes = shared.as_ref().map_or(num_routes, |s| s.num_routed_routes);
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

    let mut output_buf = create_buffer::<B, bf16>(&context, (num_routes * output_dim) as usize);

    let b_prologue = if bits == 0 {
        GemmBPrologueKind::FullPrecision
    } else if zero_points.is_some() {
        GemmBPrologueKind::ScaleZeroPointDequant
    } else if biases.is_some() {
        GemmBPrologueKind::ScaleBiasDequant
    } else {
        GemmBPrologueKind::ScaleSymmetricDequant
    };
    let output_transform = if output_hadamard_factors.is_some() {
        GemmDTransform::RHT
    } else {
        GemmDTransform::empty()
    };

    let full_tile = output_dim.is_multiple_of(MOE_GEMV_OUTPUT_ROW_TILE);
    let input_aligned = input_dim.is_multiple_of(128);
    let kernel = <B::Kernels as Kernels>::MoeGemvKernel::new(
        &context,
        DataType::BF16,
        group_size,
        bits,
        b_prologue,
        output_transform,
        input_aligned,
        false,
        full_tile,
        a_per_token,
        has_shared,
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
        &mut output_buf,
        out_bias_buf.as_ref(),
        rht_buf.as_ref(),
        shared_out_bias_buf.as_ref(),
        shared_rht_buf.as_ref(),
        input_dim,
        output_dim,
        num_routes,
        active_experts_per_token,
        group_count_x,
        b_expert_stride_bytes,
        scale_expert_stride,
        zero_point_expert_stride,
        scale_output_stride,
        scale_group_stride,
        zero_point_output_stride,
        zero_point_group_stride,
        num_routed_routes,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<B, _>(&context, &output_buf)
}

fn compare_moe_gemv<W: DataTypeRepr, Z: DataTypeRepr>(
    weights: &[W],
    scales: Option<&[bf16]>,
    zero_points: Option<&[Z]>,
    biases: Option<&[bf16]>,
    input: &[bf16],
    expert_ids: &[u32],
    rht: Option<&[i32]>,
    in_dim: u32,
    out_dim: u32,
    routes: u32,
    active: u32,
    bits: u32,
    group_size: u32,
    a_per_token: bool,
    tol: f32,
    desc: &str,
    shared: Option<&SharedGemv<'_, W, Z>>,
) {
    let b_stride = out_dim
        * if bits == 0 {
            in_dim * 2
        } else {
            in_dim * bits / 8
        };
    let num_groups = in_dim.checked_div(group_size).unwrap_or(0);
    let s_stride = num_groups * out_dim;
    let zp_stride = num_groups * (out_dim / 2);

    let expected = run_moe_gemv::<Cpu, W, Z>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        expert_ids,
        None,
        rht,
        in_dim,
        out_dim,
        routes,
        active,
        bits,
        group_size,
        b_stride,
        s_stride,
        zp_stride,
        1,
        out_dim,
        1,
        out_dim,
        a_per_token,
        shared,
    );

    for_each_non_cpu_backend!(|B| {
        let actual = run_moe_gemv::<B, W, Z>(
            weights,
            scales,
            zero_points,
            biases,
            input,
            expert_ids,
            None,
            rht,
            in_dim,
            out_dim,
            routes,
            active,
            bits,
            group_size,
            b_stride,
            s_stride,
            zp_stride,
            1,
            out_dim,
            1,
            out_dim,
            a_per_token,
            shared,
        );
        assert_vec_close(&actual, &expected, tol, desc);
    });
}

#[uzu_test]
fn moe_gemv_u4_asymmetric_matches_cpu() {
    let (num_experts, in_dim, out_dim, group_size, active) = (16u32, 128u32, 64u32, 64u32, 8u32);
    let num_groups = in_dim / group_size;
    let weights: Vec<u8> =
        (0..(num_experts * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let zero_points: Vec<u8> = (0..(num_experts * num_groups * (out_dim / 2)) as usize)
        .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
        .collect();
    let rht_factors: Vec<i32> = (0..(num_experts * out_dim) as usize)
        .map(|i| {
            if (i * 17 + 3) % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();

    for (batch, use_rht) in [(1u32, false), (1u32, true), (2u32, true), (16u32, true)] {
        let routes = batch * active;
        let input: Vec<bf16> = (0..(batch * in_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();
        let mut expert_ids: Vec<u32> = (0..routes as usize).map(|i| ((i * 5 + 3) as u32) % num_experts).collect();
        expert_ids[0] = num_experts - 1;
        let rht = use_rht.then_some(rht_factors.as_slice());

        compare_moe_gemv(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            rht,
            in_dim,
            out_dim,
            routes,
            active,
            4,
            group_size,
            true,
            2e-2,
            &format!("U4 asymmetric (batch={batch}, rht={use_rht})"),
            None,
        );
    }
}

#[uzu_test]
fn moe_gemv_u8_symmetric_matches_cpu() {
    let (num_experts, in_dim, out_dim, group_size, active, batch) = (8u32, 128u32, 64u32, 64u32, 4u32, 3u32);
    let num_groups = in_dim / group_size;
    let routes = batch * active;
    let weights: Vec<u8> =
        (0..(num_experts * out_dim * in_dim) as usize).map(|i| ((i * 59 + 11) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 23 + 7) % 37) as f32 + 1.0) * 0.002))
        .collect();
    let rht_factors: Vec<i32> = (0..(num_experts * out_dim) as usize)
        .map(|i| {
            if (i * 13 + 1) % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    let input: Vec<bf16> =
        (0..(batch * in_dim) as usize).map(|i| bf16::from_f32((((i * 41 + 9) % 61) as f32 - 30.0) * 0.05)).collect();
    let expert_ids: Vec<u32> = (0..routes as usize).map(|i| ((i * 3 + 1) as u32) % num_experts).collect();

    compare_moe_gemv::<u8, u8>(
        &weights,
        Some(&scales),
        None,
        None,
        &input,
        &expert_ids,
        Some(&rht_factors),
        in_dim,
        out_dim,
        routes,
        active,
        8,
        group_size,
        true,
        2e-2,
        "U8 symmetric",
        None,
    );
}

#[uzu_test]
fn moe_gemv_bf16_and_scale_bias_matches_cpu() {
    let (num_experts, in_dim, out_dim, group_size, active, batch) = (4u32, 128u32, 64u32, 64u32, 2u32, 2u32);
    let routes = batch * active;
    let input: Vec<bf16> =
        (0..(batch * in_dim) as usize).map(|i| bf16::from_f32((((i * 31 + 7) % 53) as f32 - 26.0) * 0.05)).collect();
    let expert_ids: Vec<u32> = (0..routes as usize).map(|i| ((i * 3 + 1) as u32) % num_experts).collect();

    // 1. BF16 unquantized
    let bf16_weights: Vec<bf16> = (0..(num_experts * out_dim * in_dim) as usize)
        .map(|i| bf16::from_f32((((i * 47 + 13) % 79) as f32 - 39.0) * 0.02))
        .collect();
    compare_moe_gemv::<bf16, u8>(
        &bf16_weights,
        None,
        None,
        None,
        &input,
        &expert_ids,
        None,
        in_dim,
        out_dim,
        routes,
        active,
        0,
        0,
        true,
        2e-2,
        "BF16",
        None,
    );

    // 2. Scale-bias dequantization
    let num_groups = in_dim / group_size;
    let u4_weights: Vec<u8> =
        (0..(num_experts * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let biases: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 17 + 3) % 23) as f32 - 11.0) * 0.02))
        .collect();
    compare_moe_gemv::<u8, u8>(
        &u4_weights,
        Some(&scales),
        None,
        Some(&biases),
        &input,
        &expert_ids,
        None,
        in_dim,
        out_dim,
        routes,
        active,
        4,
        group_size,
        true,
        2e-2,
        "Scale-bias",
        None,
    );
}

#[uzu_test]
fn moe_gemv_partial_tile_and_real_shapes_matches_cpu() {
    // 1. Partial tile (out_dim 48 is not divisible by 32 tile size) with group_size 32
    {
        let (num_experts, in_dim, out_dim, group_size, active, routes) = (4u32, 128u32, 48u32, 32u32, 2u32, 2u32);
        let num_groups = in_dim / group_size;
        let weights: Vec<u8> =
            (0..(num_experts * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
        let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
            .collect();
        let zero_points: Vec<u8> = (0..(num_experts * num_groups * (out_dim / 2)) as usize)
            .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
            .collect();
        let input: Vec<bf16> = (0..(routes * in_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();
        let expert_ids = vec![0, num_experts - 1];

        compare_moe_gemv(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            in_dim,
            out_dim,
            routes,
            active,
            4,
            group_size,
            false,
            2e-2,
            "PartialTile",
            None,
        );
    }

    // 2. Qwen3.5 production shapes (2048 -> 1024 and 512 -> 2048)
    for (in_dim, out_dim) in [(2048u32, 1024u32), (512u32, 2048u32)] {
        let (num_experts, group_size, active, batch) = (4u32, 64u32, 2u32, 1u32);
        let routes = batch * active;
        let num_groups = in_dim / group_size;
        let weights: Vec<u8> =
            (0..(num_experts * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
        let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
            .collect();
        let zero_points: Vec<u8> = (0..(num_experts * num_groups * (out_dim / 2)) as usize)
            .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
            .collect();
        let input: Vec<bf16> = (0..(batch * in_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();
        let expert_ids = vec![0, num_experts - 1];

        compare_moe_gemv(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            in_dim,
            out_dim,
            routes,
            active,
            4,
            group_size,
            true,
            2e-2,
            &format!("Shape {in_dim}->{out_dim}"),
            None,
        );
    }
}

#[uzu_test]
fn moe_gemv_shared_expert_matches_cpu() {
    let (num_experts, in_dim, out_dim, group_size, active, batch) = (4u32, 128u32, 64u32, 64u32, 2u32, 2u32);
    let num_groups = in_dim / group_size;
    let num_routed_routes = batch * active;
    let num_routes = num_routed_routes + batch;

    let weights: Vec<u8> =
        (0..(num_experts * out_dim * (in_dim / 2)) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let zero_points: Vec<u8> = (0..(num_experts * num_groups * (out_dim / 2)) as usize)
        .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
        .collect();

    let shared_weights: Vec<u8> =
        (0..(out_dim * (in_dim / 2)) as usize).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let shared_scales: Vec<bf16> = (0..(num_groups * out_dim) as usize)
        .map(|i| bf16::from_f32((((i * 17 + 3) % 31) as f32 + 1.0) * 0.01))
        .collect();
    let shared_zero_points: Vec<u8> = (0..(num_groups * (out_dim / 2)) as usize)
        .map(|i| (((i * 19 + 7) & 0x0F) | (((i * 23 + 5) & 0x0F) << 4)) as u8)
        .collect();

    let expert_ids: Vec<u32> = (0..num_routed_routes as usize).map(|i| (i as u32) % num_experts).collect();
    let shared = SharedGemv {
        weights: &shared_weights,
        scales: Some(&shared_scales),
        zero_points: Some(&shared_zero_points),
        biases: None,
        output_bias: None,
        hadamard: None,
        num_routed_routes,
    };

    for a_per_token in [true, false] {
        let input_rows = if a_per_token {
            batch
        } else {
            num_routes
        };
        let input: Vec<bf16> = (0..(input_rows * in_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();

        compare_moe_gemv(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            in_dim,
            out_dim,
            num_routes,
            active,
            4,
            group_size,
            a_per_token,
            2e-2,
            &format!("Shared expert (a_per_token={a_per_token})"),
            Some(&shared),
        );
    }
}
