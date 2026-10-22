use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{MoeCombineKernel, MoeGemvKernel, MoeRouterTopKKernel},
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

const MOE_GEMV_OUTPUT_ROW_TILE: u32 = 32;

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
    (buffer_to_vec(&expert_ids), buffer_to_vec(&expert_weights))
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

fn assert_vec_close(
    actual: &[bf16],
    expected: &[bf16],
    tol: f32,
    desc: &str,
) {
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let (af, ef) = (a.to_f32(), e.to_f32());
        assert!((af - ef).abs() <= tol, "{desc} mismatch at idx={idx}: actual={af}, expected={ef}");
    }
}

#[allow(clippy::too_many_arguments)]
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
    )
    .unwrap();

    let group_count_x = output_dim.div_ceil(MOE_GEMV_OUTPUT_ROW_TILE);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &weights_buf,
        scales_buf.as_ref(),
        zp_buf.as_ref(),
        biases_buf.as_ref(),
        &input_buf,
        &expert_ids_buf,
        &mut output_buf,
        out_bias_buf.as_ref(),
        rht_buf.as_ref(),
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
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec(&output_buf)
}

#[allow(clippy::too_many_arguments)]
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
        );
    }
}

fn run_moe_combine<B: Backend>(
    routed_outputs: &[bf16],
    expert_weights: &[f32],
    shared_output: Option<&[bf16]>,
    shared_gate: Option<&[bf16]>,
    model_dim: u32,
    active_experts_per_token: u32,
    batch_dim: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let routed_buf = create_buffer_with_data::<B, bf16>(&context, routed_outputs);
    let weights_buf = create_buffer_with_data::<B, f32>(&context, expert_weights);
    let shared_out_buf = shared_output.map(|s| create_buffer_with_data::<B, bf16>(&context, s));
    let shared_gate_buf = shared_gate.map(|g| create_buffer_with_data::<B, bf16>(&context, g));
    let mut output_buf = create_buffer::<B, bf16>(&context, (batch_dim * model_dim) as usize);

    let kernel = <B::Kernels as Kernels>::MoeCombineKernel::new(
        &context,
        DataType::BF16,
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
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec(&output_buf)
}

#[uzu_test]
fn moe_combine_matches_cpu() {
    let (model_dim, active) = (128u32, 8u32);

    for (batch_dim, has_shared, has_gate) in
        [(1u32, false, false), (2u32, true, false), (4u32, true, true), (16u32, true, true)]
    {
        let routed_outputs: Vec<bf16> = (0..(batch_dim * active * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 3) % 71) as f32 - 35.0) * 0.05))
            .collect();
        let expert_weights: Vec<f32> = (0..(batch_dim * active) as usize).map(|_| 1.0 / (active as f32)).collect();
        let shared_output: Option<Vec<bf16>> = has_shared.then(|| {
            (0..(batch_dim * model_dim) as usize)
                .map(|i| bf16::from_f32((((i * 17 + 11) % 43) as f32 - 21.0) * 0.05))
                .collect()
        });
        let shared_gate: Option<Vec<bf16>> =
            has_gate.then(|| (0..batch_dim as usize).map(|i| bf16::from_f32((i as f32 - 1.5) * 0.7)).collect());

        let expected = run_moe_combine::<Cpu>(
            &routed_outputs,
            &expert_weights,
            shared_output.as_deref(),
            shared_gate.as_deref(),
            model_dim,
            active,
            batch_dim,
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
            );
            assert_vec_close(
                &actual,
                &expected,
                1e-2,
                &format!("MoeCombine (batch={batch_dim}, shared={has_shared}, gate={has_gate})"),
            );
        });
    }
}
