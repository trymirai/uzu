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
    logits: &[bf16],
    biases: Option<&[bf16]>,
    num_experts: u32,
    k: u32,
    batch_dim: u32,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let logits_buf = create_buffer_with_data::<B, bf16>(&context, logits);
    let biases_buf = biases.map(|b| create_buffer_with_data::<B, bf16>(&context, b));
    let mut expert_ids = create_buffer::<B, u32>(&context, (batch_dim * k) as usize);
    let mut expert_weights = create_buffer::<B, f32>(&context, (batch_dim * k) as usize);

    let kernel = <B::Kernels as Kernels>::MoeRouterTopKKernel::new(&context, DataType::BF16, biases.is_some()).unwrap();
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
        (2u32, 512u32, 32u32, true),
    ] {
        let mut logits: Vec<bf16> = (0..(batch_dim * num_experts) as usize)
            .map(|i| bf16::from_f32((((i * 37 + 13) % 97) as f32 - 48.0) * 0.125))
            .collect();
        // Inject an exact tie to verify deterministic lowest-index tie-breaking
        if num_experts >= 16 {
            logits[3] = bf16::from_f32(10.0);
            logits[11] = bf16::from_f32(10.0);
        }
        let biases: Option<Vec<bf16>> = has_biases.then(|| {
            (0..num_experts as usize).map(|i| bf16::from_f32((((i * 19 + 7) % 31) as f32 - 15.0) * 0.05)).collect()
        });

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

#[uzu_test]
fn moe_gemv_u4_asymmetric_matches_cpu() {
    let num_experts = 16u32;
    let input_dim = 128u32;
    let output_dim = 64u32;
    let group_size = 64u32;
    let num_groups = input_dim / group_size;
    let active_experts_per_token = 8u32;

    let packed_k = input_dim / 2;
    let weights: Vec<u8> =
        (0..(num_experts * output_dim * packed_k) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    // Transposed layout [E, G, N] for scales and packed zero_points [E, G, N/2]
    let scales: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let zero_points: Vec<u8> = (0..(num_experts * num_groups * (output_dim / 2)) as usize)
        .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
        .collect();
    let rht_factors: Vec<i32> = (0..(num_experts * output_dim) as usize)
        .map(|i| {
            if (i * 17 + 3) % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();

    let b_expert_stride_bytes = output_dim * packed_k;
    let scale_expert_stride = num_groups * output_dim;
    let scale_output_stride = 1u32;
    let scale_group_stride = output_dim;
    let zero_point_expert_stride = num_groups * (output_dim / 2);
    let zero_point_output_stride = 1u32;
    let zero_point_group_stride = output_dim;

    for (batch_dim, use_rht) in [(1u32, false), (1u32, true), (2u32, true), (16u32, true)] {
        let num_routes = batch_dim * active_experts_per_token;
        let input: Vec<bf16> = (0..(batch_dim * input_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();
        let mut expert_ids: Vec<u32> = (0..num_routes as usize).map(|i| ((i * 5 + 3) as u32) % num_experts).collect();
        if !expert_ids.is_empty() {
            expert_ids[0] = num_experts - 1; // test boundary expert id E - 1
        }
        let rht = use_rht.then_some(rht_factors.as_slice());

        let expected = run_moe_gemv::<Cpu, u8, u8>(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            rht,
            input_dim,
            output_dim,
            num_routes,
            active_experts_per_token,
            4,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            zero_point_expert_stride,
            scale_output_stride,
            scale_group_stride,
            zero_point_output_stride,
            zero_point_group_stride,
            true,
        );

        for_each_non_cpu_backend!(|B| {
            let actual = run_moe_gemv::<B, u8, u8>(
                &weights,
                Some(&scales),
                Some(&zero_points),
                None,
                &input,
                &expert_ids,
                None,
                rht,
                input_dim,
                output_dim,
                num_routes,
                active_experts_per_token,
                4,
                group_size,
                b_expert_stride_bytes,
                scale_expert_stride,
                zero_point_expert_stride,
                scale_output_stride,
                scale_group_stride,
                zero_point_output_stride,
                zero_point_group_stride,
                true,
            );
            for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
                let af = a.to_f32();
                let ef = e.to_f32();
                assert!(
                    (af - ef).abs() <= 2e-2,
                    "U4 MoeGemv mismatch at idx={idx} (batch={batch_dim}, rht={use_rht}): actual={af}, expected={ef}"
                );
            }
        });
    }
}

#[uzu_test]
fn moe_gemv_u8_symmetric_matches_cpu() {
    let num_experts = 8u32;
    let input_dim = 128u32;
    let output_dim = 64u32;
    let group_size = 64u32;
    let num_groups = input_dim / group_size;
    let active_experts_per_token = 4u32;
    let batch_dim = 3u32;
    let num_routes = batch_dim * active_experts_per_token;

    let weights: Vec<u8> =
        (0..(num_experts * output_dim * input_dim) as usize).map(|i| ((i * 59 + 11) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
        .map(|i| bf16::from_f32((((i * 23 + 7) % 37) as f32 + 1.0) * 0.002))
        .collect();
    let rht_factors: Vec<i32> = (0..(num_experts * output_dim) as usize)
        .map(|i| {
            if (i * 13 + 1) % 2 == 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    let input: Vec<bf16> = (0..(batch_dim * input_dim) as usize)
        .map(|i| bf16::from_f32((((i * 41 + 9) % 61) as f32 - 30.0) * 0.05))
        .collect();
    let expert_ids: Vec<u32> = (0..num_routes as usize).map(|i| ((i * 3 + 1) as u32) % num_experts).collect();

    let b_expert_stride_bytes = output_dim * input_dim;
    let scale_expert_stride = num_groups * output_dim;
    let scale_output_stride = 1u32;
    let scale_group_stride = output_dim;

    let expected = run_moe_gemv::<Cpu, u8, u8>(
        &weights,
        Some(&scales),
        None,
        None,
        &input,
        &expert_ids,
        None,
        Some(&rht_factors),
        input_dim,
        output_dim,
        num_routes,
        active_experts_per_token,
        8,
        group_size,
        b_expert_stride_bytes,
        scale_expert_stride,
        0,
        scale_output_stride,
        scale_group_stride,
        0,
        0,
        true,
    );

    for_each_non_cpu_backend!(|B| {
        let actual = run_moe_gemv::<B, u8, u8>(
            &weights,
            Some(&scales),
            None,
            None,
            &input,
            &expert_ids,
            None,
            Some(&rht_factors),
            input_dim,
            output_dim,
            num_routes,
            active_experts_per_token,
            8,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            0,
            scale_output_stride,
            scale_group_stride,
            0,
            0,
            true,
        );
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let af = a.to_f32();
            let ef = e.to_f32();
            assert!((af - ef).abs() <= 2e-2, "U8 MoeGemv mismatch at idx={idx}: actual={af}, expected={ef}");
        }
    });
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
    let model_dim = 128u32;
    let active_experts_per_token = 8u32;

    for (batch_dim, has_shared, has_gate) in
        [(1u32, false, false), (2u32, true, false), (4u32, true, true), (16u32, true, true)]
    {
        let routed_outputs: Vec<bf16> = (0..(batch_dim * active_experts_per_token * model_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 3) % 71) as f32 - 35.0) * 0.05))
            .collect();
        let expert_weights: Vec<f32> = (0..(batch_dim * active_experts_per_token) as usize)
            .map(|_| 1.0 / (active_experts_per_token as f32))
            .collect();
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
            active_experts_per_token,
            batch_dim,
        );

        for_each_non_cpu_backend!(|B| {
            let actual = run_moe_combine::<B>(
                &routed_outputs,
                &expert_weights,
                shared_output.as_deref(),
                shared_gate.as_deref(),
                model_dim,
                active_experts_per_token,
                batch_dim,
            );
            for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
                let af = a.to_f32();
                let ef = e.to_f32();
                assert!(
                    (af - ef).abs() <= 1e-2,
                    "MoeCombine mismatch at idx={idx} (batch={batch_dim}, shared={has_shared}, gate={has_gate}): actual={af}, expected={ef}"
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
    let mut logits = vec![bf16::from_f32(0.0); (batch_dim * num_experts) as usize];
    for i in 0..num_experts as usize {
        logits[i] = bf16::from_f32((i as f32) * 0.1);
    }
    for i in (0..num_experts as usize).step_by(3) {
        logits[i] = bf16::from_f32(f32::NAN);
    }
    for i in (num_experts as usize)..(2 * num_experts as usize) {
        logits[i] = bf16::from_f32(f32::NAN);
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
    let logits: Vec<bf16> = vec![1.0, 3.0, 3.0, 2.0, 3.0].into_iter().map(bf16::from_f32).collect();
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

#[uzu_test]
fn moe_gemv_bf16_matches_cpu() {
    let num_experts = 4u32;
    let input_dim = 128u32;
    let output_dim = 64u32;
    let active_experts_per_token = 2u32;
    let batch_dim = 2u32;
    let num_routes = batch_dim * active_experts_per_token;

    let weights: Vec<bf16> = (0..(num_experts * output_dim * input_dim) as usize)
        .map(|i| bf16::from_f32((((i * 47 + 13) % 79) as f32 - 39.0) * 0.02))
        .collect();
    let input: Vec<bf16> = (0..(batch_dim * input_dim) as usize)
        .map(|i| bf16::from_f32((((i * 31 + 7) % 53) as f32 - 26.0) * 0.05))
        .collect();
    let expert_ids: Vec<u32> = (0..num_routes as usize).map(|i| ((i * 3 + 1) as u32) % num_experts).collect();

    let b_expert_stride_bytes = output_dim * input_dim * 2;

    let expected = run_moe_gemv::<Cpu, bf16, u8>(
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
        active_experts_per_token,
        0,
        0,
        b_expert_stride_bytes,
        0,
        0,
        0,
        0,
        0,
        0,
        true,
    );

    for_each_non_cpu_backend!(|B| {
        let actual = run_moe_gemv::<B, bf16, u8>(
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
            active_experts_per_token,
            0,
            0,
            b_expert_stride_bytes,
            0,
            0,
            0,
            0,
            0,
            0,
            true,
        );
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let af = a.to_f32();
            let ef = e.to_f32();
            assert!((af - ef).abs() <= 2e-2, "BF16 MoeGemv mismatch at idx={idx}: actual={af}, expected={ef}");
        }
    });
}

#[uzu_test]
fn moe_gemv_scale_bias_matches_cpu() {
    let num_experts = 4u32;
    let input_dim = 128u32;
    let output_dim = 64u32;
    let group_size = 64u32;
    let num_groups = input_dim / group_size;
    let active_experts_per_token = 2u32;
    let batch_dim = 2u32;
    let num_routes = batch_dim * active_experts_per_token;

    let packed_k = input_dim / 2;
    let weights: Vec<u8> =
        (0..(num_experts * output_dim * packed_k) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let biases: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
        .map(|i| bf16::from_f32((((i * 17 + 3) % 23) as f32 - 11.0) * 0.02))
        .collect();
    let input: Vec<bf16> = (0..(batch_dim * input_dim) as usize)
        .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
        .collect();
    let expert_ids: Vec<u32> = (0..num_routes as usize).map(|i| ((i * 5 + 3) as u32) % num_experts).collect();

    let b_expert_stride_bytes = output_dim * packed_k;
    let scale_expert_stride = num_groups * output_dim;
    let scale_output_stride = 1u32;
    let scale_group_stride = output_dim;

    let expected = run_moe_gemv::<Cpu, u8, u8>(
        &weights,
        Some(&scales),
        None,
        Some(&biases),
        &input,
        &expert_ids,
        None,
        None,
        input_dim,
        output_dim,
        num_routes,
        active_experts_per_token,
        4,
        group_size,
        b_expert_stride_bytes,
        scale_expert_stride,
        0,
        scale_output_stride,
        scale_group_stride,
        0,
        0,
        true,
    );

    for_each_non_cpu_backend!(|B| {
        let actual = run_moe_gemv::<B, u8, u8>(
            &weights,
            Some(&scales),
            None,
            Some(&biases),
            &input,
            &expert_ids,
            None,
            None,
            input_dim,
            output_dim,
            num_routes,
            active_experts_per_token,
            4,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            0,
            scale_output_stride,
            scale_group_stride,
            0,
            0,
            true,
        );
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let af = a.to_f32();
            let ef = e.to_f32();
            assert!((af - ef).abs() <= 2e-2, "ScaleBias MoeGemv mismatch at idx={idx}: actual={af}, expected={ef}");
        }
    });
}

#[uzu_test]
fn moe_gemv_partial_tile_and_group32_matches_cpu() {
    let num_experts = 4u32;
    let input_dim = 128u32;
    let output_dim = 48u32; // not divisible by 32 (FullTile = false)
    let group_size = 32u32; // group size 32
    let num_groups = input_dim / group_size;
    let active_experts_per_token = 2u32;
    let batch_dim = 1u32;
    let num_routes = batch_dim * active_experts_per_token;

    let packed_k = input_dim / 2;
    let weights: Vec<u8> =
        (0..(num_experts * output_dim * packed_k) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
    let scales: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
        .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
        .collect();
    let zero_points: Vec<u8> = (0..(num_experts * num_groups * (output_dim / 2)) as usize)
        .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
        .collect();

    let input: Vec<bf16> = (0..(num_routes * input_dim) as usize)
        .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
        .collect();
    let expert_ids: Vec<u32> = vec![0, num_experts - 1];

    let b_expert_stride_bytes = output_dim * packed_k;
    let scale_expert_stride = num_groups * output_dim;
    let scale_output_stride = 1u32;
    let scale_group_stride = output_dim;
    let zero_point_expert_stride = num_groups * (output_dim / 2);
    let zero_point_output_stride = 1u32;
    let zero_point_group_stride = output_dim;

    let expected = run_moe_gemv::<Cpu, u8, u8>(
        &weights,
        Some(&scales),
        Some(&zero_points),
        None,
        &input,
        &expert_ids,
        None,
        None,
        input_dim,
        output_dim,
        num_routes,
        active_experts_per_token,
        4,
        group_size,
        b_expert_stride_bytes,
        scale_expert_stride,
        zero_point_expert_stride,
        scale_output_stride,
        scale_group_stride,
        zero_point_output_stride,
        zero_point_group_stride,
        false, // a_per_token = false
    );

    for_each_non_cpu_backend!(|B| {
        let actual = run_moe_gemv::<B, u8, u8>(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            None,
            input_dim,
            output_dim,
            num_routes,
            active_experts_per_token,
            4,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            zero_point_expert_stride,
            scale_output_stride,
            scale_group_stride,
            zero_point_output_stride,
            zero_point_group_stride,
            false, // a_per_token = false
        );
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let af = a.to_f32();
            let ef = e.to_f32();
            assert!((af - ef).abs() <= 2e-2, "PartialTile MoeGemv mismatch at idx={idx}: actual={af}, expected={ef}");
        }
    });
}

#[uzu_test]
fn moe_gemv_real_shapes_matches_cpu() {
    for (input_dim, output_dim) in [(2048u32, 1024u32), (512u32, 2048u32)] {
        let num_experts = 4u32;
        let group_size = 64u32;
        let num_groups = input_dim / group_size;
        let active_experts_per_token = 2u32;
        let batch_dim = 1u32;
        let num_routes = batch_dim * active_experts_per_token;

        let packed_k = input_dim / 2;
        let weights: Vec<u8> =
            (0..(num_experts * output_dim * packed_k) as usize).map(|i| ((i * 73 + 19) & 0xFF) as u8).collect();
        let scales: Vec<bf16> = (0..(num_experts * num_groups * output_dim) as usize)
            .map(|i| bf16::from_f32((((i * 29 + 5) % 41) as f32 + 1.0) * 0.01))
            .collect();
        let zero_points: Vec<u8> = (0..(num_experts * num_groups * (output_dim / 2)) as usize)
            .map(|i| (((i * 43 + 11) & 0x0F) | (((i * 61 + 3) & 0x0F) << 4)) as u8)
            .collect();
        let input: Vec<bf16> = (0..(batch_dim * input_dim) as usize)
            .map(|i| bf16::from_f32((((i * 31 + 17) % 53) as f32 - 26.0) * 0.05))
            .collect();
        let expert_ids: Vec<u32> = vec![0, num_experts - 1];

        let b_expert_stride_bytes = output_dim * packed_k;
        let scale_expert_stride = num_groups * output_dim;
        let scale_output_stride = 1u32;
        let scale_group_stride = output_dim;
        let zero_point_expert_stride = num_groups * (output_dim / 2);
        let zero_point_output_stride = 1u32;
        let zero_point_group_stride = output_dim;

        let expected = run_moe_gemv::<Cpu, u8, u8>(
            &weights,
            Some(&scales),
            Some(&zero_points),
            None,
            &input,
            &expert_ids,
            None,
            None,
            input_dim,
            output_dim,
            num_routes,
            active_experts_per_token,
            4,
            group_size,
            b_expert_stride_bytes,
            scale_expert_stride,
            zero_point_expert_stride,
            scale_output_stride,
            scale_group_stride,
            zero_point_output_stride,
            zero_point_group_stride,
            true,
        );

        for_each_non_cpu_backend!(|B| {
            let actual = run_moe_gemv::<B, u8, u8>(
                &weights,
                Some(&scales),
                Some(&zero_points),
                None,
                &input,
                &expert_ids,
                None,
                None,
                input_dim,
                output_dim,
                num_routes,
                active_experts_per_token,
                4,
                group_size,
                b_expert_stride_bytes,
                scale_expert_stride,
                zero_point_expert_stride,
                scale_output_stride,
                scale_group_stride,
                zero_point_output_stride,
                zero_point_group_stride,
                true,
            );
            for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
                let af = a.to_f32();
                let ef = e.to_f32();
                assert!(
                    (af - ef).abs() <= 2e-2,
                    "RealShape ({input_dim}->{output_dim}) MoeGemv mismatch at idx={idx}: actual={af}, expected={ef}"
                );
            }
        });
    }
}
