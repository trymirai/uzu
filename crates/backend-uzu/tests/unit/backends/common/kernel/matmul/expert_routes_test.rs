use std::{mem::size_of, num::NonZeroU32};

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            gpu_types::QuantizationMode,
            kernel::matmul::{
                ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting,
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

fn run<B: Backend>(
    input: &[f32],
    weights: &[f32],
    expert_ids: &[i32],
    expert_biases: &[f32],
    input_layout: ExpertInput,
    routes_per_token: u32,
    expert_count: u32,
    k: usize,
    n: usize,
) -> Vec<f32> {
    run_with_offset::<B>(
        input,
        0,
        weights,
        expert_ids,
        expert_biases,
        input_layout,
        routes_per_token,
        expert_count,
        k,
        n,
    )
}

fn run_quantized_routes<B: Backend>() -> Vec<f32> {
    const EXPERTS: usize = 3;
    const ROUTES: usize = 33;
    const N: usize = 8;
    const K: usize = 32;

    let input: Vec<bf16> = (0..ROUTES * K).map(|index| bf16::from_f32((index % 17) as f32 * 0.03 - 0.2)).collect();
    let codes: Vec<u8> = (0..EXPERTS * N * K).map(|index| 124 + (index % 9) as u8).collect();
    let scales: Vec<bf16> = (0..EXPERTS * N).map(|index| bf16::from_f32(0.01 + (index % 5) as f32 * 0.002)).collect();
    let expert_ids: Vec<i32> = (0..ROUTES).map(|route| ((route * 5 + 1) % EXPERTS) as i32).collect();

    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, bf16>(context.as_ref(), &input);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales = alloc_allocation_with_data::<B, bf16>(context.as_ref(), &scales);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &expert_ids);
    let mut output = alloc_allocation::<B, bf16>(context.as_ref(), ROUTES * N);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::BF16, DataType::BF16, DataType::BF16)
            .expect("create matmul");
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");

    kernel
        .encode(
            MatmulArguments::<B> {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::ScaleSymmetricDequant {
                    b: &codes,
                    scales: &scales,
                    mode: QuantizationMode::U8,
                    group_size: K as u32,
                    signed_codes: false,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: ExpertInput::Routes,
                }),
                m: ROUTES as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .expect("encode quantized expert routes");
    encoder.end_encoding().submit().wait_until_completed().expect("execute quantized expert routes");
    allocation_to_vec::<B, bf16>(&output).into_iter().map(f32::from).collect()
}

#[allow(clippy::too_many_arguments)]
fn run_with_offset<B: Backend>(
    input: &[f32],
    input_byte_offset: usize,
    weights: &[f32],
    expert_ids: &[i32],
    expert_biases: &[f32],
    input_layout: ExpertInput,
    routes_per_token: u32,
    expert_count: u32,
    k: usize,
    n: usize,
) -> Vec<f32> {
    run_with_offset_and_output::<B>(
        input,
        input_byte_offset,
        weights,
        expert_ids,
        expert_biases,
        input_layout,
        routes_per_token,
        expert_count,
        k,
        n,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_with_offset_and_output<B: Backend>(
    input: &[f32],
    input_byte_offset: usize,
    weights: &[f32],
    expert_ids: &[i32],
    expert_biases: &[f32],
    input_layout: ExpertInput,
    routes_per_token: u32,
    expert_count: u32,
    k: usize,
    n: usize,
    initial_output: Option<&[f32]>,
) -> Vec<f32> {
    let routes = expert_ids.len();
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), input);
    let weights = alloc_allocation_with_data::<B, f32>(context.as_ref(), weights);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), expert_ids);
    let expert_biases = alloc_allocation_with_data::<B, f32>(context.as_ref(), expert_biases);
    let mut output = match initial_output {
        Some(initial_output) => {
            assert_eq!(initial_output.len(), routes * n);
            alloc_allocation_with_data::<B, f32>(context.as_ref(), initial_output)
        },
        None => alloc_allocation::<B, f32>(context.as_ref(), routes * n),
    };
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .expect("create matmul");
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");

    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: input_byte_offset,
                },
                b: MatmulB::FullPrecision {
                    b: &weights,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&expert_biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(routes_per_token).unwrap(),
                    expert_count: NonZeroU32::new(expert_count).unwrap(),
                    input: input_layout,
                }),
                m: routes as u32,
                n: n as u32,
                k: k as u32,
            },
            &mut encoder,
        )
        .expect("encode routed matmul");
    encoder.end_encoding().submit().wait_until_completed().expect("execute routed matmul");
    allocation_to_vec::<B, f32>(&output)
}

fn rejection<B: Backend>(
    weight_count: usize,
    expert_count: u32,
    b_transpose: bool,
    b_leading_dimension: Option<u32>,
) -> String {
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &[1.0, 2.0, 3.0]);
    let weights = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![0.0; weight_count]);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), 2);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::FullPrecision {
                    b: &weights,
                },
                b_leading_dimension,
                b_transpose,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(expert_count).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: 1,
                n: 2,
                k: 3,
            },
            &mut encoder,
        )
        .expect_err("invalid full-precision route contract was accepted")
        .to_string()
}

fn storage_rejection<B: Backend>(
    input_count: usize,
    input_byte_offset: usize,
    output_count: usize,
) -> String {
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![1.0; input_count]);
    let weights = alloc_allocation_with_data::<B, f32>(context.as_ref(), &[1.0; 6]);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), output_count);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: input_byte_offset,
                },
                b: MatmulB::FullPrecision {
                    b: &weights,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(1).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: 1,
                n: 2,
                k: 3,
            },
            &mut encoder,
        )
        .expect_err("invalid matmul storage was accepted")
        .to_string()
}

fn run_bf16_offset<B: Backend>() -> Vec<bf16> {
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, bf16>(
        context.as_ref(),
        &[bf16::from_f32(99.0), bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)],
    );
    let weights = alloc_allocation_with_data::<B, bf16>(
        context.as_ref(),
        &[
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
        ],
    );
    let biases = alloc_allocation_with_data::<B, bf16>(context.as_ref(), &[bf16::from_f32(0.1), bf16::from_f32(0.2)]);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, bf16>(context.as_ref(), 2);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::BF16, DataType::BF16, DataType::BF16)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: size_of::<bf16>(),
                },
                b: MatmulB::FullPrecision {
                    b: &weights,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(1).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: 1,
                n: 2,
                k: 3,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, bf16>(&output)
}

#[uzu_test]
fn backends_reject_invalid_full_precision_banks() {
    for (weight_count, leading_dimension) in [(6, None), (12, Some(2))] {
        let error = rejection::<Cpu>(weight_count, 2, true, leading_dimension);
        assert!(error.contains("full-precision weight bank layout or storage"), "{error}");
        for_each_non_cpu_backend!(|B| {
            let error = rejection::<B>(weight_count, 2, true, leading_dimension);
            assert!(error.contains("full-precision weight bank layout or storage"), "{error}");
        });
    }
}

#[uzu_test]
fn backends_reject_invalid_input_and_output_storage() {
    for (input_count, input_byte_offset, output_count, expected) in [
        (3, 1, 2, "byte offset is not aligned"),
        (2, 0, 2, "input allocation does not cover"),
        (3, 0, 1, "output allocation does not cover"),
    ] {
        let error = storage_rejection::<Cpu>(input_count, input_byte_offset, output_count);
        assert!(error.contains(expected), "{error}");
        for_each_non_cpu_backend!(|B| {
            let error = storage_rejection::<B>(input_count, input_byte_offset, output_count);
            assert!(error.contains(expected), "{error}");
        });
    }
}

#[uzu_test]
fn direct_expert_matmul_is_not_limited_by_router_capacity() {
    const EXPERT_COUNT: usize = 513;
    const K: usize = 3;
    const N: usize = 2;

    let mut weights = vec![0.0; EXPERT_COUNT * N * K];
    let last_expert = (EXPERT_COUNT - 1) * N * K;
    weights[last_expert] = 1.0;
    weights[last_expert + K + 1] = 1.0;
    let biases = vec![0.0; EXPERT_COUNT * N];
    let expected = [1.0, 2.0];

    let actual = run::<Cpu>(
        &[1.0, 2.0, 3.0],
        &weights,
        &[(EXPERT_COUNT - 1) as i32],
        &biases,
        ExpertInput::Tokens,
        1,
        EXPERT_COUNT as u32,
        K,
        N,
    );
    assert_eq_float(&expected, &actual, 1e-6, "CPU direct route beyond router capacity");
    for_each_non_cpu_backend!(|B| {
        let actual = run::<B>(
            &[1.0, 2.0, 3.0],
            &weights,
            &[(EXPERT_COUNT - 1) as i32],
            &biases,
            ExpertInput::Tokens,
            1,
            EXPERT_COUNT as u32,
            K,
            N,
        );
        assert_eq_float(&expected, &actual, 1e-6, "Metal direct route beyond router capacity");
    });
}

#[uzu_test]
fn metal_rejects_unsupported_grouped_weight_layouts() {
    for_each_non_cpu_backend!(|B| {
        let error = rejection::<B>(3 * 2 * 3, 3, false, None);
        assert!(error.contains("contiguous output-input weights"), "{error}");
    });
}

#[uzu_test]
fn full_precision_input_offsets_are_bytes() {
    let input = [99.0, 88.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = [4.0, 8.0, 1.1, 2.2, 7.0, 17.0, 0.0, 0.0];
    let cpu = run_with_offset::<Cpu>(
        &input,
        2 * size_of::<f32>(),
        &weights(),
        &[1, 0, 1, -1],
        &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
        ExpertInput::Tokens,
        2,
        3,
        3,
        2,
    );
    assert_eq_float(&expected, &cpu, 1e-6, "CPU byte-offset routes");
    for_each_non_cpu_backend!(|B| {
        let actual = run_with_offset::<B>(
            &input,
            2 * size_of::<f32>(),
            &weights(),
            &[1, 0, 1, -1],
            &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
            ExpertInput::Tokens,
            2,
            3,
            3,
            2,
        );
        assert_eq_float(&expected, &actual, 1e-6, "Metal byte-offset routes");
    });

    let expected = [bf16::from_f32(1.1), bf16::from_f32(2.2)];
    let cpu = run_bf16_offset::<Cpu>();
    assert_eq_float(&expected, &cpu, 0.01, "CPU BF16 byte-offset routes");
    for_each_non_cpu_backend!(|B| {
        let actual = run_bf16_offset::<B>();
        assert_eq_float(&expected, &actual, 0.01, "Metal BF16 byte-offset routes");
    });
}

#[uzu_test]
fn tiny_output_rows_are_safe_for_the_final_expert() {
    const EXPERT_COUNT: usize = 3;

    for n in 1..=3 {
        for k in [127, 128] {
            let input: Vec<f32> = (0..3 * k).map(|index| (index % 17) as f32 * 0.01 - 0.08).collect();
            let weights: Vec<f32> = (0..EXPERT_COUNT * n * k).map(|index| (index % 13) as f32 * 0.005 - 0.03).collect();
            let biases: Vec<f32> = (0..EXPERT_COUNT * n).map(|index| (index % 5) as f32 * 0.02 - 0.04).collect();
            let expert_ids = [2, 0, 1];
            let expected = run::<Cpu>(&input, &weights, &expert_ids, &biases, ExpertInput::Tokens, 1, 3, k, n);

            for_each_non_cpu_backend!(|B| {
                let actual = run::<B>(&input, &weights, &expert_ids, &biases, ExpertInput::Tokens, 1, 3, k, n);
                assert_eq_float(&expected, &actual, 1e-4, "tiny routed output rows");
            });
        }
    }
}

#[uzu_test]
fn invalid_routes_zero_only_their_split_k_output_tiles() {
    const EXPERT_COUNT: usize = 3;
    const K: usize = 512;
    const N: usize = 65;

    let input = vec![1.0; 2 * K];
    let weights = vec![1.0; EXPERT_COUNT * N * K];
    let biases = vec![7.0; EXPERT_COUNT * N];
    let expert_ids = [-1, EXPERT_COUNT as i32];
    let initial_output = vec![11.0; expert_ids.len() * N];
    let expected = vec![0.0; expert_ids.len() * N];

    let actual = run_with_offset_and_output::<Cpu>(
        &input,
        0,
        &weights,
        &expert_ids,
        &biases,
        ExpertInput::Tokens,
        1,
        EXPERT_COUNT as u32,
        K,
        N,
        Some(&initial_output),
    );
    assert_eq_float(&expected, &actual, 0.0, "CPU invalid routed rows");
    for_each_non_cpu_backend!(|B| {
        let actual = run_with_offset_and_output::<B>(
            &input,
            0,
            &weights,
            &expert_ids,
            &biases,
            ExpertInput::Tokens,
            1,
            EXPERT_COUNT as u32,
            K,
            N,
            Some(&initial_output),
        );
        assert_eq_float(&expected, &actual, 0.0, "Metal invalid routed rows");
    });
}

#[uzu_test]
fn maximum_expert_and_active_route_boundaries_are_safe() {
    const EXPERT_COUNT: usize = 512;
    const ROUTES_PER_TOKEN: usize = 128;
    const K: usize = 3;
    const N: usize = 2;

    let mut weights = vec![0.0; EXPERT_COUNT * N * K];
    let last_expert = (EXPERT_COUNT - 1) * N * K;
    weights[last_expert] = 1.0;
    weights[last_expert + K + 1] = 1.0;
    let biases = vec![0.0; EXPERT_COUNT * N];
    let mut expert_ids = vec![(EXPERT_COUNT - 1) as i32; ROUTES_PER_TOKEN];
    expert_ids[ROUTES_PER_TOKEN - 1] = EXPERT_COUNT as i32;

    let cpu = run::<Cpu>(
        &[1.0, 2.0, 3.0],
        &weights,
        &expert_ids,
        &biases,
        ExpertInput::Tokens,
        ROUTES_PER_TOKEN as u32,
        EXPERT_COUNT as u32,
        K,
        N,
    );
    let mut expected = [1.0, 2.0].repeat(ROUTES_PER_TOKEN);
    expected[(ROUTES_PER_TOKEN - 1) * N..].fill(0.0);
    assert_eq_float(&expected, &cpu, 1e-6, "CPU route boundaries");
    for_each_non_cpu_backend!(|B| {
        let actual = run::<B>(
            &[1.0, 2.0, 3.0],
            &weights,
            &expert_ids,
            &biases,
            ExpertInput::Tokens,
            ROUTES_PER_TOKEN as u32,
            EXPERT_COUNT as u32,
            K,
            N,
        );
        assert_eq_float(&expected, &actual, 1e-6, "Metal route boundaries");
    });
}

fn weights() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, // expert 0, row 0
        0.0, 1.0, 0.0, // expert 0, row 1
        0.0, 0.0, 1.0, // expert 1, row 0
        1.0, 1.0, 1.0, // expert 1, row 1
        2.0, 2.0, 2.0, // unused expert 2, row 0
        3.0, 3.0, 3.0, // unused expert 2, row 1
    ]
}

#[uzu_test]
fn token_rows_feed_route_major_experts_directly() {
    let actual = run::<Cpu>(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &weights(),
        &[1, 0, 1, -1],
        &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
        ExpertInput::Tokens,
        2,
        3,
        3,
        2,
    );

    assert_eq_float(&[4.0, 8.0, 1.1, 2.2, 7.0, 17.0, 0.0, 0.0], &actual, 1e-6, "token routes");
}

#[uzu_test]
fn route_rows_reuse_the_same_expert_ids() {
    let actual = run::<Cpu>(
        &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        &weights(),
        &[1, 0, 1, -1],
        &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
        ExpertInput::Routes,
        2,
        3,
        3,
        2,
    );

    assert_eq_float(&[4.0, 8.0, 7.1, 8.2, 7.0, 17.0, 0.0, 0.0], &actual, 1e-6, "route rows");
}

#[uzu_test]
fn accelerators_match_cpu_for_direct_routes() {
    const ROUTES: usize = 4;
    const ROUTES_PER_TOKEN: u32 = 2;
    const EXPERTS: u32 = 3;
    const K: usize = 128;
    const N: usize = 64;

    let expert_ids = [1, 0, 1, -1];
    let weights: Vec<f32> = (0..EXPERTS as usize * N * K).map(|index| (index % 23) as f32 * 0.01 - 0.11).collect();
    let biases: Vec<f32> = (0..EXPERTS as usize * N).map(|index| (index % 7) as f32 * 0.02).collect();

    for input_layout in [ExpertInput::Tokens, ExpertInput::Routes] {
        let input_rows = if input_layout == ExpertInput::Tokens {
            ROUTES / ROUTES_PER_TOKEN as usize
        } else {
            ROUTES
        };
        let input: Vec<f32> = (0..input_rows * K).map(|index| (index % 19) as f32 * 0.03 - 0.27).collect();
        let expected =
            run::<Cpu>(&input, &weights, &expert_ids, &biases, input_layout, ROUTES_PER_TOKEN, EXPERTS, K, N);

        for_each_non_cpu_backend!(|B| {
            let actual =
                run::<B>(&input, &weights, &expert_ids, &biases, input_layout, ROUTES_PER_TOKEN, EXPERTS, K, N);
            assert_eq_float(&expected, &actual, 1e-4, "direct expert routes");
        });
    }
}

#[uzu_test]
fn grouped_prefill_routes_remain_route_major() {
    const ROUTES: usize = 33;
    const ROUTES_PER_TOKEN: u32 = 3;
    const EXPERTS: u32 = 5;
    const K: usize = 37;
    const N: usize = 17;

    let mut expert_ids: Vec<i32> = (0..ROUTES).map(|route| ((route * 7 + 1) % 4) as i32).collect();
    expert_ids[19] = -1;
    let weights: Vec<f32> = (0..EXPERTS as usize * N * K).map(|index| (index % 29) as f32 * 0.007 - 0.09).collect();
    let biases: Vec<f32> = (0..EXPERTS as usize * N).map(|index| (index % 11) as f32 * 0.013).collect();

    for input_layout in [ExpertInput::Tokens, ExpertInput::Routes] {
        let input_rows = if input_layout == ExpertInput::Tokens {
            ROUTES / ROUTES_PER_TOKEN as usize
        } else {
            ROUTES
        };
        let input: Vec<f32> = (0..input_rows * K).map(|index| (index % 31) as f32 * 0.009 - 0.12).collect();
        let expected =
            run::<Cpu>(&input, &weights, &expert_ids, &biases, input_layout, ROUTES_PER_TOKEN, EXPERTS, K, N);

        for_each_non_cpu_backend!(|B| {
            let actual =
                run::<B>(&input, &weights, &expert_ids, &biases, input_layout, ROUTES_PER_TOKEN, EXPERTS, K, N);
            assert_eq_float(&expected, &actual, 1e-4, "private grouped expert routes");
        });
    }
}

#[uzu_test]
fn integer_quantized_weights_remain_independent_from_expert_routing() {
    let expected = run_quantized_routes::<Cpu>();
    for_each_non_cpu_backend!(|B| {
        let actual = run_quantized_routes::<B>();
        assert_eq_float(&expected, &actual, 1e-3, "quantized expert routes");
    });
}
