use std::{mem::size_of, num::NonZeroU32};

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
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
    let routes = expert_ids.len();
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), input);
    let weights = alloc_allocation_with_data::<B, f32>(context.as_ref(), weights);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), expert_ids);
    let expert_biases = alloc_allocation_with_data::<B, f32>(context.as_ref(), expert_biases);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), routes * n);
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
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                expert_routes: Some(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(routes_per_token).unwrap(),
                    expert_count: NonZeroU32::new(expert_count).unwrap(),
                    input: input_layout,
                    expert_biases: Some(&expert_biases),
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
                gather_indices: None,
                expert_routes: Some(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(expert_count).unwrap(),
                    input: ExpertInput::Tokens,
                    expert_biases: None,
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
fn backends_reject_oversized_expert_banks() {
    let error = rejection::<Cpu>(513 * 2 * 3, 513, true, None);
    assert!(error.contains("at most 512 experts"), "{error}");
    for_each_non_cpu_backend!(|B| {
        let error = rejection::<B>(513 * 2 * 3, 513, true, None);
        assert!(error.contains("at most 512 experts"), "{error}");
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
