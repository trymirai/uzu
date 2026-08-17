use std::num::NonZeroU32;

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
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

fn run(
    input: &[f32],
    weights: &[f32],
    expert_ids: &[i32],
    expert_biases: &[f32],
    input_layout: ExpertInput,
) -> Vec<f32> {
    const ROUTES: usize = 4;
    const ROUTES_PER_TOKEN: u32 = 2;
    const EXPERTS: u32 = 3;
    const K: usize = 3;
    const N: usize = 2;

    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let input = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), input);
    let weights = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), weights);
    let expert_ids = alloc_allocation_with_data::<Cpu, i32>(context.as_ref(), expert_ids);
    let expert_biases = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), expert_biases);
    let mut output = alloc_allocation::<Cpu, f32>(context.as_ref(), ROUTES * N);
    let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
        context.as_ref(),
        DataType::F32,
        DataType::F32,
        DataType::F32,
    )
    .expect("create matmul");
    let mut encoder = Encoder::<Cpu>::new(context.as_ref()).expect("create encoder");

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
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                expert_routes: Some(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(ROUTES_PER_TOKEN).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS).unwrap(),
                    input: input_layout,
                    expert_biases: Some(&expert_biases),
                }),
                m: ROUTES as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .expect("encode routed matmul");
    encoder.end_encoding().submit().wait_until_completed().expect("execute routed matmul");
    allocation_to_vec::<Cpu, f32>(&output)
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
    let actual = run(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &weights(),
        &[1, 0, 1, -1],
        &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
        ExpertInput::Tokens,
    );

    assert_eq_float(&[4.0, 8.0, 1.1, 2.2, 7.0, 17.0, 0.0, 0.0], &actual, 1e-6, "token routes");
}

#[uzu_test]
fn route_rows_reuse_the_same_expert_ids() {
    let actual = run(
        &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0],
        &weights(),
        &[1, 0, 1, -1],
        &[0.1, 0.2, 1.0, 2.0, 10.0, 20.0],
        ExpertInput::Routes,
    );

    assert_eq_float(&[4.0, 8.0, 7.1, 8.2, 7.0, 17.0, 0.0, 0.0], &actual, 1e-6, "route rows");
}
