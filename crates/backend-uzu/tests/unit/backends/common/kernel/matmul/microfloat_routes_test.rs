use std::num::NonZeroU32;

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata, decode_mxfp4},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const ROUTES: usize = 4;
const ROUTES_PER_TOKEN: u32 = 2;
const EXPERTS: usize = 3;
const K: usize = 32;
const N: usize = 4;

fn packed_codes() -> Vec<u8> {
    let mut codes = vec![0u8; EXPERTS * N * K / 2];
    for matrix in 0..EXPERTS {
        for row in 0..N {
            for inner in (0..K).step_by(2) {
                let low = ((matrix + row + inner) % 7 + 1) as u8;
                let high = ((matrix * 3 + row + inner + 1) % 7 + 1) as u8;
                codes[(matrix * N + row) * K / 2 + inner / 2] = low | (high << 4);
            }
        }
    }
    codes
}

fn run<B: Backend>(group_size: u32) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..2 * K).map(|index| (index % 11) as f32 * 0.1 - 0.4).collect();
    let codes = packed_codes();
    let scales = vec![127u8; EXPERTS * N * K / group_size as usize];
    let global_scales = [1.0f32, 2.0, 0.5];
    let biases: Vec<f32> = (0..EXPERTS * N).map(|index| index as f32 * 0.01).collect();
    let expert_ids = [1i32, 0, 1, -1];
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        group_size,
        MicrofloatLayout::OutputInput,
        EXPERTS as u32,
        N as u32,
        K as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let global_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &global_scales);
    let biases_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &biases);
    let ids_alloc = alloc_allocation_with_data::<B, i32>(context.as_ref(), &expert_ids);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), ROUTES * N);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input_alloc,
                    offset: 0,
                },
                b: MatmulB::<B>::Microfloat {
                    codes: &codes_alloc,
                    scales: &scales_alloc,
                    global_scales: &global_scales_alloc,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None,
                expert_routes: Some(ExpertRoutes {
                    expert_ids: &ids_alloc,
                    routes_per_token: NonZeroU32::new(ROUTES_PER_TOKEN).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: ExpertInput::Tokens,
                    expert_biases: Some(&biases_alloc),
                }),
                m: ROUTES as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<B, f32>(&output);

    let mut expected = vec![0.0f32; ROUTES * N];
    for route in 0..ROUTES {
        let expert = expert_ids[route];
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        let token = route / ROUTES_PER_TOKEN as usize;
        for row in 0..N {
            let mut value = biases[expert * N + row];
            for inner in 0..K {
                let packed = codes[(expert * N + row) * K / 2 + inner / 2];
                let code = if inner.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                value += input[token * K + inner] * decode_mxfp4(code, 127, global_scales[expert]);
            }
            expected[route * N + row] = value;
        }
    }
    (actual, expected)
}

#[uzu_test]
fn cpu_decodes_group_16_and_32_route_banks() {
    for group_size in [16, 32] {
        let (cpu, expected) = run::<Cpu>(group_size);
        assert_eq_float(&expected, &cpu, 1e-5, "CPU MXFP4 expert routes");
        for_each_non_cpu_backend!(|B| {
            let (actual, _) = run::<B>(group_size);
            assert_eq_float(&cpu, &actual, 1e-4, "Metal MXFP4 expert routes");
        });
    }
}
