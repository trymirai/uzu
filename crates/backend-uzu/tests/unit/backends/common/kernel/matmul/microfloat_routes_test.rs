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

fn run<B: Backend>(
    group_size: u32,
    route_count: usize,
    routes_per_token: u32,
    input_layout: ExpertInput,
) -> (Vec<f32>, Vec<f32>) {
    let input_rows = if input_layout == ExpertInput::Tokens {
        assert!(route_count.is_multiple_of(routes_per_token as usize));
        route_count / routes_per_token as usize
    } else {
        route_count
    };
    let input: Vec<f32> = (0..input_rows * K).map(|index| (index % 11) as f32 * 0.1 - 0.4).collect();
    let codes = packed_codes();
    let scales: Vec<u8> = (0..EXPERTS * N * K / group_size as usize).map(|index| 126 + (index % 3) as u8).collect();
    let global_scales = [1.0f32, 2.0, 0.5];
    let biases: Vec<f32> = (0..EXPERTS * N).map(|index| index as f32 * 0.01).collect();
    let expert_ids: Vec<i32> = (0..route_count)
        .map(|route| {
            if route % 7 == 3 {
                -1
            } else {
                ((route * 2 + 1) % EXPERTS) as i32
            }
        })
        .collect();
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
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), route_count * N);
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
                    routes_per_token: NonZeroU32::new(routes_per_token).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: input_layout,
                    expert_biases: Some(&biases_alloc),
                }),
                m: route_count as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<B, f32>(&output);

    let mut expected = vec![0.0f32; route_count * N];
    for route in 0..route_count {
        let expert = expert_ids[route];
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        let input_row = if input_layout == ExpertInput::Tokens {
            route / routes_per_token as usize
        } else {
            route
        };
        for row in 0..N {
            let mut value = biases[expert * N + row];
            for inner in 0..K {
                let packed = codes[(expert * N + row) * K / 2 + inner / 2];
                let code = if inner.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale_index = (expert * N + row) * K / group_size as usize + inner / group_size as usize;
                value += input[input_row * K + inner] * decode_mxfp4(code, scales[scale_index], global_scales[expert]);
            }
            expected[route * N + row] = value;
        }
    }
    (actual, expected)
}

fn run_dense<B: Backend>(
    group_size: u32,
    row_count: usize,
) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..row_count * K).map(|index| (index % 13) as f32 * 0.125 - 0.5).collect();
    let codes = packed_codes()[..N * K / 2].to_vec();
    let scales: Vec<u8> = (0..N * K / group_size as usize).map(|index| 126 + (index % 3) as u8).collect();
    let global_scales = [1.25f32];
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        group_size,
        MicrofloatLayout::OutputInput,
        1,
        N as u32,
        K as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let global_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &global_scales);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), row_count * N);
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
                expert_routes: None,
                m: row_count as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<B, f32>(&output);

    let mut expected = vec![0.0f32; row_count * N];
    for row in 0..row_count {
        for output_row in 0..N {
            let mut value = 0.0f32;
            for inner in 0..K {
                let packed = codes[output_row * K / 2 + inner / 2];
                let code = if inner.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale_index = output_row * K / group_size as usize + inner / group_size as usize;
                let weight = decode_mxfp4(code, scales[scale_index], global_scales[0]);
                value += input[row * K + inner] * weight;
            }
            expected[row * N + output_row] = value;
        }
    }
    (actual, expected)
}

fn run_sparse_readout<B: Backend>(group_size: u32) -> (Vec<f32>, Vec<f32>) {
    const INPUT_ROWS: usize = 9;
    const READOUT_ROWS: usize = 4;
    const VOCAB_ROWS: usize = 7;

    let input: Vec<f32> = (0..INPUT_ROWS * K).map(|index| (index % 17) as f32 * 0.0625 - 0.375).collect();
    let codes: Vec<u8> = (0..VOCAB_ROWS * K / 2)
        .map(|index| {
            let row = index / (K / 2);
            let low = ((row * 3 + index) % 7 + 1) as u8;
            let high = ((row * 5 + index * 2 + 1) % 7 + 1) as u8;
            low | (high << 4)
        })
        .collect();
    let scales: Vec<u8> = (0..VOCAB_ROWS * K / group_size as usize)
        .map(|index| 124 + ((index * 3 + 1) % 5) as u8)
        .collect();
    let global_scales = [1.25f32];
    let readout_rows: Vec<u32> = (0..INPUT_ROWS)
        .flat_map(|row| {
            let rows = [6, 1, 4, 1];
            rows.map(|physical_row| ((physical_row + row * 2) % VOCAB_ROWS) as u32)
        })
        .collect();
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        group_size,
        MicrofloatLayout::OutputInput,
        1,
        VOCAB_ROWS as u32,
        K as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let global_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &global_scales);
    let readout_rows_alloc = alloc_allocation_with_data::<B, u32>(context.as_ref(), &readout_rows);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), INPUT_ROWS * READOUT_ROWS);
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
                gather_indices: Some(&readout_rows_alloc),
                expert_routes: None,
                m: INPUT_ROWS as u32,
                n: READOUT_ROWS as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<B, f32>(&output);

    let mut expected = vec![0.0f32; INPUT_ROWS * READOUT_ROWS];
    for input_row in 0..INPUT_ROWS {
        for readout_row in 0..READOUT_ROWS {
            let physical_row = readout_rows[input_row * READOUT_ROWS + readout_row] as usize;
            for inner in 0..K {
                let packed = codes[physical_row * K / 2 + inner / 2];
                let code = if inner.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale_index = physical_row * K / group_size as usize + inner / group_size as usize;
                let weight = decode_mxfp4(code, scales[scale_index], global_scales[0]);
                expected[input_row * READOUT_ROWS + readout_row] += input[input_row * K + inner] * weight;
            }
        }
    }
    (actual, expected)
}

fn rejection<B: Backend>(
    matrix_count: u32,
    expert_count: u32,
) -> String {
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        16,
        MicrofloatLayout::OutputInput,
        matrix_count,
        N as u32,
        K as u32,
    )
    .unwrap();
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &[1.0; K]);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &vec![0x22; metadata.required_code_bytes()]);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &vec![127; metadata.required_scale_bytes()]);
    let global_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![1.0; matrix_count as usize]);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), N);
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
                b: MatmulB::<B>::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    global_scales: &global_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
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
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .expect_err("invalid microfloat routing was accepted")
        .to_string()
}

#[uzu_test]
fn backends_decode_direct_and_grouped_microfloat_routes() {
    for (route_count, routes_per_token) in [(4, 2), (33, 3)] {
        for group_size in [16, 32] {
            for input_layout in [ExpertInput::Tokens, ExpertInput::Routes] {
                let (cpu, expected) = run::<Cpu>(group_size, route_count, routes_per_token, input_layout);
                assert_eq_float(&expected, &cpu, 1e-5, "CPU MXFP4 expert routes");
                for_each_non_cpu_backend!(|B| {
                    let (actual, _) = run::<B>(group_size, route_count, routes_per_token, input_layout);
                    assert_eq_float(&cpu, &actual, 1e-4, "Metal MXFP4 expert routes");
                });
            }
        }
    }
}

#[uzu_test]
fn backends_execute_dense_microfloat_gemv_and_gemm() {
    for row_count in [1, 33] {
        for group_size in [16, 32] {
            let (cpu, expected) = run_dense::<Cpu>(group_size, row_count);
            assert_eq_float(&expected, &cpu, 1e-5, "CPU dense MXFP4");
            for_each_non_cpu_backend!(|B| {
                let (actual, _) = run_dense::<B>(group_size, row_count);
                assert_eq_float(&cpu, &actual, 1e-4, "Metal dense MXFP4");
            });
        }
    }
}

#[uzu_test]
fn backends_honor_sparse_microfloat_rows_above_the_gemv_threshold() {
    for group_size in [16, 32] {
        let (cpu, expected) = run_sparse_readout::<Cpu>(group_size);
        assert_eq_float(&expected, &cpu, 1e-5, "CPU sparse MXFP4 readout");
        for_each_non_cpu_backend!(|B| {
            let (actual, _) = run_sparse_readout::<B>(group_size);
            assert_eq_float(&cpu, &actual, 1e-4, "Metal sparse MXFP4 readout");
        });
    }
}

#[uzu_test]
fn backends_reject_incomplete_microfloat_expert_banks() {
    let expected = "microfloat storage does not match the requested matrix operand";
    let error = rejection::<Cpu>(1, 2);
    assert!(error.contains(expected), "{error}");
    for_each_non_cpu_backend!(|B| {
        let error = rejection::<B>(1, 2);
        assert!(error.contains(expected), "{error}");
    });
}
