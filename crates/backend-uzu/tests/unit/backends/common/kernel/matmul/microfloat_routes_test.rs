use std::num::NonZeroU32;

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{
                ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting,
            },
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

// Captured verbatim from openai/gpt-oss-20b revision
// 6cee5e81ee83917806bbde320786a8fb61efebee,
// model-00000-of-00002.safetensors
// (SHA-256 16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427).
// This is expert 0's first 32-value input group and its eight gate/up rows.
const GPT_OSS_INPUT_BF16: [u8; 64] = [
    0xb2, 0x3f, 0xa1, 0x3f, 0x8a, 0x3f, 0x6e, 0x40, 0x94, 0x3f, 0x2e, 0x3f, 0x00, 0x40, 0x03, 0x40, 0x51, 0x3f, 0x83,
    0x3f, 0x8b, 0x40, 0xba, 0x3f, 0x7b, 0x3f, 0x35, 0x3f, 0x70, 0x40, 0x18, 0x40, 0x44, 0x40, 0x98, 0x40, 0x96, 0x3f,
    0xc4, 0x40, 0xa2, 0x3f, 0xa7, 0x40, 0x7e, 0x3f, 0xa3, 0x40, 0x94, 0x40, 0x4d, 0x40, 0x2c, 0x40, 0x08, 0x3f, 0x13,
    0x40, 0xb6, 0x3f, 0x7c, 0x3f, 0xbb, 0x3f,
];
const GPT_OSS_GROUP32_CODES: [u8; 128] = [
    0x00, 0xc0, 0x80, 0xa9, 0x10, 0x1b, 0x81, 0x22, 0x93, 0xe4, 0xa0, 0xb2, 0x3b, 0x19, 0xb2, 0x31, 0x82, 0xf8, 0xce,
    0xa8, 0x6e, 0x82, 0xa2, 0x0f, 0xc5, 0x76, 0xc2, 0x7a, 0xcc, 0x8c, 0xc2, 0x24, 0x45, 0x65, 0xdd, 0xf5, 0xc2, 0x29,
    0x81, 0xf5, 0x66, 0xe7, 0x43, 0xca, 0x04, 0xb1, 0x52, 0x91, 0x04, 0x0a, 0x58, 0x44, 0x68, 0x45, 0x82, 0xc7, 0x62,
    0x26, 0x7a, 0x78, 0x8d, 0x02, 0x40, 0x0c, 0x4e, 0xc6, 0xa8, 0x5d, 0xa2, 0x45, 0x0a, 0x26, 0xaa, 0xd2, 0x44, 0xe7,
    0x56, 0x47, 0x2e, 0xdf, 0x5b, 0xcc, 0xa9, 0xad, 0x59, 0xe4, 0x03, 0x51, 0x4d, 0xe4, 0x54, 0x58, 0x29, 0x1a, 0x28,
    0xaa, 0x11, 0x21, 0x18, 0x12, 0xcc, 0x25, 0x00, 0xc8, 0x08, 0xe1, 0x88, 0x3b, 0x52, 0xa1, 0x0c, 0x13, 0xc5, 0x6d,
    0xd6, 0x6a, 0xde, 0x6d, 0x28, 0x0a, 0xcc, 0xc8, 0xfa, 0xdd, 0x6d, 0xa2, 0xf6, 0xac,
];
const GPT_OSS_GROUP32_SCALES: [u8; 8] = [0x7a, 0x79, 0x78, 0x79, 0x78, 0x79, 0x79, 0x79];
const GPT_OSS_BIASES_BF16: [u8; 16] =
    [0x3e, 0xbf, 0x46, 0xbf, 0xd6, 0xbe, 0x67, 0xbf, 0xfa, 0xbe, 0x65, 0xbf, 0xc7, 0xbe, 0x50, 0xbf];

fn bf16_fixture(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32()).collect()
}

fn independent_fixture_decode(
    code: u8,
    exponent: u8,
) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code)] * 2.0f32.powi(i32::from(exponent) - 127)
}

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
    let outer_scales = [1.0f32, 2.0, 0.5];
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
    let outer_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
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
                    outer_scales: &outer_scales_alloc,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&biases_alloc),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &ids_alloc,
                    routes_per_token: NonZeroU32::new(routes_per_token).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: input_layout,
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
                value += input[input_row * K + inner] * decode_mxfp4(code, scales[scale_index], outer_scales[expert]);
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
    let outer_scales = [1.25f32];
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
    let outer_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
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
                    outer_scales: &outer_scales_alloc,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Dense,
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
                let weight = decode_mxfp4(code, scales[scale_index], outer_scales[0]);
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
    let outer_scales = [1.25f32];
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
    let outer_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
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
                    outer_scales: &outer_scales_alloc,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::SparseReadout {
                    b_rows: &readout_rows_alloc,
                },
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
                let weight = decode_mxfp4(code, scales[scale_index], outer_scales[0]);
                expected[input_row * READOUT_ROWS + readout_row] += input[input_row * K + inner] * weight;
            }
        }
    }
    (actual, expected)
}

fn run_tiny_output<B: Backend>(
    n: usize,
    k: usize,
) -> Vec<f32> {
    let input: Vec<f32> = (0..EXPERTS * k).map(|index| (index % 19) as f32 * 0.03 - 0.27).collect();
    let codes: Vec<u8> = (0..EXPERTS * n * k / 2)
        .map(|index| {
            let low = (index % 7 + 1) as u8;
            let high = ((index * 3 + 2) % 7 + 1) as u8;
            low | (high << 4)
        })
        .collect();
    let scales: Vec<u8> = (0..EXPERTS * n * k / 16).map(|index| 124 + (index % 5) as u8).collect();
    let outer_scales = [0.5f32, 1.0, 1.5];
    let biases: Vec<f32> = (0..EXPERTS * n).map(|index| index as f32 * 0.01 - 0.02).collect();
    let expert_ids = [2, 0, 1];
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        16,
        MicrofloatLayout::OutputInput,
        EXPERTS as u32,
        n as u32,
        k as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let outer_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
    let biases = alloc_allocation_with_data::<B, f32>(context.as_ref(), &biases);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &expert_ids);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), EXPERTS * n);
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
                    outer_scales: &outer_scales,
                    metadata,
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
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: EXPERTS as u32,
                n: n as u32,
                k: k as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, f32>(&output)
}

#[uzu_test]
fn tiny_microfloat_output_rows_cover_aligned_and_unaligned_k() {
    for n in 1..=3 {
        for k in [32, 256] {
            let expected = run_tiny_output::<Cpu>(n, k);
            for_each_non_cpu_backend!(|B| {
                let actual = run_tiny_output::<B>(n, k);
                assert_eq_float(&expected, &actual, 1e-4, "tiny direct MXFP4 rows");
            });
        }
    }
}

fn run_external_fixture<B: Backend>() -> Vec<f32> {
    const FIXTURE_N: usize = 8;
    const FIXTURE_K: usize = 32;

    let input = bf16_fixture(&GPT_OSS_INPUT_BF16);
    let biases = bf16_fixture(&GPT_OSS_BIASES_BF16);
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        32,
        MicrofloatLayout::OutputInput,
        1,
        FIXTURE_N as u32,
        FIXTURE_K as u32,
    )
    .unwrap();
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &GPT_OSS_GROUP32_CODES);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &GPT_OSS_GROUP32_SCALES);
    let outer_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &[0.5]);
    let biases = alloc_allocation_with_data::<B, f32>(context.as_ref(), &biases);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), FIXTURE_N);
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
                    outer_scales: &outer_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&biases),
                    soft_cap: Some(7.0),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(1).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: 1,
                n: FIXTURE_N as u32,
                k: FIXTURE_K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, f32>(&output)
}

#[uzu_test]
fn direct_group32_layout_matches_external_gpt_oss_bytes() {
    const FIXTURE_N: usize = 8;
    const FIXTURE_K: usize = 32;

    let input = bf16_fixture(&GPT_OSS_INPUT_BF16);
    let biases = bf16_fixture(&GPT_OSS_BIASES_BF16);
    let expected = (0..FIXTURE_N)
        .map(|row| {
            let dot = (0..FIXTURE_K).fold(0.0, |dot, column| {
                let packed = GPT_OSS_GROUP32_CODES[row * FIXTURE_K / 2 + column / 2];
                let code = if column.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                input[column].mul_add(0.5 * independent_fixture_decode(code, GPT_OSS_GROUP32_SCALES[row]), dot)
            });
            let value = dot + biases[row];
            7.0 * (value / 7.0).tanh()
        })
        .collect::<Vec<_>>();

    let actual = run_external_fixture::<Cpu>();
    assert_eq_float(&expected, &actual, 1e-5, "captured GPT-OSS direct MXFP4 CPU oracle");
    for_each_non_cpu_backend!(|B| {
        let actual = run_external_fixture::<B>();
        assert_eq_float(&expected, &actual, 1e-4, "captured GPT-OSS direct MXFP4 Metal output");
    });
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
    let outer_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![1.0; matrix_count as usize]);
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
                    outer_scales: &outer_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(expert_count).unwrap(),
                    input: ExpertInput::Tokens,
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
