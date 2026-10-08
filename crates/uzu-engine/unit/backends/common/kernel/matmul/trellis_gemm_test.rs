#![cfg(backend = "metal")]

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, TrellisFormat,
            },
        },
        metal::{GemmEngine, Metal, MetalContext},
    },
    data_type::DataType,
    tests::{
        helpers::{buffer_to_vec, create_buffer_with_data, submit_command_buffer},
        util::shared_metal_context,
    },
};

const V4_T8: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 8,
    restart_columns: Some(64),
};
const V2_T6: TrellisFormat = TrellisFormat {
    vector_width: 2,
    transition_bits: 6,
    restart_columns: None,
};
const V2_T4: TrellisFormat = TrellisFormat {
    vector_width: 2,
    transition_bits: 4,
    restart_columns: None,
};

fn decode_state(
    format: TrellisFormat,
    row: &[u8],
    column: usize,
) -> u16 {
    let width = format.vector_width as usize;
    let (block, local) =
        format.restart_columns.map_or((0, column), |restart| (column / restart as usize, column % restart as usize));
    let block_bytes = format.restart_columns.map_or(0, |restart| row_bytes(format, restart as usize));
    let start_bit = block * block_bytes * 8 + local / width * format.transition_bits as usize;
    (start_bit..start_bit + 16).fold(0u32, |state, bit| (state << 1) | u32::from(row[bit / 8] >> (7 - bit % 8) & 1))
        as u16
}

fn row_bytes(
    format: TrellisFormat,
    columns: usize,
) -> usize {
    let block_columns = format.restart_columns.map_or(columns, |restart| restart as usize);
    let transitions = block_columns / format.vector_width as usize - 1;
    let bytes_per_block = (16 + transitions * format.transition_bits as usize).div_ceil(8);
    columns.div_ceil(block_columns) * bytes_per_block
}

fn decode_level(
    format: TrellisFormat,
    row: &[u8],
    column: usize,
) -> i32 {
    let state = u32::from(decode_state(format, row, column));
    let hash = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
    let hash = (hash ^ hash >> 16).wrapping_mul(0x85EB_CA6B);
    let hash_byte = ((hash ^ hash >> 16) >> (8 * (column % format.vector_width as usize))) & 0xFF;
    let field_sum: u32 = (0..4).map(|field| (hash_byte >> (2 * field)) & 3).sum();
    (8 * field_sum + ((3 * (hash_byte & 15)) & 15)) as i32 - 54
}

type MetalBuffer = <Metal as Backend>::GlobalBuffer;
type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

const CODEBOOK: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];
const PADDING: usize = 24;
const SENTINEL: u16 = 0x7F7F;
const RELATIVE_TOLERANCE: f32 = 1e-2;
// Cases are ordered as (m, n, k).
const CASES: [(usize, usize, usize); 7] =
    [(1, 80, 5120), (17, 80, 5120), (512, 80, 5120), (16, 128, 64), (32, 128, 64), (1, 6, 64), (2048, 2048, 128)];

fn run_projection(
    context: &MetalContext,
    format: TrellisFormat,
    dimensions: (usize, usize, usize),
    engine: Option<GemmEngine>,
    cancellation_case: bool,
) {
    let (m, n, k) = dimensions;
    let mut rng = SmallRng::seed_from_u64((m * k + n) as u64);
    let mut stored: Vec<u8> = (0..n * row_bytes(format, k)).map(|_| rng.random()).collect();
    let mut row_scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.001..0.01)).collect();
    let mut values: Vec<i8> = (0..m * k).map(|_| rng.random_range(-127..=127)).collect();
    if cancellation_case {
        stored.fill(0xFF);
        row_scales.fill(1.0);
        values[..k].fill(127);
        values[k..2 * k].fill(-128);
        values[2 * k..2 * k + k / 2].fill(127);
        values[2 * k + k / 2..3 * k].fill(-127);
        values[3 * k - 1] = -126;
    } else if m > 1 {
        // Exercise the zero-activation path in the epilogue.
        values[k..2 * k].fill(0);
    }
    let group_sums_host: Vec<f32> = values
        .chunks_exact(k)
        .flat_map(|row| {
            (0..4).map(|group| row.iter().skip(group).step_by(4).fold(0.0, |sum, &value| sum + f32::from(value)))
        })
        .collect();
    let mut activation_scales_host: Vec<f32> = (0..m).map(|_| rng.random_range(0.001..0.01)).collect();
    if cancellation_case {
        activation_scales_host.fill(1.0);
    }
    let activations = create_buffer_with_data::<Metal, i8>(context, &values);
    let group_sums = create_buffer_with_data::<Metal, f32>(context, &group_sums_host);
    let activation_scales = create_buffer_with_data::<Metal, f32>(context, &activation_scales_host);
    let codes = create_buffer_with_data::<Metal, u8>(context, &stored);
    let row_scales_buffer = create_buffer_with_data::<Metal, f32>(context, &row_scales);
    let codebook = create_buffer_with_data::<Metal, f32>(context, &CODEBOOK);
    let mut output = create_buffer_with_data::<Metal, u16>(context, &vec![SENTINEL; m * (n + PADDING)]);
    let arguments = MatmulArguments {
        a: MatmulA::Trellis {
            values: &activations,
            column_group_sums: &group_sums,
            scales: &activation_scales,
        },
        b: MatmulB::Trellis {
            codes: &codes,
            row_scales: &row_scales_buffer,
            codebook: &codebook,
            format,
        },
        b_leading_dimension: None,
        b_transpose: true,
        output: MatmulOutput {
            values: &mut output,
            row_stride: Some((n + PADDING) as u32),
            ops: MatmulDOps::none(),
        },
        gather_indices: None::<&MetalBuffer>,
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let mut matmul =
        <MetalMatmul as MatmulKernel>::new(context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    if let Some(engine) = engine {
        matmul.encode_with_gemm_engine(arguments, engine, &mut command_buffer).unwrap();
    } else {
        matmul.encode(arguments, &mut command_buffer).unwrap();
    }
    submit_command_buffer(command_buffer);
    let decoded_levels: Vec<i32> = stored
        .chunks_exact(row_bytes(format, k))
        .flat_map(|codes| (0..k).map(|column| decode_level(format, codes, column)))
        .collect();
    let actual = buffer_to_vec::<Metal, u16>(&output);
    let stride = n + PADDING;
    for (index, &got) in actual.iter().enumerate() {
        let token = index / stride;
        let row = index % stride;
        if row < n {
            let row_values = &values[token * k..(token + 1) * k];
            let levels = &decoded_levels[row * k..(row + 1) * k];
            let level_dot: i32 = levels.iter().zip(row_values).map(|(&level, &value)| level * i32::from(value)).sum();
            if cancellation_case && token < 2 {
                assert!(level_dot.abs() > (1 << 24));
            }
            if cancellation_case && token == 2 {
                assert!(level_dot.abs() < 112, "large positive and negative sums should cancel");
            }
            let sums = &group_sums_host[token * 4..token * 4 + 4];
            // A non-dyadic offset exposes rounding differences in the GPU dot.
            let offsets_dot =
                (((sums[0] * CODEBOOK[1]) + sums[1] * CODEBOOK[2]) + sums[2] * CODEBOOK[3]) + sums[3] * CODEBOOK[4];
            let dot = (level_dot as f32).mul_add(CODEBOOK[0], offsets_dot);
            let want = bf16::from_f32(dot * row_scales[row] * activation_scales_host[token]).to_f32();
            let got = bf16::from_bits(got).to_f32();
            assert!(
                (got - want).abs() <= RELATIVE_TOLERANCE * want.abs(),
                "{format:?} ({m}, {n}, {k}) token {token} row {row}: got {got}, want {want}"
            );
        } else {
            assert_eq!(got, SENTINEL, "{format:?} M {m} N {n} K {k}, token {token}, row {row}");
        }
    }
}

#[rstest]
#[test_attr(uzu_test)]
#[case::v4_t8(V4_T8)]
#[case::v2_t6(V2_T6)]
#[case::v2_t4(V2_T4)]
fn trellis_projection_matches_cpu_reference(#[case] format: TrellisFormat) {
    let context = shared_metal_context();
    println!("Trellis GEMM correctness on {} (supports_mxu={})", context.device_name, context.supports_mxu);
    run_projection(&context, format, (17, 80, 64), None, false);
    for (m, n, k) in CASES {
        run_projection(&context, format, (m, n, k), Some(GemmEngine::Simdgroup), false);
        if context.supports_mxu {
            run_projection(&context, format, (m, n, k), Some(GemmEngine::Mxu), false);
        }
    }
    run_projection(&context, format, (1, 6, 5120), Some(GemmEngine::Simdgroup), false);
    run_projection(&context, format, (3, 4, 32_768), Some(GemmEngine::Simdgroup), true);
}
