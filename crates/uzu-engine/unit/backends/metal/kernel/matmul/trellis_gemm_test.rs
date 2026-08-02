use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, MatmulShape,
                TRELLIS_CODES_TAIL_BYTES, TrellisFormat,
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            gpu_types::gemm::GemmTiling,
            kernel::matmul::{GemmKernel, MatmulOutputWork, gemm::GemmEngine},
        },
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
const V4_T6: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 6,
    restart_columns: Some(64),
};
const V4_T7: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 7,
    restart_columns: Some(64),
};
const V4_T6_R40: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 6,
    restart_columns: Some(40),
};
const V4_T8_NO_RESTART: TrellisFormat = TrellisFormat {
    vector_width: 4,
    transition_bits: 8,
    restart_columns: None,
};

// Tape 2: each block's bits reversed, LSB first; state i is the LE 16-bit field at bit B - 16 - i*T.
fn decode_state(
    format: TrellisFormat,
    row: &[u8],
    columns: usize,
    column: usize,
) -> u16 {
    let (width, transition_bits) = (format.vector_width as usize, format.transition_bits as usize);
    let block_columns = format.restart_columns.map_or(columns, |restart| restart as usize);
    let block_bits = 16 + (block_columns / width - 1) * transition_bits;
    let start_bit = (column / block_columns + 1) * block_bits - 16 - column % block_columns / width * transition_bits;
    (start_bit..start_bit + 16).rev().fold(0u32, |state, bit| (state << 1) | u32::from(row[bit / 8] >> (bit % 8) & 1))
        as u16
}

fn decode_level(
    format: TrellisFormat,
    row: &[u8],
    columns: usize,
    column: usize,
) -> i32 {
    let state = u32::from(decode_state(format, row, columns, column));
    let hash = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
    let hash = (hash ^ hash >> 16).wrapping_mul(0x85EB_CA6B);
    let hash_byte = ((hash ^ hash >> 16) >> (8 * (column % format.vector_width as usize))) & 0xFF;
    let field_sum: u32 = (0..4).map(|field| (hash_byte >> (2 * field)) & 3).sum();
    (8 * field_sum + ((3 * (hash_byte & 15)) & 15)) as i32 - 54
}

type MetalBuffer = <Metal as Backend>::GlobalGpuBuffer;
type MetalMatmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel;

const CODEBOOK: [f32; 5] = [0.0123, 0.3, -0.25, 0.125, -0.0625];
const PADDING: usize = 24;
const SENTINEL: u16 = 0x7F7F;
const RELATIVE_TOLERANCE: f32 = 1e-2;
// Cases are ordered as (m, n, k).
const CASES: [(usize, usize, usize); 9] = [
    (1, 80, 5120),
    (7, 80, 5120),
    (17, 80, 5120),
    (512, 80, 5120),
    (16, 4096, 2048),
    (16, 128, 64),
    (32, 128, 64),
    (1, 6, 64),
    (2048, 2048, 128),
];

fn run_projection(
    context: &MetalContext,
    format: TrellisFormat,
    dimensions: (usize, usize, usize),
    forced_plan: Option<(GemmTiling, u32)>,
) {
    let (m, n, k) = dimensions;
    let mut rng = SmallRng::seed_from_u64((m * k + n) as u64);
    let row_bytes = format.code_row_bytes(k as u32) as usize;
    let stored: Vec<u8> = (0..n * row_bytes).map(|_| rng.random()).collect();
    let row_scales: Vec<f32> = (0..n).map(|_| rng.random_range(0.001..0.01)).collect();
    let mut values: Vec<i8> = (0..m * k).map(|_| rng.random_range(-127..=127)).collect();
    if m > 1 {
        // Exercise the zero-activation path in the epilogue.
        values[k..2 * k].fill(0);
    }
    if m > 3 {
        for (column, value) in values[3 * k..4 * k].iter_mut().enumerate() {
            *value = if column % 2 == 0 {
                -128
            } else {
                127
            };
        }
    }
    let group_sums_host: Vec<f32> = values
        .chunks_exact(k)
        .flat_map(|row| {
            (0..4).map(|group| row.iter().skip(group).step_by(4).fold(0.0, |sum, &value| sum + f32::from(value)))
        })
        .collect();
    let activation_scales_host: Vec<f32> = (0..m).map(|_| rng.random_range(0.001..0.01)).collect();
    let activations = create_buffer_with_data::<Metal, i8>(context, &values);
    let group_sums = create_buffer_with_data::<Metal, f32>(context, &group_sums_host);
    let activation_scales = create_buffer_with_data::<Metal, f32>(context, &activation_scales_host);
    // The GEMM may read a word past the last row's codes; the tail must not change the result.
    let mut codes_with_tail = stored.clone();
    codes_with_tail.resize(stored.len() + TRELLIS_CODES_TAIL_BYTES, 0xA5);
    let codes = create_buffer_with_data::<Metal, u8>(context, &codes_with_tail);
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
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    if let Some((tiling, split_k)) = forced_plan {
        let shape = MatmulShape::from_arguments(&arguments);
        let mut gemm = GemmKernel::new(DataType::BF16, DataType::BF16, DataType::BF16);
        let mut plan = gemm.select_plan_for_engine(&shape, GemmEngine::Mxu, context).unwrap();
        plan.tiling = tiling;
        plan.split_k = split_k;
        let output_work = MatmulOutputWork::new(context, DataType::BF16, DataType::BF16).unwrap();
        gemm.encode_plan(arguments, plan, &output_work, &mut command_buffer).unwrap();
    } else {
        let mut matmul =
            <MetalMatmul as MatmulKernel>::new(context, DataType::BF16, DataType::BF16, DataType::BF16).unwrap();
        matmul.encode(arguments, &mut command_buffer).unwrap();
    }
    submit_command_buffer(command_buffer);
    let decoded_levels: Vec<i32> = stored
        .chunks_exact(row_bytes)
        .flat_map(|codes| (0..k).map(|column| decode_level(format, codes, k, column)))
        .collect();
    let actual = buffer_to_vec::<Metal, u16>(context, &output);
    let stride = n + PADDING;
    for (index, &got) in actual.iter().enumerate() {
        let token = index / stride;
        let row = index % stride;
        if row < n {
            let row_values = &values[token * k..(token + 1) * k];
            let levels = &decoded_levels[row * k..(row + 1) * k];
            let level_dot: i32 = levels.iter().zip(row_values).map(|(&level, &value)| level * i32::from(value)).sum();
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
#[case::v4_t6(V4_T6)]
#[case::v4_t7(V4_T7)]
#[case::v4_t6_r40(V4_T6_R40)]
#[case::v4_t8_no_restart(V4_T8_NO_RESTART)]
fn trellis_projection_matches_cpu_reference(#[case] format: TrellisFormat) {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    // A projection's K is a whole number of restart blocks.
    let whole_blocks = |k: usize| k.is_multiple_of(format.restart_columns.unwrap_or(1) as usize);
    for dimensions in CASES.into_iter().filter(|&(_, _, k)| whole_blocks(k)) {
        run_projection(&context, format, dimensions, None);
    }
    // The forced plans' K are whole 64-column blocks.
    if whole_blocks(64) {
        run_projection(&context, format, (129, 132, 1024), Some((GemmTiling::Tile128x128x256_Simdgroups4x4, 1)));
        run_projection(&context, format, (129, 132, 64), Some((GemmTiling::Tile128x128x256_Simdgroups4x4, 1)));
        run_projection(&context, format, (17, 80, 192), Some((GemmTiling::Tile32x64x256_Simdgroups2x2, 1)));
        run_projection(&context, format, (17, 80, 192), Some((GemmTiling::Tile16x32x256_Simdgroups1x1, 1)));
        run_projection(&context, format, (7, 80, 192), Some((GemmTiling::Tile16x32x256_Simdgroups1x1, 1)));
    }
}
