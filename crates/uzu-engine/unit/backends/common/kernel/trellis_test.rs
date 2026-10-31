use futures::executor::block_on;
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::trellis::COLUMN_GROUP_COUNT,
            kernel::{TrellisTransformKernel, mixing_dimension},
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

const MIXING_ENTRY_BOUND: f32 = 0.6;
const INPUT_BASE_BOUND: f32 = 3.0;
const INPUT_OUTLIER_EXPONENT: i32 = 3;
const ALL_ZERO_TOKEN: usize = 1;
const BATCHES: [u32; 5] = [1, 8, 16, 32, 1024];

fn random_transform_data(
    rng: &mut SmallRng,
    columns: u32,
    batch: u32,
) -> (Vec<f32>, Vec<f32>, Vec<bf16>) {
    let mixing_dimension = mixing_dimension(columns);
    let rht_factors = (0..columns).map(|_| [1.0, -1.0][rng.random_range(0..2)]).collect();
    let mixing = (0..mixing_dimension * mixing_dimension)
        .map(|_| rng.random_range(-MIXING_ENTRY_BOUND..MIXING_ENTRY_BOUND))
        .collect();
    let input = (0..batch * columns)
        .map(|_| bf16::from_f32(rng.random_range(-INPUT_BASE_BOUND..INPUT_BASE_BOUND).powi(INPUT_OUTLIER_EXPONENT)))
        .collect();
    (rht_factors, mixing, input)
}

fn run<B: Backend>(
    columns: u32,
    batch: u32,
    rht_factors: &[f32],
    mixing: &[f32],
    input: &[bf16],
) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let context = block_on(B::Context::new()).expect("context");
    let transform =
        <B::Kernels as Kernels>::TrellisTransformKernel::new(context.as_ref(), columns).expect("trellis transform");
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    // kept alive until the command buffer has run: the CPU backend reads them at submit time
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), input);
    let rht_factors = create_buffer_with_data::<B, f32>(context.as_ref(), rht_factors);
    let mixing = create_buffer_with_data::<B, f32>(context.as_ref(), mixing);
    let mut activations = create_buffer::<B, i8>(context.as_ref(), (batch * columns) as usize);
    let mut column_group_sums = create_buffer::<B, f32>(context.as_ref(), COLUMN_GROUP_COUNT as usize * batch as usize);
    let mut scales = create_buffer::<B, f32>(context.as_ref(), batch as usize);
    transform.encode(
        &input,
        &rht_factors,
        &mixing,
        &mut activations,
        &mut column_group_sums,
        &mut scales,
        batch,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    (
        buffer_to_vec::<B, i8>(&context, &activations),
        buffer_to_vec::<B, f32>(&context, &column_group_sums),
        buffer_to_vec::<B, f32>(&context, &scales),
    )
}

#[rstest]
#[test_attr(uzu_test)]
fn transform_matches_cpu(
    #[values(2048, 2560, 3072, 3584, 4096, 5120, 6144, 6656, 7168, 9216, 10240, 12288, 14336, 17408, 18944, 19968)]
    columns: u32
) {
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    for batch in BATCHES {
        let (rht_factors, mixing, mut input) = random_transform_data(&mut rng, columns, batch);
        if batch as usize > ALL_ZERO_TOKEN {
            input[ALL_ZERO_TOKEN * columns as usize..][..columns as usize].fill(bf16::ZERO);
        }
        let (expected_activations, _, expected_scales) = run::<Cpu>(columns, batch, &rht_factors, &mixing, &input);
        for_each_backend!(|B| {
            let (activations, column_group_sums, scales) = run::<B>(columns, batch, &rht_factors, &mixing, &input);
            let message = format!("{} columns {columns} batch {batch}", std::any::type_name::<B>());
            for (index, (&actual, &expected)) in activations.iter().zip(&expected_activations).enumerate() {
                assert!((i32::from(actual) - i32::from(expected)).abs() <= 1, "code {index}, {message}");
            }
            for (token, column_group_sums) in column_group_sums.chunks(COLUMN_GROUP_COUNT as usize).enumerate() {
                let mut sums_from_codes = [0.0f32; COLUMN_GROUP_COUNT as usize];
                for (column, &code) in activations[token * columns as usize..][..columns as usize].iter().enumerate() {
                    sums_from_codes[column % COLUMN_GROUP_COUNT as usize] += f32::from(code);
                }
                assert_eq!(column_group_sums, sums_from_codes, "column group sums, token {token}, {message}");
                let expected_scale = expected_scales[token];
                let relative_error = (scales[token] - expected_scale).abs() / expected_scale.abs().max(1e-6);
                assert!(relative_error < 1e-3, "scale, token {token}, {message}");
            }
        });
    }
}

/// Trellis projections checked against the levels their stored codes decode to; backends with their own plans reuse it.
#[cfg(any(backend = "metal", backend = "vulkan"))]
pub mod projection {
    use std::sync::Arc;

    use half::bf16;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    use rstest::rstest;
    use uzu_engine_macros::uzu_test;

    use crate::{
        backends::common::{
            Backend, CommandBuffer, Context, Kernels,
            kernel::matmul::{
                MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulOutput, TRELLIS_CODES_TAIL_BYTES,
                TrellisFormat,
            },
        },
        data_type::DataType,
        tests::helpers::{
            buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend, submit_command_buffer,
        },
    };
    #[cfg(backend = "metal")]
    use crate::{backends::metal::Metal, utils::downcast::downcast_ref};

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

    pub fn format(
        vector_width: u32,
        transition_bits: u32,
        restart_columns: Option<u32>,
    ) -> TrellisFormat {
        TrellisFormat {
            vector_width,
            transition_bits,
            restart_columns,
        }
    }

    /// Whether `k` is a whole number of the format's restart blocks, as a projection's K is.
    pub fn whole_blocks(
        format: TrellisFormat,
        k: usize,
    ) -> bool {
        k.is_multiple_of(format.restart_columns.unwrap_or(1) as usize)
    }

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
        let start_bit =
            (column / block_columns + 1) * block_bits - 16 - column % block_columns / width * transition_bits;
        (start_bit..start_bit + 16)
            .rev()
            .fold(0u32, |state, bit| (state << 1) | u32::from(row[bit / 8] >> (bit % 8) & 1)) as u16
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

    /// Encodes one projection of random codes with `encode` and checks every output, and that padding is untouched.
    pub fn check_projection<B: Backend>(
        context: &Arc<B::Context>,
        format: TrellisFormat,
        (m, n, k): (usize, usize, usize),
        encode: impl FnOnce(
            MatmulArguments<
                '_,
                B,
                &B::GlobalGpuBuffer,
                &B::GlobalGpuBuffer,
                &mut B::GlobalGpuBuffer,
                &B::GlobalGpuBuffer,
            >,
            &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        ),
    ) {
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
        let activations = create_buffer_with_data::<B, i8>(context, &values);
        let group_sums = create_buffer_with_data::<B, f32>(context, &group_sums_host);
        let activation_scales = create_buffer_with_data::<B, f32>(context, &activation_scales_host);
        // A backend may read past the last row's codes; the tail must not change the result.
        let mut codes_with_tail = stored.clone();
        codes_with_tail.resize(stored.len() + TRELLIS_CODES_TAIL_BYTES, 0xA5);
        let codes = create_buffer_with_data::<B, u8>(context, &codes_with_tail);
        let row_scales_buffer = create_buffer_with_data::<B, f32>(context, &row_scales);
        let codebook = create_buffer_with_data::<B, f32>(context, &CODEBOOK);
        let mut output = create_buffer_with_data::<B, u16>(context, &vec![SENTINEL; m * (n + PADDING)]);
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
            gather_indices: None,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        encode(arguments, &mut command_buffer);
        submit_command_buffer(command_buffer);
        let decoded_levels: Vec<i32> = stored
            .chunks_exact(row_bytes)
            .flat_map(|codes| (0..k).map(|column| decode_level(format, codes, k, column)))
            .collect();
        let actual = buffer_to_vec::<B, u16>(context, &output);
        let stride = n + PADDING;
        for (index, &got) in actual.iter().enumerate() {
            let token = index / stride;
            let row = index % stride;
            if row < n {
                let row_values = &values[token * k..(token + 1) * k];
                let levels = &decoded_levels[row * k..(row + 1) * k];
                let level_dot: i32 =
                    levels.iter().zip(row_values).map(|(&level, &value)| level * i32::from(value)).sum();
                let sums = &group_sums_host[token * 4..token * 4 + 4];
                let terms = [
                    f64::from(level_dot) * f64::from(CODEBOOK[0]),
                    f64::from(sums[0]) * f64::from(CODEBOOK[1]),
                    f64::from(sums[1]) * f64::from(CODEBOOK[2]),
                    f64::from(sums[2]) * f64::from(CODEBOOK[3]),
                    f64::from(sums[3]) * f64::from(CODEBOOK[4]),
                ];
                let scale = f64::from(row_scales[row]) * f64::from(activation_scales_host[token]);
                let want = bf16::from_f32((terms.iter().sum::<f64>() * scale) as f32).to_f32();
                // Five FP32 terms and two scales may round in a different order or contract. Bound that roundoff
                // before BF16 rounding, including near cancellation; zero input still has an exact zero bound.
                let roundoff =
                    8.0 * f64::from(f32::EPSILON) * terms.iter().map(|term| term.abs()).sum::<f64>() * scale.abs();
                let got = bf16::from_bits(got).to_f32();
                assert!(
                    f64::from((got - want).abs()) <= f64::from(RELATIVE_TOLERANCE * want.abs()) + roundoff,
                    "{format:?} ({m}, {n}, {k}) token {token} row {row}: got {got}, want {want}, roundoff {roundoff}"
                );
            } else {
                assert_eq!(got, SENTINEL, "{format:?} M {m} N {n} K {k}, token {token}, row {row}");
            }
        }
    }

    #[rstest]
    #[test_attr(uzu_test)]
    #[case::v4_t8(format(4, 8, Some(64)))]
    #[case::v2_t6(format(2, 6, None))]
    #[case::v2_t4(format(2, 4, None))]
    #[case::v4_t6(format(4, 6, Some(64)))]
    #[case::v4_t7(format(4, 7, Some(64)))]
    #[case::v4_t6_r40(format(4, 6, Some(40)))]
    #[case::v4_t8_no_restart(format(4, 8, None))]
    fn trellis_projection_matches_tape_levels(#[case] format: TrellisFormat) {
        for_each_non_cpu_backend!(|B| trellis_projection_matches_tape_levels_impl::<B>(format));
    }

    fn trellis_projection_matches_tape_levels_impl<B: Backend>(format: TrellisFormat) {
        let context = create_context::<B>();
        // Metal plans trellis projections on MXU only.
        #[cfg(backend = "metal")]
        if downcast_ref::<<Metal as Backend>::Context>(&*context).is_some_and(|context| !context.supports_mxu) {
            return;
        }
        for dimensions in CASES.into_iter().filter(|&(_, _, k)| whole_blocks(format, k)) {
            check_projection::<B>(&context, format, dimensions, |arguments, command_buffer| {
                let mut matmul = <<B as Backend>::Kernels as Kernels>::MatmulKernel::new(
                    &context,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::BF16,
                )
                .unwrap();
                matmul.encode(arguments, command_buffer).unwrap();
            });
        }
    }
}
