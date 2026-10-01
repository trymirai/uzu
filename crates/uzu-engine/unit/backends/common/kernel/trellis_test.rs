use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::trellis,
            kernel::{TrellisTransform, mixing_order},
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
};

const MIXING_ENTRY_BOUND: f32 = 0.6;
const INPUT_BASE_BOUND: f32 = 3.0;
const INPUT_OUTLIER_EXPONENT: i32 = 3;
const COLUMN_CLASS_COUNT: usize = trellis::COLUMN_CLASS_COUNT as usize;
const TOKEN_STATISTICS_LEN: usize = trellis::TOKEN_STATISTICS_LEN as usize;
const ALL_ZERO_TOKEN: usize = 1;
const BATCHES: [u32; 4] = [1, 3, 17, 65];

pub(super) fn random_transform_data(
    rng: &mut SmallRng,
    columns: u32,
    batch: u32,
) -> (Vec<f32>, Vec<f32>, Vec<bf16>) {
    let mixing_order = mixing_order(columns);
    let signs = (0..columns).map(|_| [1.0, -1.0][rng.random_range(0..2)]).collect();
    let mixing =
        (0..mixing_order * mixing_order).map(|_| rng.random_range(-MIXING_ENTRY_BOUND..MIXING_ENTRY_BOUND)).collect();
    let input = (0..batch * columns)
        .map(|_| bf16::from_f32(rng.random_range(-INPUT_BASE_BOUND..INPUT_BASE_BOUND).powi(INPUT_OUTLIER_EXPONENT)))
        .collect();
    (signs, mixing, input)
}

fn run<B: Backend>(
    columns: u32,
    batch: u32,
    signs: &[f32],
    mixing: &[f32],
    input: &[bf16],
) -> (Vec<i8>, Vec<f32>) {
    let context = B::Context::new().expect("context");
    let transform = TrellisTransform::<B>::new(context.as_ref(), columns).expect("trellis transform").unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    // kept alive until the command buffer has run: the CPU backend reads them at submit time
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), input);
    let signs = create_buffer_with_data::<B, f32>(context.as_ref(), signs);
    let mixing = create_buffer_with_data::<B, f32>(context.as_ref(), mixing);
    let rotated = transform.encode(&input, &signs, &mixing, batch, &mut command_buffer).expect("encode");
    // the rotated input lives in command buffer scratch, so copy it out before submitting
    let mut activations = create_buffer::<B, i8>(context.as_ref(), (batch * columns) as usize);
    let mut statistics = create_buffer::<B, f32>(context.as_ref(), TOKEN_STATISTICS_LEN * batch as usize);
    command_buffer.encode_copy(&rotated.activations, &mut activations);
    command_buffer.encode_copy(&rotated.token_statistics, &mut statistics);
    drop(rotated);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    (buffer_to_vec::<B, i8>(&activations), buffer_to_vec::<B, f32>(&statistics))
}

#[rstest]
#[test_attr(uzu_test)]
fn transform_matches_cpu(#[values(5120, 6144, 17408)] columns: u32) {
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    for batch in BATCHES {
        let (signs, mixing, mut input) = random_transform_data(&mut rng, columns, batch);
        if batch as usize > ALL_ZERO_TOKEN {
            input[ALL_ZERO_TOKEN * columns as usize..][..columns as usize].fill(bf16::ZERO);
        }
        let (expected_activations, expected_statistics) = run::<Cpu>(columns, batch, &signs, &mixing, &input);
        for_each_non_cpu_backend!(|B| {
            let (activations, statistics) = run::<B>(columns, batch, &signs, &mixing, &input);
            let message = format!("{} columns {columns} batch {batch}", std::any::type_name::<B>());
            for (index, (&actual, &expected)) in activations.iter().zip(&expected_activations).enumerate() {
                assert!((i32::from(actual) - i32::from(expected)).abs() <= 1, "code {index}, {message}");
            }
            let statistics_rows =
                statistics.chunks(TOKEN_STATISTICS_LEN).zip(expected_statistics.chunks(TOKEN_STATISTICS_LEN));
            for (token, (statistics, expected)) in statistics_rows.enumerate() {
                let (class_sums, scale_and_zeros) = statistics.split_at(COLUMN_CLASS_COUNT);
                let mut sums_from_codes = [0.0f32; COLUMN_CLASS_COUNT];
                for (column, &code) in activations[token * columns as usize..][..columns as usize].iter().enumerate() {
                    sums_from_codes[column % COLUMN_CLASS_COUNT] += f32::from(code);
                }
                assert_eq!(class_sums, sums_from_codes, "class sums, token {token}, {message}");
                let expected_scale = expected[COLUMN_CLASS_COUNT];
                let relative_error = (scale_and_zeros[0] - expected_scale).abs() / expected_scale.abs().max(1e-6);
                assert!(relative_error < 1e-3, "scale, token {token}, {message}");
                assert_eq!(scale_and_zeros[1..], [0.0; 3], "zeros, token {token}, {message}");
            }
        });
    }
}
