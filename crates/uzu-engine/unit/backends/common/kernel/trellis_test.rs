use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::trellis::COLUMN_GROUP_COUNT,
            kernel::{TrellisTransform, mixing_dimension},
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend},
};

const MIXING_ENTRY_BOUND: f32 = 0.6;
const INPUT_BASE_BOUND: f32 = 3.0;
const INPUT_OUTLIER_EXPONENT: i32 = 3;
const ALL_ZERO_TOKEN: usize = 1;
const BATCHES: [u32; 4] = [1, 3, 17, 65];

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
    let context = B::Context::new().expect("context");
    let transform = TrellisTransform::<B>::new(context.as_ref(), columns).expect("trellis transform").unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    // kept alive until the command buffer has run: the CPU backend reads them at submit time
    let input = create_buffer_with_data::<B, bf16>(context.as_ref(), input);
    let rht_factors = create_buffer_with_data::<B, f32>(context.as_ref(), rht_factors);
    let mixing = create_buffer_with_data::<B, f32>(context.as_ref(), mixing);
    let rotated = transform.encode(&input, &rht_factors, &mixing, batch, &mut command_buffer).expect("encode");
    // the rotated input lives in command buffer scratch, so copy it out before submitting
    let mut activations = create_buffer::<B, i8>(context.as_ref(), (batch * columns) as usize);
    let mut column_group_sums = create_buffer::<B, f32>(context.as_ref(), COLUMN_GROUP_COUNT as usize * batch as usize);
    let mut scales = create_buffer::<B, f32>(context.as_ref(), batch as usize);
    command_buffer.encode_copy(&rotated.activations, &mut activations);
    command_buffer.encode_copy(&rotated.column_group_sums, &mut column_group_sums);
    command_buffer.encode_copy(&rotated.scales, &mut scales);
    drop(rotated);
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    (
        buffer_to_vec::<B, i8>(&activations),
        buffer_to_vec::<B, f32>(&column_group_sums),
        buffer_to_vec::<B, f32>(&scales),
    )
}

#[rstest]
#[test_attr(uzu_test)]
fn transform_matches_cpu(#[values(5120, 6144, 17408)] columns: u32) {
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
