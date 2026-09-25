use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::kernel::mirai_s::mixing_order,
    encodable_block::linear::mirai_s::{
        codebook_table,
        tests::{AFFINE, package_codebook},
        trellis_levels,
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        util::shared_metal_context,
    },
};

/// Level of every column of a row: V4 restarts a little-endian 16-bit state every 64 columns and shifts in one
/// byte per 4 columns; V2 states are the 16-bit big-endian windows at bit `transition_bits * group`.
fn row_levels(
    codec: TrellisCodec,
    row: &[u8],
    columns: usize,
) -> Vec<i32> {
    let bit = |index: usize| ((row[index / 8] >> (7 - index % 8)) & 1) as u32;
    (0..columns)
        .map(|column| match codec {
            TrellisCodec::Vector4Restart64 => {
                let block = &row[column / 64 * 17..][..17];
                let mut state = block[0] as u32 | (block[1] as u32) << 8;
                for group in 1..=column % 64 / 4 {
                    state = ((state << 8) | block[group + 1] as u32) & 0xFFFF;
                }
                trellis_levels(state)[column % 4]
            },
            TrellisCodec::Vector2Transition6 | TrellisCodec::Vector2Transition4 => {
                let start = column / 2 * codec.transition_bits() as usize;
                let state = (0..16).fold(0, |state, offset| (state << 1) | bit(start + offset));
                trellis_levels(state)[column % 2]
            },
        })
        .collect()
}

fn transform_reference(
    input: &[f32],
    signs: &[f32],
    mixing: &[f32],
) -> (Vec<i8>, f32, [f32; 4]) {
    let columns = input.len();
    let order = mixing_order(columns as u32) as usize;
    let power = columns / order;
    let normalization = if power == 2048 {
        std::f32::consts::FRAC_1_SQRT_2
    } else {
        1.0
    } / 32.0;
    let mut transformed = vec![0.0f32; columns];
    for q_out in 0..order {
        let mut values: Vec<f32> = (0..power)
            .map(|h| {
                (0..order).fold(0.0f32, |value, q| {
                    (input[h * order + q] * signs[h * order + q]).mul_add(mixing[q_out * order + q], value)
                })
            })
            .collect();
        let mut stride = 1;
        while stride < power {
            for low in (0..power).filter(|h| h & stride == 0) {
                let (a, b) = (values[low], values[low + stride]);
                values[low] = a + b;
                values[low + stride] = a - b;
            }
            stride *= 2;
        }
        for h in 0..power {
            transformed[h * order + q_out] = bf16::from_f32(values[h] * normalization).to_f32();
        }
    }
    let maximum = transformed.iter().fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let scale = if maximum > 0.0 {
        maximum / 127.0
    } else {
        1.0
    };
    let quantized: Vec<i8> =
        transformed.iter().map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8).collect();
    let sums = std::array::from_fn(|class| quantized.iter().skip(class).step_by(4).map(|&value| value as f32).sum());
    (quantized, scale, sums)
}

#[rstest]
#[test_attr(uzu_test)]
fn transform_matches_reference(#[values(5120, 6144, 17408)] columns: u32) {
    let context = shared_metal_context();
    let transform = MetalMiraiSTransform::new(&context, columns).unwrap().unwrap();
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    let order = mixing_order(columns);
    let signs: Vec<f32> = (0..columns).map(|_| [1.0, -1.0][rng.random_range(0..2)]).collect();
    let mixing: Vec<f32> = (0..order * order).map(|_| rng.random_range(-0.6f32..0.6)).collect();
    for batch in [1u32, 3, 17] {
        let input: Vec<bf16> =
            (0..batch * columns).map(|_| bf16::from_f32(rng.random_range(-3.0f32..3.0).powi(3))).collect();
        let mut encoder = Encoder::<Metal>::new(&context).unwrap();
        let rotated = transform
            .encode(
                &alloc_allocation_with_data::<Metal, bf16>(&context, &input),
                &alloc_allocation_with_data::<Metal, f32>(&context, &signs),
                &alloc_allocation_with_data::<Metal, f32>(&context, &mixing),
                batch,
                &mut encoder,
            )
            .unwrap();
        // the rotated input lives in encoder scratch, so copy it out before submitting
        let mut activations = alloc_allocation::<Metal, i8>(&context, (batch * columns) as usize);
        let mut statistics = alloc_allocation::<Metal, f32>(&context, 8 * batch as usize);
        encoder.encode_copy(&rotated.activations, ..activations.size(), &mut activations, ..);
        encoder.encode_copy(&rotated.token_statistics, .., &mut statistics, ..);
        drop(rotated);
        encoder.end_encoding().submit().wait_until_completed().unwrap();
        let activations = allocation_to_vec::<Metal, i8>(&activations);
        let statistics = allocation_to_vec::<Metal, f32>(&statistics);
        for token in 0..batch as usize {
            let row = &input[token * columns as usize..][..columns as usize];
            let row: Vec<f32> = row.iter().map(|value| value.to_f32()).collect();
            let (expected, scale, expected_sums) = transform_reference(&row, &signs, &mixing);
            assert_eq!(&activations[token * columns as usize..][..columns as usize], expected, "token {token}");
            let record = &statistics[token * 8..][..8];
            assert_eq!(record[..4], expected_sums, "token {token}");
            assert_eq!(record[4].to_bits(), scale.to_bits(), "token {token}");
            assert_eq!(record[5..], [0.0; 3], "token {token}");
        }
    }
}

/// Encodes one projection kernel: (activations, token statistics, batch, (output, byte offset), output stride).
type Encode<'a> = Box<
    dyn Fn(&Allocation<Metal>, &Allocation<Metal>, u32, (&mut Allocation<Metal>, usize), u32, &mut Encoder<Metal>) + 'a,
>;

/// The SIMDgroup kernel matches an f64 reference, every other kernel matches it bit for bit, and each writes its
/// rows at an output row offset and stride without touching the rest.
#[rstest]
#[test_attr(uzu_test)]
fn projection_matches_reference(
    #[values(TrellisCodec::Vector4Restart64, TrellisCodec::Vector2Transition6, TrellisCodec::Vector2Transition4)]
    codec: TrellisCodec
) {
    let context = shared_metal_context();
    let mut rng = SmallRng::seed_from_u64(codec.transition_bits() as u64);
    let (vector_width, transition_bits) = (codec.vector_width(), codec.transition_bits());
    // rows: not a multiple of the narrow kernels' 32- and 64-row SIMDgroup tiles
    let (rows, columns, wide_stride) = (48usize, 5120usize, 80usize);
    let row_bytes = codec.row_bytes(columns as u32) as usize;
    let codes: Vec<u8> = (0..rows * row_bytes).map(|_| rng.random()).collect();
    let row_scales: Vec<f32> = (0..rows).map(|_| rng.random_range(0.001f32..0.01)).collect();
    let weights: Vec<Vec<f64>> = codes
        .chunks_exact(row_bytes)
        .map(|row| {
            let levels = row_levels(codec, row, columns);
            (0..columns)
                .map(|column| {
                    AFFINE[0] as f64 * levels[column] as f64 + AFFINE[1 + column % vector_width as usize] as f64
                })
                .collect()
        })
        .collect();
    let codebook = codebook_table(&package_codebook(vector_width as usize), vector_width as usize).unwrap();
    let codes = alloc_allocation_with_data::<Metal, u8>(&context, &codes);
    let scales = alloc_allocation_with_data::<Metal, f32>(&context, &row_scales);
    let codebook = alloc_allocation_with_data::<Metal, u8>(&context, &codebook);
    let (codes, scales, codebook) = (&codes, &scales, &codebook);

    let mut kernels: Vec<(&str, Encode)> = Vec::new();
    macro_rules! kernel {
        ($name:expr, $kernel:ident, $tile:expr) => {
            let kernel = $kernel::new(&context, $tile, vector_width, transition_bits).unwrap();
            kernels.push((
                $name,
                Box::new(move |activations, statistics, batch, output, stride, encoder| {
                    let (rows, columns) = (rows as u32, columns as u32);
                    kernel.encode(
                        codes,
                        activations,
                        statistics,
                        scales,
                        codebook,
                        output,
                        rows,
                        columns,
                        batch,
                        stride,
                        encoder,
                    )
                }),
            ));
        };
    }
    kernel!("simdgroup 1", MiraiSSimdgroupProjectionMetalKernel, 1);
    kernel!("simdgroup 8", MiraiSSimdgroupProjectionMetalKernel, 8);
    // the MXU kernels need M5 or later
    if context.supports_mxu {
        kernel!("wide 32", MiraiSProjectionMetalKernel, 32);
        kernel!("wide 64", MiraiSProjectionMetalKernel, 64);
        kernel!("narrow 2", MiraiSNarrowProjectionMetalKernel, 2);
        kernel!("narrow 4", MiraiSNarrowProjectionMetalKernel, 4);
    }
    let bits = |allocation: &Allocation<Metal>| -> Vec<u16> {
        allocation_to_vec::<Metal, bf16>(allocation).into_iter().map(bf16::to_bits).collect()
    };

    for batch in [1usize, 3, 17, 130] {
        let activations: Vec<i8> =
            (0..batch.next_multiple_of(64) * columns).map(|_| rng.random_range(-127i8..=127)).collect();
        let activation_scales: Vec<f32> = (0..batch).map(|_| rng.random_range(0.001f32..0.1)).collect();
        let token_statistics: Vec<f32> = (0..batch)
            .flat_map(|token| {
                let row = &activations[token * columns..][..columns];
                let sums: [f32; 4] =
                    std::array::from_fn(|class| row.iter().skip(class).step_by(4).map(|&value| value as f32).sum());
                [sums[0], sums[1], sums[2], sums[3], activation_scales[token], 0.0, 0.0, 0.0]
            })
            .collect();
        let activations_allocation = alloc_allocation_with_data::<Metal, i8>(&context, &activations);
        let statistics = alloc_allocation_with_data::<Metal, f32>(&context, &token_statistics);
        let outputs: Vec<Vec<u16>> = kernels
            .iter()
            .map(|(name, encode)| {
                let mut output = alloc_allocation::<Metal, bf16>(&context, batch * rows);
                let mut wide =
                    alloc_allocation_with_data::<Metal, bf16>(&context, &vec![bf16::ZERO; batch * wide_stride]);
                let mut encoder = Encoder::<Metal>::new(&context).unwrap();
                encode(&activations_allocation, &statistics, batch as u32, (&mut output, 0), rows as u32, &mut encoder);
                encode(
                    &activations_allocation,
                    &statistics,
                    batch as u32,
                    (&mut wide, 32),
                    wide_stride as u32,
                    &mut encoder,
                );
                encoder.end_encoding().submit().wait_until_completed().unwrap();
                let (output, wide) = (bits(&output), bits(&wide));
                for (token, wide) in wide.chunks_exact(wide_stride).enumerate() {
                    assert!(
                        wide[16..][..rows] == output[token * rows..][..rows],
                        "{name} batch {batch} token {token} at row 16"
                    );
                    assert!(
                        wide[..16].iter().chain(&wide[16 + rows..]).all(|&bits| bits == 0),
                        "{name} batch {batch} token {token}"
                    );
                }
                output
            })
            .collect();

        for token in 0..batch {
            let token_activations = &activations[token * columns..][..columns];
            for row in 0..rows {
                let products = weights[row].iter().zip(token_activations).map(|(w, &a)| w * a as f64);
                let (dot, magnitude) = products.fold((0.0, 0.0), |(dot, magnitude), p| (dot + p, magnitude + p.abs()));
                let factor = row_scales[row] as f64 * activation_scales[token] as f64;
                let expected = dot * factor;
                // bf16 output rounding plus the f32 epilogue on the (possibly cancelling) level and offset terms
                let tolerance = expected.abs() / 256.0 + magnitude * factor * 1e-6;
                let actual = bf16::from_bits(outputs[0][token * rows + row]).to_f64();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "batch {batch} token {token} row {row}: {actual} vs {expected}"
                );
            }
        }
        for ((name, _), output) in kernels.iter().zip(&outputs) {
            assert!(*output == outputs[0], "{name} batch {batch} differs from simdgroup 1");
        }
    }
}
