use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::kernel::mirai_s::mixing_order,
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
        util::shared_metal_context,
    },
};

// the Qwen3.8 S package's codebooks: [scale, offset of column class 0..4]
const V4_CODEBOOK: [f32; 5] = [0.05203748, -0.08205986, -0.07758855, -0.07814354, -0.0810989];
const V2_CODEBOOK: [f32; 5] = [0.052127663, -0.08220189, -0.077723004, -0.08220189, -0.077723004];

/// Signed level of each byte of fmix32(state * 0xCFCCB83F + 0x584B4AA3).
fn levels(state: u32) -> [i32; 4] {
    let mut x = state.wrapping_mul(0xCFCC_B83F).wrapping_add(0x584B_4AA3);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    std::array::from_fn(|byte| {
        let byte = (x >> (8 * byte)) & 0xFF;
        let field_sum: u32 = (0..4).map(|field| (byte >> (2 * field)) & 3).sum();
        (8 * field_sum + (3 * (byte & 15)) % 16) as i32 - 54
    })
}

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
                levels(state)[column % 4]
            },
            TrellisCodec::Vector2Transition6 | TrellisCodec::Vector2Transition4 => {
                let start = column / 2 * codec.transition_bits() as usize;
                let state = (0..16).fold(0, |state, offset| (state << 1) | bit(start + offset));
                levels(state)[column % 2]
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
    if !context.supports_mxu {
        return;
    }
    let transform = MetalMiraiSTransform::new(&context, columns).unwrap().unwrap();
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    let order = mixing_order(columns);
    let signs: Vec<f32> = (0..columns)
        .map(|_| {
            if rng.random::<bool>() {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
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

#[uzu_test]
fn transform_rejects_other_widths() {
    assert!(MetalMiraiSTransform::new(&shared_metal_context(), 4096).unwrap().is_none());
}

/// The projection kernels' codebook: f32 [scale, offsets, 0, 0, 0], then for V2 the int8 level pair of every state.
fn codebook_bytes(codec: TrellisCodec) -> Vec<u8> {
    let affine = match codec {
        TrellisCodec::Vector4Restart64 => V4_CODEBOOK,
        TrellisCodec::Vector2Transition6 | TrellisCodec::Vector2Transition4 => V2_CODEBOOK,
    };
    let header = [affine[0], affine[1], affine[2], affine[3], affine[4], 0.0, 0.0, 0.0];
    let mut bytes = bytemuck::cast_slice::<f32, u8>(&header).to_vec();
    if codec.vector_width() == 2 {
        bytes.extend((0..1u32 << 16).flat_map(|state| {
            let [first, second, ..] = levels(state);
            [first as i8 as u8, second as i8 as u8]
        }));
    }
    bytes
}

#[rstest]
#[test_attr(uzu_test)]
fn projection_matches_reference(
    #[values(TrellisCodec::Vector4Restart64, TrellisCodec::Vector2Transition6, TrellisCodec::Vector2Transition4)]
    codec: TrellisCodec,
    #[values(5120, 6144, 17408)] columns: u32,
) {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    let mut rng = SmallRng::seed_from_u64(u64::from(columns) + codec.transition_bits() as u64);
    let (vector_width, transition_bits) = (codec.vector_width(), codec.transition_bits());
    let wide_32 = MiraiSProjectionMetalKernel::new(&context, 32, vector_width, transition_bits).unwrap();
    let wide_64 = MiraiSProjectionMetalKernel::new(&context, 64, vector_width, transition_bits).unwrap();
    let narrow_2 = MiraiSNarrowProjectionMetalKernel::new(&context, 2, vector_width, transition_bits).unwrap();
    let narrow_4 = MiraiSNarrowProjectionMetalKernel::new(&context, 4, vector_width, transition_bits).unwrap();
    // not a multiple of the narrow kernels' 32- and 64-row SIMDgroup tiles
    let rows = 48u32;
    let codebook = match codec {
        TrellisCodec::Vector4Restart64 => V4_CODEBOOK,
        TrellisCodec::Vector2Transition6 | TrellisCodec::Vector2Transition4 => V2_CODEBOOK,
    };
    let row_bytes = codec.row_bytes(columns) as usize;
    let codes: Vec<u8> = (0..rows as usize * row_bytes).map(|_| rng.random()).collect();
    let row_scales: Vec<f32> = (0..rows).map(|_| rng.random_range(0.001f32..0.01)).collect();
    let weights: Vec<Vec<f64>> = codes
        .chunks_exact(row_bytes)
        .map(|row| {
            let row_levels = row_levels(codec, row, columns as usize);
            row_levels
                .iter()
                .enumerate()
                .map(|(column, &level)| codebook[0] as f64 * level as f64 + codebook[1 + column % 4] as f64)
                .collect()
        })
        .collect();
    let codes = alloc_allocation_with_data::<Metal, u8>(&context, &codes);
    let row_scales_allocation = alloc_allocation_with_data::<Metal, f32>(&context, &row_scales);
    let codebook_allocation = alloc_allocation_with_data::<Metal, u8>(&context, &codebook_bytes(codec));

    for batch in [1u32, 3, 8, 16, 17, 32, 33, 64, 130] {
        let padded_batch = batch.next_multiple_of(64);
        let activations: Vec<i8> = (0..padded_batch * columns).map(|_| rng.random_range(-127i8..=127)).collect();
        let activation_scales: Vec<f32> = (0..batch).map(|_| rng.random_range(0.001f32..0.1)).collect();
        let token_statistics: Vec<f32> = (0..batch as usize)
            .flat_map(|token| {
                let row = &activations[token * columns as usize..][..columns as usize];
                let sums: [f32; 4] =
                    std::array::from_fn(|class| row.iter().skip(class).step_by(4).map(|&value| value as f32).sum());
                [sums[0], sums[1], sums[2], sums[3], activation_scales[token], 0.0, 0.0, 0.0]
            })
            .collect();
        let activations_allocation = alloc_allocation_with_data::<Metal, i8>(&context, &activations);
        let token_statistics_allocation = alloc_allocation_with_data::<Metal, f32>(&context, &token_statistics);
        for kernel in ["wide 32", "wide 64", "narrow 2", "narrow 4"] {
            let mut output = alloc_allocation::<Metal, bf16>(&context, (batch * rows) as usize);
            let mut encoder = Encoder::<Metal>::new(&context).unwrap();
            macro_rules! encode {
                ($kernel:expr) => {
                    $kernel.encode(
                        &codes,
                        &activations_allocation,
                        &token_statistics_allocation,
                        &row_scales_allocation,
                        &codebook_allocation,
                        &mut output,
                        rows,
                        columns,
                        batch,
                        rows,
                        &mut encoder,
                    )
                };
            }
            match kernel {
                "wide 32" => encode!(wide_32),
                "wide 64" => encode!(wide_64),
                "narrow 2" => encode!(narrow_2),
                _ => encode!(narrow_4),
            }
            encoder.end_encoding().submit().wait_until_completed().unwrap();
            let output = allocation_to_vec::<Metal, bf16>(&output);

            for token in 0..batch as usize {
                let token_activations = &activations[token * columns as usize..][..columns as usize];
                for row in 0..rows as usize {
                    let products = weights[row].iter().zip(token_activations).map(|(w, &a)| w * a as f64);
                    let (dot, magnitude) =
                        products.fold((0.0, 0.0), |(dot, magnitude), p| (dot + p, magnitude + p.abs()));
                    let factor = row_scales[row] as f64 * activation_scales[token] as f64;
                    let expected = dot * factor;
                    // bf16 output rounding plus the f32 epilogue on the (possibly cancelling) level and offset terms
                    let tolerance = expected.abs() / 256.0 + magnitude * factor * 1e-6;
                    let actual = output[token * rows as usize + row].to_f64();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{kernel} batch {batch} token {token} row {row}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}
