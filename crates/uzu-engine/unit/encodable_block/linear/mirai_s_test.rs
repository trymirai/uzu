use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use super::*;

// the Qwen3.8 S package's V4 codebook: [scale, offset of column class 0..4]; V2 uses the first two offsets
pub(crate) const AFFINE: [f32; 5] = [0.05203748, -0.08205986, -0.07758855, -0.07814354, -0.0810989];

/// The package codebook of `AFFINE`: scale * level + offset[component] for every trellis state.
pub(crate) fn package_codebook(vector_width: usize) -> Vec<f32> {
    (0..TRELLIS_STATES as u32)
        .flat_map(|state| {
            let levels = trellis_levels(state);
            (0..vector_width).map(move |component| AFFINE[0] * levels[component] as f32 + AFFINE[1 + component])
        })
        .collect()
}

#[uzu_test]
fn codebook_table_fits_computed_levels() {
    for vector_width in [4, 2] {
        let table = codebook_table(&package_codebook(vector_width), vector_width).unwrap();
        let header: &[f32] = bytemuck::cast_slice(&table[..32]);
        let expected: Vec<f32> = [AFFINE[0]]
            .into_iter()
            .chain((0..4).map(|class| AFFINE[1 + class % vector_width]))
            .chain([0.0; 3])
            .collect();
        assert!(header.iter().zip(&expected).all(|(fitted, expected)| (fitted - expected).abs() <= 1e-6), "{header:?}");
        // V2 level pairs follow the header; the kernels hash V4 levels
        assert_eq!(table.len(), 32 + (vector_width == 2) as usize * 2 * TRELLIS_STATES);
        if vector_width == 2 {
            let [first, second, ..] = trellis_levels(12345);
            assert_eq!(table[32 + 2 * 12345..][..2], [first as i8 as u8, second as i8 as u8]);
        }
    }
    let mut values = package_codebook(4);
    values[4 * 777 + 2] += 1e-3;
    assert!(codebook_table(&values, 4).unwrap_err().starts_with("entry (777, 2) is not a computed level"));
}

/// Every V2 group's state is the 16-bit big-endian window of the repacked row, by the package's trellis recursion:
/// a 16-bit little-endian seed, then each transition (packed LSB first) shifted in at the bottom.
#[rstest]
#[test_attr(uzu_test)]
fn repack_windows_follow_the_trellis(
    #[values(TrellisCodec::Vector2Transition6, TrellisCodec::Vector2Transition4)] codec: TrellisCodec,
    #[values(128, 5120)] columns: u32,
) {
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    let (row_bytes, transition_bits) = (codec.row_bytes(columns) as usize, codec.transition_bits() as usize);
    let physical: Vec<u8> = (0..3 * row_bytes).map(|_| rng.random()).collect();
    let mut repacked = physical.clone();
    repack_msb_first(&mut repacked, codec, columns);
    for (source, row) in physical.chunks_exact(row_bytes).zip(repacked.chunks_exact(row_bytes)) {
        let source_bit = |index: usize| ((source[index / 8] >> (index % 8)) & 1) as u32;
        let window_bit = |index: usize| ((row[index / 8] >> (7 - index % 8)) & 1) as u32;
        let mut state = source[0] as u32 | (source[1] as u32) << 8;
        for group in 0..columns as usize / 2 {
            if group > 0 {
                let start = 16 + (group - 1) * transition_bits;
                let transition =
                    (0..transition_bits).fold(0, |value, offset| value | source_bit(start + offset) << offset);
                state = ((state << transition_bits) | transition) & 0xFFFF;
            }
            let window = (0..16).fold(0, |value, offset| (value << 1) | window_bit(group * transition_bits + offset));
            assert_eq!(window, state, "group {group}");
        }
    }
}

/// The symmetric U4 repack of the Mirai S readout dequantizes to row_scale * ladder[index] * (2c - 7) for the
/// 3-bit code c read LSB first, with the group scale rounded to bf16 and the padding rows zero.
#[uzu_test]
fn mirai_s_readout_repack_matches_reference() {
    let mut rng = SmallRng::seed_from_u64(5);
    let (rows, columns, padded_rows) = (6usize, 256usize, 8usize);
    let codes: Vec<u8> = (0..rows * columns * 3 / 8).map(|_| rng.random()).collect();
    let row_scales: Vec<bf16> = (0..rows).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.1))).collect();
    let ladder_indices: Vec<u8> = (0..rows * columns / 128).map(|_| rng.random()).collect();
    let ladder: Vec<f16> = (0..16).map(|index| f16::from_f32(2.0f32.powf(index as f32 / 2.0 - 5.5))).collect();
    let (u4_codes, scales) = repack_readout(&codes, &row_scales, &ladder_indices, &ladder, padded_rows);
    for (group, group_scales) in scales.chunks_exact(padded_rows).enumerate() {
        assert!(group_scales[rows..].iter().all(|&scale| scale == bf16::ZERO));
        for (row, &scale) in group_scales[..rows].iter().enumerate() {
            let ladder_index = (ladder_indices[row * columns / 128 + group / 2] >> (4 * (group % 2))) & 15;
            assert_eq!(scale, bf16::from_f32(row_scales[row].to_f32() * ladder[ladder_index as usize].to_f32()));
            for column in group * 64..(group + 1) * 64 {
                let bit = |offset: usize| {
                    let index = 3 * column + offset;
                    ((codes[(row * columns * 3 + index) / 8] >> (index % 8)) & 1) as i32
                };
                let code = bit(0) | bit(1) << 1 | bit(2) << 2;
                let nibble = (u4_codes[(row * columns + column) / 2] >> (4 * (column % 2))) & 15;
                assert_eq!(nibble as i32 - 8, 2 * code - 7, "row {row} column {column}");
            }
        }
    }
}
