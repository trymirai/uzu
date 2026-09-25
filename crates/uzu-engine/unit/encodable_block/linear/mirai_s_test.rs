use half::{bf16, f16};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use uzu_engine_macros::uzu_test;

use super::*;

// the Qwen3.8 S package's V4 codebook: [scale, offset of component 0..4]
const V4_AFFINE: [f32; 5] = [0.05203748, -0.08205986, -0.07758855, -0.07814354, -0.0810989];

fn package_codebook(
    affine: [f32; 5],
    vector_width: usize,
) -> Vec<f32> {
    (0..TRELLIS_STATES as u32)
        .flat_map(|state| {
            let levels = trellis_levels(state);
            (0..vector_width).map(move |component| affine[0] * levels[component] as f32 + affine[1 + component])
        })
        .collect()
}

#[uzu_test]
fn codebook_table_fits_computed_levels() {
    let v2_affine = [0.052127663, -0.08220189, -0.077723004, -0.08220189, -0.077723004];
    for (affine, vector_width) in [(V4_AFFINE, 4), (v2_affine, 2)] {
        let table = codebook_table(&package_codebook(affine, vector_width), vector_width).unwrap();
        let header: &[f32] = bytemuck::cast_slice(&table[..32]);
        for (fitted, expected) in header.iter().zip(affine.iter().chain(&[0.0; 3])) {
            assert!((fitted - expected).abs() <= 1e-6, "{header:?} vs {affine:?}");
        }
        // V2 level pairs follow the header; the kernels hash V4 levels
        if vector_width == 2 {
            let [first, second, ..] = trellis_levels(12345);
            assert_eq!(table[32 + 2 * 12345..][..2], [first as i8 as u8, second as i8 as u8]);
        }
        assert_eq!(table.len(), 32 + (vector_width == 2) as usize * 2 * TRELLIS_STATES);
    }
    let mut values = package_codebook(V4_AFFINE, 4);
    values[4 * 777 + 2] += 1e-3;
    let error = codebook_table(&values, 4).unwrap_err();
    assert!(error.starts_with("entry (777, 2) is not a computed level"), "{error}");
}

#[rstest]
#[test_attr(uzu_test)]
fn repack_windows_follow_the_trellis(
    #[values(TrellisCodec::Vector2Transition6, TrellisCodec::Vector2Transition4)] codec: TrellisCodec,
    #[values(128, 5120)] columns: u32,
) {
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    let row_bytes = codec.row_bytes(columns) as usize;
    let physical: Vec<u8> = (0..3 * row_bytes).map(|_| rng.random()).collect();
    let mut repacked = physical.clone();
    repack_msb_first(&mut repacked, codec, columns);
    let transition_bits = codec.transition_bits() as usize;
    for (source, row) in physical.chunks_exact(row_bytes).zip(repacked.chunks_exact(row_bytes)) {
        // the package layout by the trellis recursion: a 16-bit little-endian seed, then each transition (packed
        // LSB first) shifted in at the bottom
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

#[cfg(backend = "metal")]
mod row_stack {
    use std::{fs::File, io::Write};

    use serde_json::json;

    use super::*;
    use crate::{
        backends::{common::Encoder, metal::Metal},
        parameters::ParameterLoader,
        tests::{
            helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
            util::shared_metal_context,
        },
    };

    const COLUMNS: u32 = 5120;

    /// (name, safetensors dtype, shape, bytes)
    type Tensor = (String, &'static str, Vec<u32>, Vec<u8>);

    /// The spec and tensors of a random leaf under `prefix`; the same seed gives the same leaf.
    fn leaf(
        prefix: &str,
        codec: TrellisCodec,
        rows: u32,
        seed: u64,
    ) -> (serde_json::Value, Vec<Tensor>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (vector_width, transition_bits, restart_columns, dtype, scale_dtype) = match codec {
            TrellisCodec::Vector4Restart64 => (4, 8, 64, "F32", "float32"),
            codec => (2, codec.transition_bits(), 0, "F16", "float16"),
        };
        let spec = json!({
            "type": "QtipGaussianSpec",
            "layout": "output_input",
            "vector_width": vector_width,
            "transition_bits": transition_bits,
            "restart_columns": restart_columns,
            "scale_dtype": scale_dtype,
            "post_gain_axes": ["row"],
        });
        let scales: Vec<f32> = (0..rows).map(|_| rng.random_range(0.001f32..0.01)).collect();
        let scales = match vector_width {
            4 => bytemuck::cast_slice(&scales).to_vec(),
            _ => bytemuck::cast_slice(&scales.iter().map(|&scale| f16::from_f32(scale)).collect::<Vec<_>>()).to_vec(),
        };
        let gains: Vec<bf16> = (0..rows).map(|_| bf16::from_f32(rng.random_range(0.5f32..2.0))).collect();
        let post_gains: Vec<f32> = (0..rows).map(|_| rng.random_range(0.5f32..2.0)).collect();
        let row_bytes = codec.row_bytes(COLUMNS);
        let codes: Vec<u8> = (0..rows * row_bytes).map(|_| rng.random()).collect();
        let tensors = vec![
            (format!("{prefix}.codes"), "U8", vec![rows, row_bytes], codes),
            (format!("{prefix}.scales"), dtype, vec![rows], scales),
            (format!("{prefix}.gains"), "BF16", vec![rows], bytemuck::cast_slice(&gains).to_vec()),
            (format!("{prefix}.post_gains.0"), "F32", vec![rows], bytemuck::cast_slice(&post_gains).to_vec()),
        ];
        (spec, tensors)
    }

    fn safetensors(
        metadata: serde_json::Value,
        tensors: &[Tensor],
    ) -> File {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let offsets = [data.len(), data.len() + bytes.len()];
            header.insert(name.clone(), json!({"dtype": dtype, "shape": shape, "data_offsets": offsets}));
            data.extend_from_slice(bytes);
        }
        header.insert("__metadata__".into(), metadata);
        let header = serde_json::to_vec(&header).unwrap();
        let path = std::env::temp_dir().join(format!("mirai_s_row_stack_{}.safetensors", std::process::id()));
        let mut file = File::options().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&data).unwrap();
        file
    }

    /// A row stack of a V2 and a V4 part equals the two parts loaded as plain leaves, row-concatenated, bit for bit.
    #[uzu_test]
    fn row_stack_matches_its_parts() {
        let context = shared_metal_context();
        let mut rng = SmallRng::seed_from_u64(3);
        let (first_rows, second_rows) = (32u32, 48u32);
        let (first_spec, first) = leaf("first.weights", TrellisCodec::Vector2Transition6, first_rows, 1);
        let (second_spec, second) = leaf("second.weights", TrellisCodec::Vector4Restart64, second_rows, 2);
        let (_, stack_first) = leaf("stack.weights.parts.0", TrellisCodec::Vector2Transition6, first_rows, 1);
        let (_, stack_second) = leaf("stack.weights.parts.1", TrellisCodec::Vector4Restart64, second_rows, 2);
        let f32_bytes = |values: Vec<f32>| bytemuck::cast_slice(&values).to_vec();
        let signs = (0..COLUMNS).map(|_| [1.0, -1.0][rng.random_range(0..2)]).collect();
        let mixing = (0..25).map(|_| rng.random_range(-0.5f32..0.5)).collect();
        let v2_affine = [0.052127663, -0.08220189, -0.077723004, 0.0, 0.0];
        let shared = vec![
            ("qtip_shared.signs_5120".into(), "F32", vec![COLUMNS], f32_bytes(signs)),
            ("qtip_shared.q_5120".into(), "F32", vec![5, 5], f32_bytes(mixing)),
            (
                "qtip_shared.codebook_v2".into(),
                "F32",
                vec![TRELLIS_STATES as u32, 2],
                f32_bytes(package_codebook(v2_affine, 2)),
            ),
            (
                "qtip_shared.codebook_v4".into(),
                "F32",
                vec![TRELLIS_STATES as u32, 4],
                f32_bytes(package_codebook(V4_AFFINE, 4)),
            ),
        ];
        let stack_spec = json!({
            "type": "RowStackSpec",
            "parts": [[first_rows, first_spec], [second_rows, second_spec]],
            "layout": "output_input",
        });
        let metadata = json!({
            "stack.weights.spec": stack_spec.to_string(),
            "first.weights.spec": first_spec.to_string(),
            "second.weights.spec": second_spec.to_string(),
        });
        let file = safetensors(metadata, &[shared, first, second, stack_first, stack_second].concat());
        let loader = ParameterLoader::<Metal>::new(&file, &*context).unwrap();
        let load = |name: &str, rows: u32| {
            <dyn Linear<Metal>>::new(COLUMNS, [rows], false, &*context, DataType::BF16, &loader.tree().subtree(name))
                .unwrap()
        };
        let stack = load("stack", first_rows + second_rows);
        let parts = [load("first", first_rows), load("second", second_rows)];
        loader.tree().assert_all_tensors_validated().unwrap();

        for batch in [1u32, 40] {
            let input: Vec<bf16> =
                (0..batch * COLUMNS).map(|_| bf16::from_f32(rng.random_range(-2.0f32..2.0))).collect();
            let input = alloc_allocation_with_data::<Metal, bf16>(&context, &input);
            let encode = |linear: &dyn Linear<Metal>, rows: u32| -> Vec<u16> {
                let mut encoder = Encoder::<Metal>::new(&context).unwrap();
                let mut input_copy = encoder.allocate_scratch(input.size()).unwrap();
                encoder.encode_copy(&input, .., &mut input_copy, ..);
                let output = linear.encode(input_copy, batch, &mut encoder).unwrap();
                let mut result = alloc_allocation::<Metal, bf16>(&context, (batch * rows) as usize);
                encoder.encode_copy(&output, .., &mut result, ..);
                drop(output);
                encoder.end_encoding().submit().wait_until_completed().unwrap();
                allocation_to_vec::<Metal, bf16>(&result).into_iter().map(bf16::to_bits).collect()
            };
            let stacked = encode(stack.as_ref(), first_rows + second_rows);
            let (first, second) = (encode(parts[0].as_ref(), first_rows), encode(parts[1].as_ref(), second_rows));
            let (first_rows, second_rows) = (first_rows as usize, second_rows as usize);
            let expected: Vec<u16> = (0..batch as usize)
                .flat_map(|token| {
                    [&first[token * first_rows..][..first_rows], &second[token * second_rows..][..second_rows]].concat()
                })
                .collect();
            assert_eq!(stacked, expected, "batch {batch}");
        }
    }
}

/// The symmetric U4 repack of the Mirai S readout dequantizes to row_scale * ladder[index] * (2c - 7) for the
/// 3-bit code c read LSB first, with the group scale rounded to bf16.
#[uzu_test]
fn mirai_s_readout_repack_matches_reference() {
    let mut rng = SmallRng::seed_from_u64(5);
    let (rows, columns) = (6usize, 256usize);
    let codes: Vec<u8> = (0..rows * columns * 3 / 8).map(|_| rng.random()).collect();
    let row_scales: Vec<bf16> = (0..rows).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.1))).collect();
    let ladder_indices: Vec<u8> = (0..rows * columns / 128).map(|_| rng.random()).collect();
    let ladder: Vec<f16> = (0..16).map(|index| f16::from_f32(2.0f32.powf(index as f32 / 2.0 - 5.5))).collect();
    let groups = columns / 64;
    // GroupOutput scales pad each group's rows to a multiple of 4
    let padded_rows = rows.next_multiple_of(4);
    let (u4_codes, scales) = repack_readout(&codes, &row_scales, &ladder_indices, &ladder, padded_rows);

    for row in 0..rows {
        let row_codes = &codes[row * columns * 3 / 8..][..columns * 3 / 8];
        for column in 0..columns {
            let code = (0..3).fold(0, |code, bit| {
                let index = 3 * column + bit;
                code | (((row_codes[index / 8] >> (index % 8)) & 1) as i32) << bit
            });
            let group = column / 64;
            let ladder_index = (ladder_indices[row * groups / 2 + group / 2] >> (4 * (group % 2))) & 15;
            let scale = bf16::from_f32(row_scales[row].to_f32() * ladder[ladder_index as usize].to_f32());
            let nibble = (u4_codes[(row * columns + column) / 2] >> (4 * (column % 2))) & 15;
            assert_eq!(nibble as i32 - 8, 2 * code - 7, "row {row} column {column}");
            assert_eq!(scales[group * padded_rows + row], scale, "row {row} column {column}");
        }
    }
    for group in 0..groups {
        assert!(scales[group * padded_rows + rows..(group + 1) * padded_rows].iter().all(|&scale| scale == bf16::ZERO));
    }
}
