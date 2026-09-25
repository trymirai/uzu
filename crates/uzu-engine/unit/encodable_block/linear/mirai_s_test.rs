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
fn codebook_fit_recovers_scale_and_offsets() {
    let v2_affine = [0.052127663, -0.08220189, -0.077723004, 0.0, 0.0];
    for (affine, vector_width) in [(V4_AFFINE, 4), (v2_affine, 2)] {
        let fitted = fit_codebook(&package_codebook(affine, vector_width), vector_width).unwrap();
        let expected = std::array::from_fn::<f32, 5, _>(|index| match index {
            0 => affine[0],
            class => affine[1 + (class - 1) % vector_width],
        });
        for (fitted, expected) in fitted.iter().zip(expected) {
            assert!((fitted - expected).abs() <= 1e-6, "{fitted:?} vs {expected:?}");
        }
        let table = codebook_table(&package_codebook(affine, vector_width), vector_width).unwrap();
        // V2 level pairs follow the header; the kernels hash V4 levels
        match vector_width {
            2 => {
                assert_eq!(table.len(), 32 + 2 * TRELLIS_STATES);
                let [first, second, ..] = trellis_levels(12345);
                assert_eq!(table[32 + 2 * 12345..][..2], [first as i8 as u8, second as i8 as u8]);
            },
            _ => assert_eq!(table.len(), 32),
        }
    }
}

#[uzu_test]
fn codebook_fit_rejects_other_tables() {
    let mut values = package_codebook(V4_AFFINE, 4);
    values[4 * 777 + 2] += 1e-3;
    let error = fit_codebook(&values, 4).unwrap_err();
    assert!(error.starts_with("entry (777, 2) is not a computed level"), "{error}");
}

/// States of a V2 row in the package layout, by the trellis recursion: a 16-bit little-endian seed, then each
/// transition (packed LSB first) shifted in at the bottom.
fn package_states(
    row: &[u8],
    transition_bits: usize,
    groups: usize,
) -> Vec<u32> {
    let bit = |index: usize| ((row[index / 8] >> (index % 8)) & 1) as u32;
    let mut state = row[0] as u32 | (row[1] as u32) << 8;
    let mut states = vec![state];
    for group in 1..groups {
        let start = 16 + (group - 1) * transition_bits;
        let transition = (0..transition_bits).fold(0, |value, offset| value | bit(start + offset) << offset);
        state = ((state << transition_bits) | transition) & 0xFFFF;
        states.push(state);
    }
    states
}

#[rstest]
#[test_attr(uzu_test)]
fn repack_windows_follow_the_trellis(
    #[values(TrellisCodec::Vector2Transition6, TrellisCodec::Vector2Transition4)] codec: TrellisCodec,
    #[values(128, 5120, 6144)] columns: u32,
) {
    let mut rng = SmallRng::seed_from_u64(u64::from(columns));
    let rows = 3;
    let row_bytes = codec.row_bytes(columns) as usize;
    let physical: Vec<u8> = (0..rows * row_bytes).map(|_| rng.random()).collect();
    let mut repacked = physical.clone();
    repack_msb_first(&mut repacked, codec, columns);
    let transition_bits = codec.transition_bits() as usize;
    for (source, row) in physical.chunks_exact(row_bytes).zip(repacked.chunks_exact(row_bytes)) {
        let bit = |index: usize| ((row[index / 8] >> (7 - index % 8)) & 1) as u32;
        for (group, state) in package_states(source, transition_bits, columns as usize / 2).into_iter().enumerate() {
            let window = (0..16).fold(0, |value, offset| (value << 1) | bit(group * transition_bits + offset));
            assert_eq!(window, state, "group {group}");
        }
    }
}

#[cfg(backend = "metal")]
mod row_stack {
    use std::{fs::File, io::Write};

    use half::f16;
    use serde_json::json;

    use super::*;
    use crate::{
        backends::{common::Encoder, metal::Metal},
        encodable_block::linear::LinearBlockError,
        parameters::ParameterLoader,
        tests::{
            helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
            util::shared_metal_context,
        },
    };

    const COLUMNS: u32 = 5120;

    struct Tensors {
        header: serde_json::Map<String, serde_json::Value>,
        data: Vec<u8>,
    }

    impl Tensors {
        fn add(
            &mut self,
            name: &str,
            dtype: &str,
            shape: &[u32],
            bytes: &[u8],
        ) {
            let offsets = [self.data.len(), self.data.len() + bytes.len()];
            self.header.insert(name.into(), json!({"dtype": dtype, "shape": shape, "data_offsets": offsets}));
            self.data.extend_from_slice(bytes);
        }

        fn add_leaf(
            &mut self,
            prefix: &str,
            leaf: &Leaf,
        ) {
            let rows = leaf.rows;
            self.add(&format!("{prefix}.codes"), "U8", &[rows, leaf.codec.row_bytes(COLUMNS)], &leaf.codes);
            let (scale_dtype, scales) = match leaf.codec {
                TrellisCodec::Vector4Restart64 => ("F32", bytemuck::cast_slice(&leaf.scales).to_vec()),
                _ => {
                    let scales: Vec<f16> = leaf.scales.iter().map(|&scale| f16::from_f32(scale)).collect();
                    ("F16", bytemuck::cast_slice(&scales).to_vec())
                },
            };
            self.add(&format!("{prefix}.scales"), scale_dtype, &[rows], &scales);
            self.add(&format!("{prefix}.gains"), "BF16", &[rows], bytemuck::cast_slice(&leaf.gains));
            self.add(&format!("{prefix}.post_gains.0"), "F32", &[rows], bytemuck::cast_slice(&leaf.post_gains));
        }
    }

    struct Leaf {
        codec: TrellisCodec,
        rows: u32,
        codes: Vec<u8>,
        scales: Vec<f32>,
        gains: Vec<bf16>,
        post_gains: Vec<f32>,
    }

    impl Leaf {
        fn random(
            codec: TrellisCodec,
            rows: u32,
            rng: &mut SmallRng,
        ) -> Self {
            Self {
                codec,
                rows,
                codes: (0..rows * codec.row_bytes(COLUMNS)).map(|_| rng.random()).collect(),
                scales: (0..rows).map(|_| rng.random_range(0.001f32..0.01)).collect(),
                gains: (0..rows).map(|_| bf16::from_f32(rng.random_range(0.5f32..2.0))).collect(),
                post_gains: (0..rows).map(|_| rng.random_range(0.5f32..2.0)).collect(),
            }
        }

        fn spec(&self) -> serde_json::Value {
            let (vector_width, transition_bits, restart_columns, scale_dtype) = match self.codec {
                TrellisCodec::Vector4Restart64 => (4, 8, 64, "float32"),
                codec => (2, codec.transition_bits(), 0, "float16"),
            };
            json!({
                "type": "QtipGaussianSpec",
                "layout": "output_input",
                "vector_width": vector_width,
                "transition_bits": transition_bits,
                "restart_columns": restart_columns,
                "scale_dtype": scale_dtype,
                "post_gain_axes": ["row"],
            })
        }
    }

    /// A row stack of a V2 and a V4 part equals the two parts loaded as plain leaves, row-concatenated, bit for bit.
    #[uzu_test]
    fn row_stack_matches_its_parts() {
        let context = shared_metal_context();
        let mut rng = SmallRng::seed_from_u64(3);
        let first = Leaf::random(TrellisCodec::Vector2Transition6, 32, &mut rng);
        let second = Leaf::random(TrellisCodec::Vector4Restart64, 48, &mut rng);

        let mut tensors = Tensors {
            header: serde_json::Map::new(),
            data: Vec::new(),
        };
        let signs: Vec<f32> = (0..COLUMNS)
            .map(|_| {
                if rng.random::<bool>() {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let mixing: Vec<f32> = (0..25).map(|_| rng.random_range(-0.5f32..0.5)).collect();
        tensors.add("qtip_shared.signs_5120", "F32", &[COLUMNS], bytemuck::cast_slice(&signs));
        tensors.add("qtip_shared.q_5120", "F32", &[5, 5], bytemuck::cast_slice(&mixing));
        let v2_affine = [0.052127663, -0.08220189, -0.077723004, 0.0, 0.0];
        tensors.add(
            "qtip_shared.codebook_v2",
            "F32",
            &[TRELLIS_STATES as u32, 2],
            bytemuck::cast_slice(&package_codebook(v2_affine, 2)),
        );
        tensors.add(
            "qtip_shared.codebook_v4",
            "F32",
            &[TRELLIS_STATES as u32, 4],
            bytemuck::cast_slice(&package_codebook(V4_AFFINE, 4)),
        );
        tensors.add_leaf("stack.weights.parts.0", &first);
        tensors.add_leaf("stack.weights.parts.1", &second);
        tensors.add_leaf("first.weights", &first);
        tensors.add_leaf("second.weights", &second);
        let stack_spec = json!({
            "type": "RowStackSpec",
            "parts": [[first.rows, first.spec()], [second.rows, second.spec()]],
            "layout": "output_input",
        });
        let empty_spec = json!({"type": "RowStackSpec", "parts": [], "layout": "output_input"});
        tensors.header.insert(
            "__metadata__".into(),
            json!({
                "stack.weights.spec": stack_spec.to_string(),
                "empty.weights.spec": empty_spec.to_string(),
                "first.weights.spec": first.spec().to_string(),
                "second.weights.spec": second.spec().to_string(),
            }),
        );

        let path = std::env::temp_dir().join(format!("mirai_s_row_stack_{}.safetensors", std::process::id()));
        let header = serde_json::to_vec(&tensors.header).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&tensors.data).unwrap();
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let loader = ParameterLoader::<Metal>::new(&file, &*context).unwrap();
        let load = |name: &str, rows: u32| {
            <dyn Linear<Metal>>::new(COLUMNS, [rows], false, &*context, DataType::BF16, &loader.tree().subtree(name))
                .unwrap()
        };
        let stack = load("stack", first.rows + second.rows);
        let linears = [load("first", first.rows), load("second", second.rows)];
        loader.tree().assert_all_tensors_validated().unwrap();
        let empty =
            <dyn Linear<Metal>>::new(COLUMNS, [0], false, &*context, DataType::BF16, &loader.tree().subtree("empty"));
        assert!(matches!(empty, Err(LinearBlockError::UnsupportedConfiguration(_))));

        for batch in [1u32, 5, 40] {
            let input: Vec<bf16> =
                (0..batch * COLUMNS).map(|_| bf16::from_f32(rng.random_range(-2.0f32..2.0))).collect();
            let input = alloc_allocation_with_data::<Metal, bf16>(&context, &input);
            let encode = |linear: &dyn Linear<Metal>, rows: u32| {
                let mut encoder = Encoder::<Metal>::new(&context).unwrap();
                let mut input_copy = encoder.allocate_scratch(input.size()).unwrap();
                encoder.encode_copy(&input, .., &mut input_copy, ..);
                let output = linear.encode(input_copy, batch, &mut encoder).unwrap();
                let mut result = alloc_allocation::<Metal, bf16>(&context, (batch * rows) as usize);
                encoder.encode_copy(&output, .., &mut result, ..);
                drop(output);
                encoder.end_encoding().submit().wait_until_completed().unwrap();
                allocation_to_vec::<Metal, bf16>(&result)
            };
            let stacked = encode(stack.as_ref(), first.rows + second.rows);
            let parts = [encode(linears[0].as_ref(), first.rows), encode(linears[1].as_ref(), second.rows)];
            for token in 0..batch as usize {
                let expected: Vec<u16> = parts
                    .iter()
                    .zip([first.rows, second.rows])
                    .flat_map(|(part, rows)| part[token * rows as usize..][..rows as usize].iter().map(|v| v.to_bits()))
                    .collect();
                let rows = (first.rows + second.rows) as usize;
                let actual: Vec<u16> = stacked[token * rows..][..rows].iter().map(|value| value.to_bits()).collect();
                assert_eq!(actual, expected, "batch {batch} token {token}");
            }
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
    let mut u4_codes = vec![0u8; rows * columns / 2];
    // GroupOutput scales pad each group's rows to a multiple of 4
    let padded_rows = rows.next_multiple_of(4);
    let mut scales = vec![bf16::ONE; groups * padded_rows];
    repack_readout(&codes, &row_scales, &ladder_indices, &ladder, &mut u4_codes, &mut scales);

    for row in 0..rows {
        let row_codes = &codes[row * columns * 3 / 8..][..columns * 3 / 8];
        for column in 0..columns {
            let code = (0..3).fold(0, |code, bit| {
                let index = 3 * column + bit;
                code | (((row_codes[index / 8] >> (index % 8)) & 1) as i32) << bit
            });
            let group = column / 64;
            let packed_index = ladder_indices[row * groups / 2 + group / 2];
            let ladder_index = if group % 2 == 0 {
                packed_index & 15
            } else {
                packed_index >> 4
            };
            let scale = bf16::from_f32(row_scales[row].to_f32() * ladder[ladder_index as usize].to_f32());
            let packed = u4_codes[(row * columns + column) / 2];
            let nibble = if column % 2 == 0 {
                packed & 15
            } else {
                packed >> 4
            };
            assert_eq!(nibble as i32 - 8, 2 * code - 7, "row {row} column {column}");
            assert_eq!(scales[group * padded_rows + row], scale, "row {row} column {column}");
        }
    }
    for group in 0..groups {
        assert!(scales[group * padded_rows + rows..(group + 1) * padded_rows].iter().all(|&scale| scale == bf16::ZERO));
    }
}
