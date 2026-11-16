use std::path::Path;

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde_json::{Value, json};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{Backend, Buffer, CommandBufferEncoding, Context},
    data_type::DataType,
    encodable_block::linear::UntiedReadout,
    safetensors::{ParameterLoader, ParameterTree},
    tests::helpers::{
        SafetensorsWriter, buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend,
        submit_command_buffer,
    },
};

const MODEL_DIM: u32 = 1024;
const PARTS: [(u32, u32, u32, u32); 2] = [(96, 8, 64, 128), (240, 4, 32, 32)];

struct PartWeights {
    signs: Vec<i32>,
    codes: Vec<u8>,
    scales: Vec<bf16>,
    zero_points: Vec<u8>,
}

fn part_spec(
    bits: u32,
    group_size: u32,
    hadamard_block_size: u32,
) -> Value {
    json!({
        "type": "HybridSpec",
        "quantization_spec": {
            "type": "IntSpec",
            "bits": bits,
            "group_size": group_size,
            "is_symmetric": bits == 8,
            "layout": "output_input",
        },
        "adapter_spec": null,
        "incoherence_spec": {"type": "HadamardSpec", "block_size": hadamard_block_size, "processing_mode": "input"},
    })
}

fn write_checkpoint(path: &Path) -> Vec<PartWeights> {
    let mut rng = SmallRng::seed_from_u64(5);
    let mut checkpoint = SafetensorsWriter::default();
    let parts = PARTS.map(|(rows, bits, group_size, hadamard_block_size)| {
        json!({"rows": rows, "spec": part_spec(bits, group_size, hadamard_block_size)})
    });
    checkpoint.add_metadata("stack.spec".into(), &json!({"type": "OutputStackSpec", "parts": parts}));
    let mut weights = Vec::new();
    for (index, (rows, bits, group_size, hadamard_block_size)) in PARTS.into_iter().enumerate() {
        checkpoint.add_metadata(format!("reference.{index}.spec"), &part_spec(bits, group_size, hadamard_block_size));
        let groups = MODEL_DIM / group_size;
        let part = PartWeights {
            signs: (0..MODEL_DIM).map(|_| rng.random_range(0..2) * 2 - 1).collect(),
            codes: (0..rows * MODEL_DIM * bits / 8).map(|_| rng.random()).collect(),
            scales: (0..groups * rows).map(|_| bf16::from_f32(rng.random_range(0.001..0.01))).collect(),
            zero_points: (0..groups * rows / 2).map(|_| rng.random()).collect(),
        };
        for prefix in [format!("stack.parts.{index}"), format!("reference.{index}")] {
            checkpoint.add_tensor(
                format!("{prefix}.incoherence_signs.input_signs"),
                "I32",
                &[MODEL_DIM],
                bytemuck::cast_slice(&part.signs),
            );
            checkpoint.add_tensor(
                format!("{prefix}.quantized.weights"),
                "U8",
                &[rows, MODEL_DIM * bits / 8],
                &part.codes,
            );
            checkpoint.add_tensor(
                format!("{prefix}.quantized.scales"),
                "BF16",
                &[groups, rows],
                bytemuck::cast_slice(&part.scales),
            );
            if bits == 4 {
                checkpoint.add_tensor(
                    format!("{prefix}.quantized.zero_points"),
                    "U8",
                    &[groups, rows / 2],
                    &part.zero_points,
                );
            }
        }
        weights.push(part);
    }
    checkpoint.write(path);
    weights
}

fn reference(
    parts: &[PartWeights],
    input: &[bf16],
) -> Vec<f32> {
    let model_dim = MODEL_DIM as usize;
    let nibble = |byte: u8, high: bool| {
        f32::from(if high {
            byte >> 4
        } else {
            byte & 15
        })
    };
    let mut output = Vec::new();
    for input_row in input.chunks(model_dim) {
        for (part, (rows, bits, group_size, hadamard_block_size)) in parts.iter().zip(PARTS) {
            let rows = rows as usize;
            let group_size = group_size as usize;
            let hadamard_block_size = hadamard_block_size as usize;
            let signed = input_row
                .iter()
                .zip(&part.signs)
                .map(|(value, &sign)| value.to_f32() * sign as f32)
                .collect::<Vec<_>>();
            let rotated = signed
                .chunks(hadamard_block_size)
                .flat_map(|block| {
                    (0..hadamard_block_size).map(move |output_index| {
                        let sum = block
                            .iter()
                            .enumerate()
                            .map(|(input_index, &value)| {
                                if (output_index & input_index).count_ones() % 2 == 0 {
                                    value
                                } else {
                                    -value
                                }
                            })
                            .sum::<f32>();
                        bf16::from_f32(sum / (hadamard_block_size as f32).sqrt()).to_f32()
                    })
                })
                .collect::<Vec<_>>();
            for output_row in 0..rows {
                let dot = rotated
                    .iter()
                    .enumerate()
                    .map(|(column, activation)| {
                        let group = column / group_size;
                        let scale = part.scales[group * rows + output_row].to_f32();
                        let weight_index = output_row * model_dim + column;
                        let code = if bits == 8 {
                            f32::from(part.codes[weight_index]) - 128.0
                        } else {
                            nibble(part.codes[weight_index / 2], column % 2 == 1)
                                - nibble(part.zero_points[(group * rows + output_row) / 2], output_row % 2 == 1)
                        };
                        code * scale * activation
                    })
                    .sum::<f32>();
                output.push(dot);
            }
        }
    }
    output
}

fn run<B: Backend>(
    context: &B::Context,
    readout: &UntiedReadout<B>,
    input: &[bf16],
    batch_dim: u32,
    output_dim: u32,
    gather_indices: Option<&[u32]>,
) -> Vec<f32> {
    let source = create_buffer_with_data::<B, bf16>(context, input);
    let gather_indices = gather_indices.map(|indices| create_buffer_with_data::<B, u32>(context, indices));
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let output = readout
        .encode(&source, batch_dim, output_dim, gather_indices.as_ref(), &mut command_buffer)
        .expect("readout encode");
    let mut result = create_buffer::<B, u8>(context, output.size());
    command_buffer.encode_copy(&output, &mut result);
    submit_command_buffer(command_buffer);
    buffer_to_vec::<B, bf16>(context, &result).into_iter().map(f32::from).collect()
}

fn relative_l2(
    expected: &[f32],
    actual: &[f32],
) -> f32 {
    assert_eq!(expected.len(), actual.len());
    let squared_error = expected.iter().zip(actual).map(|(expected, actual)| (expected - actual).powi(2)).sum::<f32>();
    (squared_error / expected.iter().map(|value| value * value).sum::<f32>()).sqrt()
}

#[uzu_test]
fn stacked_readout_matches_parts_and_reference() {
    let path = std::env::temp_dir().join(format!("uzu-stacked-readout-{}.safetensors", std::process::id()));
    let vocab_size = PARTS.iter().map(|&(rows, ..)| rows).sum::<u32>();
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    for_each_backend!(|B| {
        let context = create_context::<B>();
        let weights = write_checkpoint(&path);
        let (stack, references) = runtime.block_on(async {
            let loader = ParameterLoader::<B>::new(&path, &context).await.expect("parameter loader");
            let tree = loader.tree();
            let load = async |rows, tree: &ParameterTree<'_, B>| {
                let spec = tree.metadata("spec").expect("readout spec");
                UntiedReadout::<B>::load(rows, MODEL_DIM, DataType::BF16, spec, tree, &context)
                    .await
                    .expect("untied readout")
            };
            let stack = load(vocab_size, &tree.subtree("stack")).await;
            let mut references = Vec::new();
            for (index, &(rows, ..)) in PARTS.iter().enumerate() {
                references.push(load(rows, &tree.subtree(&format!("reference.{index}"))).await);
            }
            (stack, references)
        });
        std::fs::remove_file(&path).expect("remove safetensors");

        let mut rng = SmallRng::seed_from_u64(9);
        for batch_dim in [1, 4, 33] {
            let input =
                (0..batch_dim * MODEL_DIM).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect::<Vec<_>>();
            let expected = reference(&weights, &input);
            let part_outputs = references
                .iter()
                .zip(PARTS)
                .map(|(readout, (rows, ..))| run::<B>(&context, readout, &input, batch_dim, rows, None))
                .collect::<Vec<_>>();
            let concatenated = (0..batch_dim as usize)
                .flat_map(|row| {
                    PARTS.iter().zip(&part_outputs).flat_map(move |(&(rows, ..), part)| {
                        part[row * rows as usize..(row + 1) * rows as usize].iter().copied()
                    })
                })
                .collect::<Vec<_>>();
            if batch_dim <= 8 {
                let ids = (0..batch_dim as usize)
                    .flat_map(|row| {
                        [0, 95, 96, 97, vocab_size as usize - 1]
                            .into_iter()
                            .chain((0..19).map(move |column| (row * 131 + column * 37 + 5) % vocab_size as usize))
                    })
                    .map(|id| id as u32)
                    .collect::<Vec<_>>();
                let ids_per_row = ids.len() as u32 / batch_dim;
                let actual = run::<B>(&context, &stack, &input, batch_dim, ids_per_row, Some(&ids));
                for (baseline, full) in [("f32 reference", &expected), ("single-part readouts", &concatenated)] {
                    let gathered = ids
                        .chunks(ids_per_row as usize)
                        .enumerate()
                        .flat_map(|(row, row_ids)| {
                            row_ids.iter().map(move |&id| full[row * vocab_size as usize + id as usize])
                        })
                        .collect::<Vec<_>>();
                    let relative_error = relative_l2(&gathered, &actual);
                    assert!(
                        relative_error < 1e-2,
                        "{} batch {batch_dim} gathered: relative L2 {relative_error} against {baseline}",
                        std::any::type_name::<B>()
                    );
                }
            }
            for output_dim in [50, 200, vocab_size] {
                let actual = run::<B>(&context, &stack, &input, batch_dim, output_dim, None);
                for (baseline, full) in [("f32 reference", &expected), ("single-part readouts", &concatenated)] {
                    let prefix = full
                        .chunks(vocab_size as usize)
                        .flat_map(|row| &row[..output_dim as usize])
                        .copied()
                        .collect::<Vec<_>>();
                    let relative_error = relative_l2(&prefix, &actual);
                    assert!(
                        relative_error < 1e-2,
                        "{} batch {batch_dim} output_dim {output_dim}: relative L2 {relative_error} against {baseline}",
                        std::any::type_name::<B>()
                    );
                }
            }
        }
    });
}
