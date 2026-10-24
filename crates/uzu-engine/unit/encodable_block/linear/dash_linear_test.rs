use std::{fs::File, io::Write, path::Path};

use futures::executor::block_on;
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde_json::json;
use uzu_engine_macros::uzu_test;

use super::Linear;
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        kernel::DASH_HADAMARD_BLOCK_SIZE,
    },
    data_type::DataType,
    safetensors::ParameterLoader,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend},
};

// One MLX group spans the whole 512-wide row (four H128 blocks).
const INPUT_DIM: usize = 512;
const OUTPUT_DIM: usize = 256;

struct DashWeights {
    /// Unpacked 2-bit codes, `[OUTPUT_DIM, INPUT_DIM]`.
    codes: Vec<u8>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
    signs: Vec<i32>,
}

/// Writes the linear as lalamo exports Dash: Hybrid(MLX bits=2, whole-row group), input-only signs, block 128.
fn write_safetensors(
    path: &Path,
    weights: &DashWeights,
) {
    let packed: Vec<u8> = weights
        .codes
        .chunks_exact(4)
        .map(|codes| codes[0] | (codes[1] << 2) | (codes[2] << 4) | (codes[3] << 6))
        .collect();
    let tensors: [(&str, &str, Vec<usize>, &[u8]); 4] = [
        ("weights.quantized.weights", "U8", vec![OUTPUT_DIM, INPUT_DIM / 4], &packed),
        ("weights.quantized.scales", "BF16", vec![1, OUTPUT_DIM], bytemuck::cast_slice(&weights.scales)),
        ("weights.quantized.biases", "BF16", vec![1, OUTPUT_DIM], bytemuck::cast_slice(&weights.biases)),
        ("weights.incoherence_signs.input_signs", "I32", vec![INPUT_DIM], bytemuck::cast_slice(&weights.signs)),
    ];

    let quantized_spec = json!({"type": "MLXSpec", "bits": 2, "group_size": INPUT_DIM, "layout": "output_input"});
    let spec = json!({
        "type": "HybridSpec",
        "quantization_spec": quantized_spec,
        "adapter_spec": null,
        "incoherence_block_size": DASH_HADAMARD_BLOCK_SIZE,
        "incoherence_processing_mode": "input",
    });
    let mut header = json!({"__metadata__": {
        "weights.spec": spec.to_string(),
        "weights.quantized.spec": quantized_spec.to_string(),
    }});
    let mut offset = 0;
    for (name, dtype, shape, bytes) in &tensors {
        header[*name] = json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]});
        offset += bytes.len();
    }
    let header = header.to_string();

    let mut file = File::create(path).expect("create safetensors fixture");
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for (_, _, _, bytes) in &tensors {
        file.write_all(bytes).unwrap();
    }
}

/// y = (q * scale + bias) . (H128 blocks . (signs * x)), with the transformed activation rounded to bf16
/// like the GPU's in-place activation buffer.
fn reference(
    weights: &DashWeights,
    input: &[bf16],
) -> Vec<f64> {
    let block_size = DASH_HADAMARD_BLOCK_SIZE as usize;
    let mut output = Vec::new();
    for row in input.chunks_exact(INPUT_DIM) {
        let transformed: Vec<f64> = (0..INPUT_DIM)
            .map(|index| {
                let block_start = index - index % block_size;
                let sum: f64 = (0..block_size)
                    .map(|column| {
                        let sylvester_sign = if ((index % block_size) & column).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        let channel = block_start + column;
                        sylvester_sign * f64::from(weights.signs[channel]) * f64::from(row[channel].to_f32())
                    })
                    .sum();
                f64::from(bf16::from_f64(sum / (block_size as f64).sqrt()).to_f32())
            })
            .collect();
        for output_index in 0..OUTPUT_DIM {
            let scale = f64::from(weights.scales[output_index].to_f32());
            let bias = f64::from(weights.biases[output_index].to_f32());
            let codes = &weights.codes[output_index * INPUT_DIM..][..INPUT_DIM];
            output.push(codes.iter().zip(&transformed).map(|(&code, &x)| (f64::from(code) * scale + bias) * x).sum());
        }
    }
    output
}

fn run<B: Backend>(
    path: &Path,
    input: &[bf16],
) -> Vec<f32> {
    let context = create_context::<B>();
    let file = File::open(path).expect("open safetensors fixture");
    let loader = ParameterLoader::<B>::new(&file, &context).expect("parse safetensors fixture");
    let linear =
        <dyn Linear<B>>::new(INPUT_DIM as u32, OUTPUT_DIM as u32, DataType::BF16, false, &loader.tree(), &context)
            .expect("load Dash linear");
    loader.assert_all_tensors_read().expect("every Dash tensor is consumed");

    let batch_dim = input.len() / INPUT_DIM;
    let input_buffer = create_buffer_with_data::<B, bf16>(&context, input);
    let mut output_buffer = create_buffer::<B, bf16>(&context, batch_dim * OUTPUT_DIM);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut scratch_input = command_buffer.allocate_scratch(size_of_val(input)).expect("scratch input");
    command_buffer.encode_copy(&input_buffer, &mut scratch_input);
    let output = linear.encode(scratch_input, batch_dim as u32, &mut command_buffer).expect("encode Dash linear");
    command_buffer.encode_copy(&output, &mut output_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run Dash linear");
    buffer_to_vec::<B, bf16>(&output_buffer).into_iter().map(f32::from).collect()
}

#[uzu_test]
fn dash_linear_matches_reference() {
    let mut rng = SmallRng::seed_from_u64(0xDA5B);
    let weights = DashWeights {
        codes: (0..OUTPUT_DIM * INPUT_DIM).map(|_| rng.random_range(0..4u8)).collect(),
        scales: (0..OUTPUT_DIM).map(|_| bf16::from_f32(rng.random_range(0.01f32..0.05))).collect(),
        biases: (0..OUTPUT_DIM).map(|_| bf16::from_f32(rng.random_range(-0.1f32..0.1))).collect(),
        signs: (0..INPUT_DIM)
            .map(|_| {
                if rng.random_bool(0.5) {
                    1
                } else {
                    -1
                }
            })
            .collect(),
    };
    // Kept in the build directory, overwritten on every run.
    let path = std::env::current_exe().unwrap().with_extension("dash_linear_test.safetensors");
    write_safetensors(&path, &weights);

    // Batch 1 takes the GEMV path, batch 64 the GEMM path.
    for batch_dim in [1, 64] {
        let input: Vec<bf16> =
            (0..batch_dim * INPUT_DIM).map(|_| bf16::from_f32(rng.random_range(-2.0f32..2.0))).collect();
        let expected = reference(&weights, &input);
        let rms = (expected.iter().map(|value| value * value).sum::<f64>() / expected.len() as f64).sqrt();

        for_each_backend!(|B| {
            let actual = run::<B>(&path, &input);
            let errors: Vec<f64> =
                actual.iter().zip(&expected).map(|(&actual, &expected)| (f64::from(actual) - expected).abs()).collect();
            let error_rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
            let (worst_index, worst_error) =
                errors.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            // bf16 activations and outputs give ~0.2-0.3% rms error; a layout or transform bug gives ~100%.
            assert!(
                error_rms <= 5e-3 * rms && worst_error <= 5e-2 * rms,
                "{} batch={batch_dim}: error_rms={error_rms}, worst output {worst_index}: actual={}, expected={}, rms={rms}",
                std::any::type_name::<B>(),
                actual[worst_index],
                expected[worst_index]
            );
        });
    }
}
