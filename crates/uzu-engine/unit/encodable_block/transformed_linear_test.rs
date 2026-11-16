use std::path::Path;

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde_json::{Value, json};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{Backend, Buffer, CommandBufferEncoding, Context},
    data_type::DataType,
    encodable_block::linear::Linear,
    safetensors::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{
            SafetensorsWriter, buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_backend,
            submit_command_buffer,
        },
    },
};

const INPUT_DIM: u32 = 1024;
const PARTS: [(u32, u32, u32); 3] = [(512, 4, 32), (256, 8, 64), (512, 4, 32)];

fn write_checkpoint(
    path: &Path,
    output_dim: u32,
) {
    let mut rng = SmallRng::seed_from_u64(17);
    let int_spec = |bits: u32, group_size: u32| {
        json!({
            "type": "IntSpec",
            "bits": bits,
            "group_size": group_size,
            "is_symmetric": bits == 8,
            "layout": "output_input",
        })
    };
    let hybrid_spec = |quantization_spec: Value| {
        json!({
            "type": "HybridSpec",
            "quantization_spec": quantization_spec,
            "adapter_spec": null,
            "incoherence_spec": {"type": "HadamardSpec", "block_size": 32, "processing_mode": "input_output"},
        })
    };
    let parts = PARTS.map(|(rows, bits, group_size)| json!({"rows": rows, "spec": int_spec(bits, group_size)}));
    let mut checkpoint = SafetensorsWriter::default();
    checkpoint
        .add_metadata("stack.weights.spec".into(), &hybrid_spec(json!({"type": "OutputStackSpec", "parts": parts})));

    let mut signs = |count: u32| (0..count).map(|_| rng.random_range(0..2) * 2 - 1).collect::<Vec<i32>>();
    let input_signs = signs(INPUT_DIM);
    let output_signs = signs(output_dim);
    checkpoint.add_tensor(
        "stack.weights.incoherence_signs.input_signs".into(),
        "I32",
        &[INPUT_DIM],
        bytemuck::cast_slice(&input_signs),
    );
    checkpoint.add_tensor(
        "stack.weights.incoherence_signs.output_signs".into(),
        "I32",
        &[output_dim],
        bytemuck::cast_slice(&output_signs),
    );
    let mut first_row = 0;
    for (index, (rows, bits, group_size)) in PARTS.into_iter().enumerate() {
        let groups = INPUT_DIM / group_size;
        let codes = (0..rows * INPUT_DIM * bits / 8).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
        let scales = (0..groups * rows).map(|_| bf16::from_f32(rng.random_range(0.001..0.01))).collect::<Vec<_>>();
        let zero_points = (0..groups * rows / 2).map(|_| rng.random::<u8>()).collect::<Vec<_>>();
        let reference = format!("reference.{index}.weights");
        checkpoint.add_metadata(format!("{reference}.spec"), &hybrid_spec(int_spec(bits, group_size)));
        checkpoint.add_tensor(
            format!("{reference}.incoherence_signs.input_signs"),
            "I32",
            &[INPUT_DIM],
            bytemuck::cast_slice(&input_signs),
        );
        checkpoint.add_tensor(
            format!("{reference}.incoherence_signs.output_signs"),
            "I32",
            &[rows],
            bytemuck::cast_slice(&output_signs[first_row as usize..(first_row + rows) as usize]),
        );
        for prefix in [format!("stack.weights.quantized.parts.{index}"), format!("{reference}.quantized")] {
            checkpoint.add_tensor(format!("{prefix}.weights"), "U8", &[rows, INPUT_DIM * bits / 8], &codes);
            checkpoint.add_tensor(format!("{prefix}.scales"), "BF16", &[groups, rows], bytemuck::cast_slice(&scales));
            if bits == 4 {
                checkpoint.add_tensor(format!("{prefix}.zero_points"), "U8", &[groups, rows / 2], &zero_points);
            }
        }
        first_row += rows;
    }
    checkpoint.write(path);
}

fn run<B: Backend>(
    context: &B::Context,
    linear: &dyn Linear<B>,
    input: &[bf16],
    batch_dim: u32,
) -> Vec<bf16> {
    let source = create_buffer_with_data::<B, bf16>(context, input);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    let mut values = command_buffer.allocate_scratch(source.size()).expect("input scratch");
    command_buffer.encode_copy(&source, &mut values);
    let output = linear.encode(values, batch_dim, &mut command_buffer).expect("linear encode");
    let mut result = create_buffer::<B, u8>(context, output.size());
    command_buffer.encode_copy(&output, &mut result);
    submit_command_buffer(command_buffer);
    buffer_to_vec(context, &result)
}

#[uzu_test]
fn hadamard_stack_matches_concatenated_parts() {
    let path = std::env::temp_dir().join(format!("uzu-hadamard-stack-{}.safetensors", std::process::id()));
    let output_dim = PARTS.iter().map(|&(rows, ..)| rows).sum::<u32>();
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    for_each_backend!(|B| {
        let context = create_context::<B>();
        write_checkpoint(&path, output_dim);
        let (stack, references) = runtime.block_on(async {
            let loader = ParameterLoader::<B>::new(&path, &context).await.expect("parameter loader");
            let tree = loader.tree();
            let (stack, preparation) = <dyn Linear<B>>::new_with_input_rht(
                INPUT_DIM,
                output_dim,
                DataType::BF16,
                false,
                &tree.subtree("stack"),
                &context,
            )
            .await
            .expect("stacked linear");
            assert!(preparation.is_none());
            let mut references = Vec::new();
            for (index, &(rows, ..)) in PARTS.iter().enumerate() {
                let tree = tree.subtree(&format!("reference.{index}"));
                references.push(
                    <dyn Linear<B>>::new(INPUT_DIM, rows, DataType::BF16, false, &tree, &context)
                        .await
                        .expect("reference linear"),
                );
            }
            (stack, references)
        });
        std::fs::remove_file(&path).expect("remove safetensors");
        let mut rng = SmallRng::seed_from_u64(23);
        for batch_dim in [1, 4, 16, 64] {
            let input =
                (0..batch_dim * INPUT_DIM).map(|_| bf16::from_f32(rng.random_range(-1.0..1.0))).collect::<Vec<_>>();
            let actual = run::<B>(&context, stack.as_ref(), &input, batch_dim);
            let parts = references
                .iter()
                .map(|linear| run::<B>(&context, linear.as_ref(), &input, batch_dim))
                .collect::<Vec<_>>();
            let expected = (0..batch_dim as usize)
                .flat_map(|row| {
                    PARTS.iter().zip(&parts).flat_map(move |(&(rows, ..), part)| {
                        part[row * rows as usize..(row + 1) * rows as usize].iter().copied()
                    })
                })
                .collect::<Vec<_>>();
            let label = format!("{} batch {batch_dim}", std::any::type_name::<B>());
            assert_eq_float(&expected, &actual, 0.125, &label);
        }
    });
}
