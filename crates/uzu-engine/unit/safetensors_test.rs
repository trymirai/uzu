use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{Backend, CommandBufferEncoding, Context},
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, RHTLinearWrapper},
        weight_matrix::WeightMatrix,
    },
    safetensors::{ParameterLoader, ParameterLoaderError},
    tests::helpers::{
        buffer_readback, buffer_to_vec, create_buffer_with_data, for_each_backend, submit_command_buffer,
    },
};

#[uzu_test]
fn trellis_async_loading_preserves_tensor_checks_and_zeroes_tail() {
    let header = serde_json::json!({
        "codes": {"dtype": "U8", "shape": [2, 32], "data_offsets": [0, 64]},
        "scales": {"dtype": "F32", "shape": [2], "data_offsets": [64, 72]},
        "codebook": {"dtype": "F32", "shape": [5], "data_offsets": [72, 92]}
    });
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    let codes: Vec<u8> = (1..=64).collect();
    file.extend_from_slice(&codes);
    file.extend_from_slice(bytemuck::cast_slice(&[1.0f32; 7]));
    let path = std::env::temp_dir().join(format!("uzu-pr899-loading-{}.safetensors", std::process::id()));
    std::fs::write(&path, file).unwrap();

    for_each_backend!(|B| {
        block_on(async {
            let context = <B as Backend>::Context::new().await.unwrap();
            let loader = ParameterLoader::<B>::new(&path, &context).await.unwrap();
            let tree = loader.tree();
            // A padded allocation must still validate the tensor's unpadded shape and byte count.
            assert!(matches!(
                tree.read_buffer_with_tail("codes", &[2, 33], DataType::U8, 16).await,
                Err(ParameterLoaderError::InvalidTensor { .. })
            ));
            assert!(loader.assert_all_tensors_read().is_err());
            let matrix = WeightMatrix::load(
                2,
                64,
                DataType::BF16,
                serde_json::from_value::<AnyWeightMatrixSpec>(serde_json::json!({
                    "type": "QtipGaussianSpec",
                    "vector_width": 4,
                    "transition_bits": 8,
                    "restart_columns": 64
                }))
                .unwrap(),
                Layout::OutputInput,
                &tree,
            )
            .await
            .unwrap();
            let loaded = context.copyout::<u8>(matrix.values()).await.unwrap();
            assert_eq!(&loaded[..64], &codes);
            assert_eq!(&loaded[64..], &[0; 16]);
            loader.assert_all_tensors_read().unwrap();
        });
    });
    std::fs::remove_file(path).unwrap();
}

#[uzu_test]
fn trellis_model_loading_future_is_send() {
    fn check<B: Backend>(engine: &std::sync::Arc<crate::engine::Engine<B>>) {
        fn assert_send<T: Send>(_: T) {}
        assert_send(engine.load_language_model(std::path::Path::new("unused")));
    }
    // Check the generic backend contract without starting model IO.
    let _ = check::<crate::backends::cpu::Cpu>;
}

#[uzu_test]
fn rht_loading_preserves_input_only_and_rotated_output() {
    for (mode, rotated_output) in [("input", false), ("input_output", false), ("input_output", true)] {
        let quantization = serde_json::json!({
            "type": "IntSpec", "bits": 8, "group_size": 32, "is_symmetric": true, "layout": "output_input"
        });
        let mut header = serde_json::json!({"__metadata__": {
            "weights.spec": serde_json::json!({
                "type": "HybridSpec", "quantization_spec": quantization,
                "adapter_spec": null, "incoherence_block_size": 32,
                "incoherence_processing_mode": mode, "incoherence_kind": "hadamard"
            }).to_string(),
            "weights.quantized.spec": quantization.to_string()
        }});
        let mut data = Vec::new();
        let mut tensor = |name: &str, dtype: &str, shape: &[u32], values: &[u8]| {
            let start = data.len();
            data.extend_from_slice(values);
            header[name] = serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]});
        };
        let mut weights = vec![128u8; 32 * 32];
        for column in 0..32 {
            weights[column * 32 + column] = 129;
        }
        tensor("weights.quantized.weights", "U8", &[32, 32], &weights);
        tensor("weights.quantized.scales", "BF16", &[1, 32], bytemuck::cast_slice(&[bf16::ONE; 32]));
        tensor("weights.incoherence_signs.input_signs", "I32", &[32], bytemuck::cast_slice(&[1i32; 32]));
        if mode == "input_output" {
            tensor("weights.incoherence_signs.output_signs", "I32", &[32], bytemuck::cast_slice(&[-1i32; 32]));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!("uzu-pr899-rht-{}.safetensors", std::process::id()));
        std::fs::write(&path, file).unwrap();
        for_each_backend!(|B| {
            let context = block_on(<B as Backend>::Context::new()).unwrap();
            let loader = block_on(ParameterLoader::<B>::new(&path, &context)).unwrap();
            let linear: Box<dyn Linear<B>> = if rotated_output {
                let result = block_on(RHTLinearWrapper::new_with_rotated_output(
                    32,
                    32,
                    DataType::BF16,
                    &loader.tree(),
                    &context,
                ));
                Box::new(result.unwrap())
            } else {
                block_on(<dyn Linear<B>>::new(32, 32, DataType::BF16, false, &loader.tree(), &context)).unwrap()
            };
            loader.assert_all_tensors_read().unwrap();
            let input = create_buffer_with_data::<B, bf16>(&context, &[bf16::ONE; 32]);
            let mut command = context.create_command_buffer(None, None).unwrap();
            let mut scratch = command.allocate_scratch_for_shape(&[1, 32], DataType::BF16).unwrap();
            command.encode_copy(&input, &mut scratch);
            let output = linear.encode(scratch, 1, &mut command).unwrap();
            submit_command_buffer(command);
            let output = buffer_readback::<B>(&context, &output);
            let actual = buffer_to_vec::<B, bf16>(&context, &output);
            let expected: Vec<bf16> = (0..32)
                .map(|column| {
                    bf16::from_f32(if mode == "input_output" && !rotated_output {
                        -1.0
                    } else if column == 0 {
                        32.0f32.sqrt()
                    } else {
                        0.0
                    })
                })
                .collect();
            assert_eq!(actual, expected, "{} {mode} rotated_output={rotated_output}", B::NAME);
        });
        std::fs::remove_file(path).unwrap();
    }
}
