use uzu_engine_macros::uzu_test;

use super::{D3InputError, validate_input};

#[uzu_test]
fn d3_input_bounds() {
    assert_eq!(validate_input(&[], 10, None), Err(D3InputError::EmptyInput));
    assert_eq!(validate_input(&[0, 9], 10, Some(2)), Ok(2));
    assert_eq!(
        validate_input(&[0, 10], 10, None),
        Err(D3InputError::InvalidToken {
            position: 1,
            token_id: 10,
            vocab_size: 10
        }),
    );
    assert_eq!(
        validate_input(&[0, 1, 2], 10, Some(2)),
        Err(D3InputError::ContextOverflow {
            input_tokens: 3,
            max_tokens: 2
        }),
    );
}

#[uzu_test]
fn d3_input_accepts_the_whole_prompt_across_1024_and_declared_context_bounds() {
    for length in [1, 1024, 1025, 2048] {
        assert_eq!(validate_input(&vec![9; length], 10, Some(2048)), Ok(length as u32));
    }
    assert_eq!(
        validate_input(&vec![1; 2049], 10, Some(2048)),
        Err(D3InputError::ContextOverflow {
            input_tokens: 2049,
            max_tokens: 2048
        }),
    );
    assert_eq!(validate_input(&vec![1; 1025], 10, None), Ok(1025));
    assert_eq!(
        validate_input(&vec![1; 1025], 10, Some(1024)),
        Err(D3InputError::ContextOverflow {
            input_tokens: 1025,
            max_tokens: 1024
        }),
    );
    let mut prompt = vec![1; 1025];
    prompt[1024] = 10;
    assert_eq!(
        validate_input(&prompt, 10, Some(1025)),
        Err(D3InputError::InvalidToken {
            position: 1024,
            token_id: 10,
            vocab_size: 10
        }),
    );
}

#[cfg(backend = "cpu")]
mod cpu {
    use std::{
        fs::{self, File},
        io::Write,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use half::bf16;
    use serde_json::{Map, Value, json};
    use uzu_engine_macros::uzu_test;

    use super::super::EngineLoadD3DecisionError;
    use crate::{
        backends::{
            common::{
                Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
                Context, Kernels, gpu_types::trie::TrieNode, kernel::TensorCastKernel,
            },
            cpu::Cpu,
        },
        config::{model::d3_decision::tests::config_json, transformer::TransformerConfig},
        data_type::DataType,
        encodable_block::{batch_topology::BatchTopology, linear::Linear, transformer::Transformer},
        engine::Engine,
        safetensors::ParameterLoader,
        tests::helpers::{create_buffer_with_data, create_context},
    };

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "uzu-d3-unit-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&path).expect("create isolated D3 unit directory");
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[uzu_test]
    fn d3_loader_rejects_malformed_config_before_opening_weights() {
        let directory = TestDirectory::new();
        let engine = Engine::<Cpu>::new().expect("CPU engine");
        let config_path = directory.0.join("config.json");
        fs::write(&config_path, b"{not JSON}").unwrap();
        assert!(matches!(engine.load_d3_decision_model(&directory.0), Err(EngineLoadD3DecisionError::Serde(_))));

        // No model.safetensors exists: semantic validation must win over a weight-file I/O error.
        for (pointer, replacement) in [
            ("/activation_dtype", json!("float32")),
            ("/decision/readout_dtype", json!("bfloat16")),
            ("/decision/pooling", json!("mean")),
            ("/decision/max_length", json!(0)),
            ("/transformer_config/layer_configs/3/mixer_config/is_causal", json!(true)),
        ] {
            let mut value = config_json();
            *value.pointer_mut(pointer).unwrap() = replacement;
            fs::write(&config_path, serde_json::to_vec(&value).unwrap()).unwrap();
            match engine.load_d3_decision_model(&directory.0) {
                Err(EngineLoadD3DecisionError::D3(_)) => {},
                Err(other) => panic!("invalid {pointer} reached weight loading: {other}"),
                Ok(_) => panic!("accepted invalid config at {pointer}"),
            }
        }
        fs::write(&config_path, serde_json::to_vec(&config_json()).unwrap()).unwrap();
        assert!(matches!(engine.load_d3_decision_model(&directory.0), Err(EngineLoadD3DecisionError::IO(_))));
    }

    const DIM: usize = 4;
    // Generated attention kernels start at 64 dimensions. Keep the residual
    // stream four-dimensional and pad the attention head with exact zeros.
    const HEAD_DIM: usize = 64;
    const READOUT: [[f32; DIM]; 2] = [[1.0003, -0.33333334, 0.0, 0.0], [0.14285715, 1.0007, 0.0, 0.0]];

    fn add_matrix(
        header: &mut Map<String, Value>,
        metadata: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        prefix: &str,
        shape: [usize; 2],
        dtype: &str,
        bytes: &[u8],
    ) {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            format!("{prefix}.weights"),
            json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()],
            }),
        );
        metadata.insert(
            format!("{prefix}.spec"),
            Value::String(json!({"type": "FullPrecisionSpec", "layout": "output_input"}).to_string()),
        );
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes()).collect()
    }

    fn tiny_weights(directory: &TestDirectory) -> File {
        let mut header = Map::new();
        let mut metadata = Map::new();
        let mut payload = Vec::new();
        let value_identity = (0..HEAD_DIM * DIM)
            .map(|index| {
                if index / DIM == index % DIM {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        // Zero queries/keys give uniform attention. Values and output projections are identities.
        let mut qkv = vec![0.0; 3 * HEAD_DIM * DIM];
        qkv[2 * HEAD_DIM * DIM..].copy_from_slice(&value_identity);
        let output_identity = (0..DIM * HEAD_DIM)
            .map(|index| {
                if index / HEAD_DIM == index % HEAD_DIM {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        for layer in 0..2 {
            for (name, shape, values) in [
                ("mixer.qkvg_projection", [3 * HEAD_DIM, DIM], qkv.clone()),
                ("mixer.out_projection", [DIM, HEAD_DIM], output_identity.clone()),
                ("mlp.up_projection", [2 * DIM, DIM], vec![0.0; 2 * DIM * DIM]),
                ("mlp.down_projection", [DIM, DIM], vec![0.0; DIM * DIM]),
            ] {
                add_matrix(
                    &mut header,
                    &mut metadata,
                    &mut payload,
                    &format!("layers.{layer}.{name}.weights"),
                    shape,
                    "BF16",
                    &bf16_bytes(&values),
                );
            }
        }
        let head_bytes = READOUT.iter().flatten().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        add_matrix(&mut header, &mut metadata, &mut payload, "readout.weights", [2, DIM], "F32", &head_bytes);
        assert_eq!(payload.len(), 4320, "fixture should contain only 4320 weight bytes");
        header.insert("__metadata__".into(), Value::Object(metadata));
        let mut json_header = serde_json::to_vec(&Value::Object(header)).unwrap();
        while json_header.len() % 8 != 0 {
            json_header.push(b' ');
        }
        let path = directory.0.join("model.safetensors");
        let mut file = File::create(&path).unwrap();
        file.write_all(&(json_header.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json_header).unwrap();
        file.write_all(&payload).unwrap();
        drop(file);
        File::open(path).unwrap()
    }

    fn tiny_transformer_config(causal: bool) -> TransformerConfig {
        // Reuse the real schema, but test the public Transformer component rather than
        // bypassing the production D3 validator's 9B geometry and 255-code contract.
        let mut value = config_json()["transformer_config"].clone();
        value["model_dim"] = json!(DIM);
        value["hidden_dim"] = json!(DIM);
        let mut layer = value["layer_configs"][3].clone();
        let mixer = &mut layer["mixer_config"];
        for (key, setting) in [
            ("num_heads", json!(1)),
            ("num_groups", json!(1)),
            ("head_dim", json!(HEAD_DIM)),
            ("has_gate", json!(false)),
            ("is_causal", json!(causal)),
            ("query_norm_config", Value::Null),
            ("key_norm_config", Value::Null),
        ] {
            mixer[key] = setting;
        }
        for field in ["pre_mixer_norm_config", "pre_mlp_norm_config"] {
            layer[field]["has_scale"] = json!(false);
            layer[field]["scale_offset"] = Value::Null;
        }
        value["layer_configs"] = json!([layer.clone(), layer]);
        value["output_norm_config"]["has_scale"] = json!(false);
        value["output_norm_config"]["scale_offset"] = Value::Null;
        serde_json::from_value(value).unwrap()
    }

    struct TinyPipeline {
        context: Arc<<Cpu as Backend>::Context>,
        transformer: Transformer<Cpu>,
        readout: Box<dyn Linear<Cpu>>,
        cast: <<Cpu as Backend>::Kernels as Kernels>::TensorCastKernel,
    }

    impl TinyPipeline {
        fn new(
            file: &File,
            causal: bool,
        ) -> Self {
            let context = create_context::<Cpu>();
            let loader = ParameterLoader::<Cpu>::new(file, context.as_ref()).unwrap();
            let tree = loader.tree();
            let transformer =
                Transformer::new(DataType::BF16, None, &tiny_transformer_config(causal), &tree, &context).unwrap();
            let readout = <dyn Linear<Cpu>>::new_mixed_precision(
                DIM as u32,
                2,
                DataType::F32,
                DataType::F32,
                DataType::F32,
                false,
                &tree.subtree("readout"),
                &context,
            )
            .unwrap();
            let cast =
                <<Cpu as Backend>::Kernels as Kernels>::TensorCastKernel::new(&context, DataType::BF16, DataType::F32)
                    .unwrap();
            loader.assert_all_tensors_read().unwrap();
            Self {
                context,
                transformer,
                readout,
                cast,
            }
        }

        fn forward(
            &self,
            input: &[[f32; DIM]],
        ) -> (Vec<f32>, Vec<f32>) {
            let n = u32::try_from(input.len()).unwrap();
            let data = input.iter().flatten().map(|&value| bf16::from_f32(value)).collect::<Vec<_>>();
            let input_buffer = create_buffer_with_data::<Cpu, bf16>(&self.context, &data);
            let nodes = (0..n)
                .map(|index| TrieNode {
                    trie_start: index,
                    trie_end: n - 1,
                    height: index,
                })
                .collect::<Vec<_>>();
            let mut state = self.transformer.create_empty_state_for_prefill(n, &self.context).unwrap();
            state.prepare(0, n).unwrap();
            let mut command = self.context.create_command_buffer(None, None).unwrap();
            let mut embedded = command.allocate_scratch(input_buffer.size()).unwrap();
            command.encode_copy(&input_buffer, &mut embedded);
            let output = self
                .transformer
                .encode(
                    embedded,
                    None::<&<Cpu as Backend>::ScratchBuffer>,
                    &BatchTopology::new(&nodes, true),
                    Some((n - 1..n).into()),
                    None,
                    Some(&mut state),
                    &mut command,
                )
                .unwrap()
                .output
                .unwrap();
            let mut hidden = command.allocate_scratch(DIM * size_of::<f32>()).unwrap();
            self.cast.encode(&output, &mut hidden, DIM as u32, &mut command);
            let mut head_input = command.allocate_scratch(hidden.size()).unwrap();
            command.encode_copy(&hidden, &mut head_input);
            let logits = self.readout.encode(head_input, 1, &mut command).unwrap();
            command.end_encoding().submit().wait_until_completed().unwrap();
            let hidden = hidden.copyout::<f32>();
            let logits = logits.copyout::<f32>();
            assert_eq!(hidden.len(), DIM);
            assert_eq!(logits.len(), READOUT.len());
            assert!(hidden.iter().chain(&logits).all(|value| value.is_finite()));
            (hidden, logits)
        }
    }

    fn rounded(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn normalized(vector: [f32; DIM]) -> [f32; DIM] {
        let inverse_rms = (vector.iter().map(|v| v * v).sum::<f32>() / DIM as f32 + 1e-6).sqrt().recip();
        vector.map(|value| rounded(value * inverse_rms))
    }

    fn uniform_attention_oracle(
        input: &[[f32; DIM]],
        causal: bool,
    ) -> [f32; DIM] {
        let mut residual = input.iter().map(|row| row.map(rounded)).collect::<Vec<_>>();
        for _ in 0..2 {
            let values = residual.iter().copied().map(normalized).collect::<Vec<_>>();
            residual = residual
                .iter()
                .enumerate()
                .map(|(row, &shortcut)| {
                    let count = if causal {
                        row + 1
                    } else {
                        values.len()
                    };
                    std::array::from_fn(|column| {
                        let mean = values[..count].iter().map(|v| v[column]).sum::<f32>() / count as f32;
                        rounded(shortcut[column] + rounded(mean))
                    })
                })
                .collect();
        }
        normalized(*residual.last().unwrap())
    }

    #[uzu_test]
    fn d3_tiny_two_layer_prefill_is_noncausal() {
        let directory = TestDirectory::new();
        let file = tiny_weights(&directory);
        let input = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let full = TinyPipeline::new(&file, false).forward(&input).0;
        let causal = TinyPipeline::new(&file, true).forward(&input).0;
        for (actual, expected) in full.iter().zip(uniform_attention_oracle(&input, false)) {
            assert!((actual - expected).abs() < 0.02, "noncausal scalar oracle: {actual} != {expected}");
        }
        for (actual, expected) in causal.iter().zip(uniform_attention_oracle(&input, true)) {
            assert!((actual - expected).abs() < 0.02, "causal control oracle: {actual} != {expected}");
        }
        // One layer's last token can see the same prefix in both modes. The second
        // layer exposes whether earlier queries could see future tokens in layer one.
        assert!(full.iter().zip(&causal).any(|(a, b)| (a - b).abs() > 0.02));
    }

    #[uzu_test]
    fn d3_tiny_prefill_accepts_1024_and_1025_and_uses_fresh_state() {
        let directory = TestDirectory::new();
        let file = tiny_weights(&directory);
        let model = TinyPipeline::new(&file, false);
        for length in [1024, 1025] {
            let mut input = vec![[1.0, 0.0, 0.0, 0.0]; length];
            input[length - 1] = [0.0, 1.0, 0.0, 0.0];
            let first = model.forward(&input);
            let unrelated = model.forward(&[[0.0, 0.0, 1.0, 0.0]; 3]);
            assert_ne!(first.0, unrelated.0);
            let repeated = model.forward(&input);
            assert_eq!(first, repeated, "fresh state differs after an unrelated request at {length} tokens");
            for (actual, expected) in first.0.iter().zip(uniform_attention_oracle(&input, false)) {
                assert!((actual - expected).abs() < 0.02, "long prefill scalar oracle: {actual} != {expected}");
            }
        }
    }

    #[uzu_test]
    fn d3_tiny_readout_preserves_fp32_weights_and_logits() {
        let directory = TestDirectory::new();
        let file = tiny_weights(&directory);
        let model = TinyPipeline::new(&file, false);
        let (hidden, logits) = model.forward(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
        for (actual, weights) in logits.iter().zip(READOUT) {
            let expected = weights.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f32>();
            assert!((actual - expected).abs() < 2e-6, "FP32 readout: {actual} != {expected}");
        }
        assert!(logits.iter().any(|&value| (value - rounded(value)).abs() > 1e-5), "readout was rounded back to BF16");
    }
}
