use std::sync::Arc;

use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Buffer, CommandBufferEncoding, Context, gpu_types::trie::TrieNode},
        cpu::Cpu,
    },
    config::{
        linear::LinearConfig,
        normalization::{NormalizationConfig, UpcastMode},
        residual::gated::GatedResidualConfig,
        stream_op::ngram_injection::{NGramHashConfig, NGramInjectionConfig},
        token_mixer::convolutions::SeparableCausalConvConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        stream_op::ngram_injection::{NGramInjection, NGramInjectionState, hash, num_heads},
    },
    safetensors::ParameterLoader,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, submit_command_buffer},
    },
};

const EOS: u32 = 7;
const NGRAM_SIZE: u32 = 3;
const HEADS_PER_NGRAM: u32 = 2;
const MULTIPLIERS: [u64; 3] = [27696880803969, 9426225827651, 20661530663711];
const HEAD_VOCAB_SIZES: [u32; 4] = [1031, 1033, 1039, 1049];
const TOKENS: [u32; 11] = [151935, 42, 9001, EOS, 5, 151000, 77, 3, EOS, EOS, 1234];
// lalamo `NGramHashConfig.ngram_ids` with an all-EOS history, vocab 151936, PLE layer 1, seed 1234.
const EXPECTED_IDS: [[u32; 4]; 11] = [
    [802, 1111, 2883, 3854],
    [900, 1477, 3035, 3350],
    [356, 1160, 2207, 3447],
    [902, 1892, 2187, 3904],
    [388, 1383, 2988, 3696],
    [217, 1520, 2805, 3272],
    [607, 1929, 2852, 3738],
    [331, 1469, 2076, 3870],
    [974, 1253, 2177, 3660],
    [625, 1870, 2996, 4100],
    [612, 1931, 2251, 3353],
];

fn hasher() -> NGramHashConfig {
    NGramHashConfig {
        multipliers: MULTIPLIERS.into(),
        head_vocab_sizes: HEAD_VOCAB_SIZES.into(),
        ngram_size: NGRAM_SIZE,
        heads_per_ngram: HEADS_PER_NGRAM,
        eos_token_id: EOS,
        token_vocab_size: 69420,
    }
}

#[uzu_test]
fn test_ngram_hasher_matches_lalamo() {
    let num_heads = num_heads(&hasher());
    let mut ids = vec![0; TOKENS.len() * num_heads as usize];
    hash(&hasher(), &[], &TOKENS, None, &mut ids);
    assert_eq!(ids, EXPECTED_IDS.iter().flatten().copied().collect::<Vec<_>>());
}

#[uzu_test]
fn test_ngram_hasher_history_matches_full_sequence() {
    let num_heads = num_heads(&hasher());
    let split = 5;
    let mut ids = vec![0; (TOKENS.len() - split) * num_heads as usize];
    hash(&hasher(), &TOKENS[..split], &TOKENS[split..], None, &mut ids);
    assert_eq!(ids, EXPECTED_IDS[split..].iter().flatten().copied().collect::<Vec<_>>());
}

#[uzu_test]
fn test_ngram_hasher_follows_parents() {
    let num_heads = num_heads(&hasher()) as usize;
    // A trie with two branches below the first token: the flat sequence and a different one.
    let tokens = [TOKENS[0], TOKENS[1], 4242, TOKENS[2], 4343];
    let parents = [-1, 0, 0, 1, 2];
    let mut ids = vec![0; tokens.len() * num_heads];
    hash(&hasher(), &[], &tokens, Some(&parents), &mut ids);
    let node = |index: usize| &ids[index * num_heads..(index + 1) * num_heads];

    assert_eq!(node(0), EXPECTED_IDS[0]);
    assert_eq!(node(1), EXPECTED_IDS[1]);
    assert_eq!(node(3), EXPECTED_IDS[2]);

    let mut branch_ids = vec![0; 3 * num_heads];
    hash(&hasher(), &[], &[TOKENS[0], 4242, 4343], None, &mut branch_ids);
    assert_eq!(node(2), &branch_ids[num_heads..2 * num_heads]);
    assert_eq!(node(4), &branch_ids[2 * num_heads..]);
}

const MODEL_DIM: u32 = 16;
const STREAM_COUNT: u32 = 2;
const EMBEDDING_DIM: u32 = 16;
const CONV_KERNEL_SIZE: u32 = 3;
const STREAM_WIDTH: usize = (STREAM_COUNT * MODEL_DIM) as usize;

fn sines(
    count: usize,
    seed: f32,
) -> Vec<f32> {
    (0..count).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
}

/// An [`NGramInjection`] over random f32 weights on the CPU backend.
struct Block {
    context: Arc<<Cpu as Backend>::Context>,
    block: NGramInjection<Cpu>,
}

impl Block {
    fn new() -> Self {
        let num_rows = HEAD_VOCAB_SIZES.iter().sum::<u32>();
        let channels = STREAM_COUNT * MODEL_DIM;
        let tensors = [
            ("ngram_embedding.weights", vec![num_rows, EMBEDDING_DIM / num_heads(&hasher())]),
            ("key_projection.weights.weights", vec![channels, EMBEDDING_DIM]),
            ("value_projection.weights.weights", vec![MODEL_DIM, EMBEDDING_DIM]),
            ("key_norm.scales", vec![STREAM_COUNT, MODEL_DIM]),
            ("query_norm.scales", vec![STREAM_COUNT, MODEL_DIM]),
            ("conv_norm.scales", vec![STREAM_COUNT, MODEL_DIM]),
            ("conv.weights", vec![channels, CONV_KERNEL_SIZE]),
        ];
        let spec = r#"{"type": "FullPrecisionSpec", "layout": "output_input"}"#;
        let mut header = serde_json::json!({"__metadata__": {
            "key_projection.weights.spec": spec,
            "value_projection.weights.spec": spec,
        }});
        let mut data = Vec::new();
        for (seed, (name, shape)) in tensors.into_iter().enumerate() {
            let values = sines(shape.iter().product::<u32>() as usize, seed as f32);
            let start = data.len();
            data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
            header[name] =
                serde_json::json!({"dtype": DataType::F32, "shape": shape, "data_offsets": [start, data.len()]});
        }
        let header = serde_json::to_vec(&header).expect("safetensors header");
        let path = std::env::temp_dir().join(format!("uzu-ngram-injection-{}.safetensors", std::process::id()));
        std::fs::write(&path, [(header.len() as u64).to_le_bytes().as_slice(), &header, &data].concat())
            .expect("write safetensors");

        let norm_config = NormalizationConfig {
            epsilon: 1e-6,
            scale_offset: None,
            upcast_mode: UpcastMode::OnlyNormalization,
            subtract_mean: false,
            has_scale: true,
            has_biases: false,
        };
        // Both configs carry a private `type` tag, so they can only be built through serde.
        let config = serde_json::from_value::<NGramInjectionConfig>(serde_json::json!({
            "type": "NGramInjectionConfig",
            "hash_config": hasher(),
            "linear_config": LinearConfig {},
            "norm_config": norm_config,
            "conv_config": SeparableCausalConvConfig {
                has_biases: false,
            },
            "embedding_dim": EMBEDDING_DIM,
            "conv_kernel_size": CONV_KERNEL_SIZE,
        }))
        .expect("n-gram injection config");
        let residual_config = serde_json::from_value::<GatedResidualConfig>(serde_json::json!({
            "type": "GatedResidualConfig",
            "stream_count": STREAM_COUNT,
            "low_rank_dim": 1,
            "norm_config": norm_config,
            "linear_config": LinearConfig {},
        }))
        .expect("gated residual config");

        let context = create_context::<Cpu>();
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let block = runtime.block_on(async {
            let loader = ParameterLoader::<Cpu>::new(&path, &context).await.expect("parameter loader");
            let block =
                NGramInjection::new(MODEL_DIM, DataType::F32, &config, &residual_config, &loader.tree(), &context)
                    .await
                    .unwrap_or_else(|_| panic!("failed to load n-gram injection"));
            loader.assert_all_tensors_read().expect("every tensor is used");
            block
        });
        std::fs::remove_file(&path).expect("remove safetensors");
        Self {
            context,
            block,
        }
    }

    /// Runs `tokens` through prepare and encode, returns the output stream,
    /// `[tokens, STREAM_WIDTH]`, then accepts `accepted`.
    fn step(
        &self,
        state: &mut NGramInjectionState<Cpu>,
        tokens: &[u32],
        parents: Option<&[i32]>,
        accepted: &[u32],
    ) -> Vec<f32> {
        let num_tokens = tokens.len() as u32;
        self.block.prepare(state, tokens, parents).expect("prepare");

        let batch_dim = match parents {
            None => BatchTopology::flat(num_tokens),
            Some(parents) => {
                let nodes = parents
                    .iter()
                    .enumerate()
                    .map(|(node, &parent)| TrieNode {
                        trie_start: node as u32,
                        trie_end: node as u32,
                        height: 0,
                        parent,
                    })
                    .collect::<Vec<_>>();
                let nodes = create_buffer_with_data::<Cpu, u32>(&self.context, bytemuck::cast_slice(&nodes));
                BatchTopology::trie(nodes, num_tokens, false)
            },
        };

        // Each token's input row depends only on the token, so the same token
        // gets the same input in the trie and in the flat runs.
        let input = tokens.iter().flat_map(|&token| sines(STREAM_WIDTH, token as f32)).collect::<Vec<_>>();
        let mut output = create_buffer::<Cpu, f32>(&self.context, input.len());

        let mut command_buffer = self.context.create_command_buffer(None, None).expect("command buffer");
        let input = command_buffer.allocate_constant_from_slice(&input).expect("constant");
        let mut stream = command_buffer.allocate_scratch(input.size()).expect("scratch");
        command_buffer.encode_copy(&input, &mut stream);
        drop(input);
        let stream = self
            .block
            .encode(stream, &batch_dim, state, &mut command_buffer)
            .unwrap_or_else(|_| panic!("failed to encode n-gram injection"));
        command_buffer.encode_copy(&stream, &mut output);
        drop(stream);
        submit_command_buffer(command_buffer);

        let mut command_buffer = self.context.create_command_buffer(None, None).expect("command buffer");
        state.encode_accept(accepted, &mut command_buffer).expect("accept");
        submit_command_buffer(command_buffer);

        buffer_to_vec::<Cpu, f32>(&self.context, &output)
    }
}

/// Accepting a path of a trie must leave the state exactly as running that
/// path as a flat batch would: same outputs on the path, same history and
/// conv state for the next token.
#[uzu_test]
fn test_ngram_injection_trie_accept_matches_flat_path() {
    let block = Block::new();
    let row = |stream: &[f32], index: usize| stream[index * STREAM_WIDTH..(index + 1) * STREAM_WIDTH].to_vec();

    // 30 -> { 41 -> 53, 42 -> 52 }. The accepted path skips the first branch
    // and ends before the last node, so rejected tokens sit both inside and
    // after it in batch order.
    let trie_tokens = [30, 41, 42, 52, 53];
    let parents = [-1, 0, 0, 2, 1];
    let accepted = [0, 2, 3];
    let path = accepted.map(|node| trie_tokens[node as usize]);

    let mut trie_state = block.block.create_empty_state(&block.context).expect("state");
    block.step(&mut trie_state, &[10, 20], None, &[0, 1]);
    let trie_stream = block.step(&mut trie_state, &trie_tokens, Some(&parents), &accepted);
    let trie_next = block.step(&mut trie_state, &[60], None, &[0]);

    let mut flat_state = block.block.create_empty_state(&block.context).expect("state");
    block.step(&mut flat_state, &[10, 20], None, &[0, 1]);
    let flat_stream = block.step(&mut flat_state, &path, None, &[0, 1, 2]);
    let flat_next = block.step(&mut flat_state, &[60], None, &[0]);

    for (position, &node) in accepted.iter().enumerate() {
        assert_eq_float::<f32>(
            &row(&flat_stream, position),
            &row(&trie_stream, node as usize),
            1e-5,
            &format!("trie node {node} differs from position {position} of the flat path"),
        );
    }
    assert_eq!(trie_state.history, flat_state.history, "history after accepting the path");
    assert_eq_float::<f32>(&flat_next, &trie_next, 1e-5, "token after the accepted path");
}
