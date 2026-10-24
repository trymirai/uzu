use std::path::Path;

use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Buffer, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        },
        cpu::Cpu,
    },
    encodable_block::batch_topology::BatchTopology,
    engine::Engine,
    tests::{helpers::for_each_non_cpu_backend, path::get_test_model_path},
    trie::TrieNode,
};

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs every evening at dusk. He trimmed the wick, \
                      polished the great lens and wrote the weather in a leather logbook: wind from the north-east, \
                      a falling barometer, three fishing boats returning before the storm.";
/// Mean and largest KL divergence, in nats, of a backend's next-token distributions from the CPU's: about 3 times the
/// 1.5e-3 mean and 5e-3 to 7e-3 largest that Vulkan measures on the Qwen3.5 0.8B and 4B Mirai checkpoints.
const MEAN_KL_BOUND: f64 = 0.005;
const MAX_KL_BOUND: f64 = 0.025;

/// The next-token log-probabilities after every prompt token, teacher-forced through forward passes of `step` tokens:
/// one prefill of the whole prompt, or decoding steps of one token.
fn log_probabilities<B: Backend>(
    model_path: &Path,
    step: usize,
) -> Vec<Vec<f64>> {
    let engine = block_on(Engine::<B>::new()).expect("engine");
    let model = engine.load_language_model(model_path).expect("language model");
    let ids = model.tokenizer.encode(PROMPT, false).expect("tokens");
    let tokens = ids.get_ids().iter().map(|&id| u64::from(id)).collect::<Vec<_>>();
    let mut state = model.create_empty_state(Some(tokens.len() as u32), 0).expect("state");
    let context = &model.engine.context;
    let mut rows = Vec::new();
    for (index, chunk) in tokens.chunks(step).enumerate() {
        let length = chunk.len() as u32;
        state.transformer_state.prepare(state.transformer_state.context_length(), length).expect("prepare");
        let mut command_buffer =
            context.create_command_buffer(Some("logits"), Some(context.create_allocation_pool())).expect("buffer");
        // The token constants and the logits scratch end before the encoding does, as in the stream's passes.
        let readback = {
            let token_ids = command_buffer
                .allocate_constant_from_slice(&chunk.iter().map(|&token| token as u32).collect::<Box<[u32]>>())
                .expect("token ids");
            let trie = TrieNode::flat(index * step, chunk, &state.prng);
            let nodes = trie.linearize().token_subtrie_ranges().collect::<Box<[_]>>();
            let output = model
                .decoder
                .encode(
                    &token_ids,
                    &BatchTopology::new(&nodes, true),
                    Some((0..length).into()),
                    None,
                    &mut state.transformer_state,
                    &mut command_buffer,
                )
                .expect("decoder");
            let logits = output.logits.expect("logits");
            let mut readback = context.create_readback_buffer(logits.size()).expect("readback buffer");
            command_buffer.encode_copy(&logits, &mut readback);
            readback
        };
        state
            .transformer_state
            .encode_accept(&(0..length).collect::<Box<[u32]>>(), &mut command_buffer)
            .expect("accept");
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("forward pass");
        let logits = readback.as_slice::<bf16>();
        rows.extend(logits.chunks(logits.len() / chunk.len()).map(|row| {
            let row = row.iter().map(|&logit| f64::from(logit.to_f32())).collect::<Vec<_>>();
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = row.iter().map(|&logit| (logit - max).exp()).sum::<f64>().ln() + max;
            row.iter().map(|&logit| logit - log_sum).collect::<Vec<_>>()
        }));
    }
    rows
}

/// Each backend's teacher-forced next-token distributions over a fixed prompt, from one prefill and from one-token
/// decoding steps, stay within a small KL divergence of the CPU prefill's: the model-level bound on the kernels'
/// numerical differences.
#[uzu_test]
#[ignore] // Downloads the test model.
fn next_token_distributions_match_cpu() {
    let model_path = get_test_model_path();
    let reference = log_probabilities::<Cpu>(&model_path, usize::MAX);
    for_each_non_cpu_backend!(|B| {
        for (pass, step) in [("prefill", usize::MAX), ("decode", 1)] {
            let actual = log_probabilities::<B>(&model_path, step);
            assert_eq!(actual.len(), reference.len());
            let divergences = reference
                .iter()
                .zip(&actual)
                .map(|(expected, actual)| expected.iter().zip(actual).map(|(&p, &q)| p.exp() * (p - q)).sum::<f64>())
                .collect::<Vec<_>>();
            let mean = divergences.iter().sum::<f64>() / divergences.len() as f64;
            let max = divergences.iter().copied().fold(0.0, f64::max);
            let backend = std::any::type_name::<B>();
            println!(
                "{backend} {pass}: KL from the CPU over {} positions: mean {mean:.2e}, max {max:.2e}",
                divergences.len()
            );
            assert!(
                mean <= MEAN_KL_BOUND && max <= MAX_KL_BOUND,
                "{backend} {pass}: mean KL {mean:.2e}, max {max:.2e}"
            );
        }
    });
}
