//! KL of prompt logits: GPU-only (`UZU_ANE=0`) vs the default engine (ANE prefill MLPs).
//!
//! Run:
//!   UZU_PREFILL_MODEL=~/models/Qwen3.5-9B-M \
//!     cargo test --release -p uzu-engine --lib prefill_kl -- --ignored --nocapture

use std::path::PathBuf;

use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, CpuBuffer,
            GlobalBufferCpuGpu, gpu_types::trie::TrieNode as GpuTrieNode,
        },
        metal::Metal,
    },
    encodable_block::batch_topology::BatchTopology,
    engine::{Engine, language_model::LanguageModel},
    trie::TrieNode,
};

/// Logits of every prompt position, `[tokens][vocab]`, from one prefill chunk.
fn prompt_logits(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
) -> Vec<Vec<f32>> {
    let count = tokens.len() as u32;
    let mut state = model.create_empty_state(Some(count + 64), 0).unwrap();
    state.transformer_state.prepare(0, count).unwrap();
    let context = model.engine.context.clone();
    let mut command_buffer = context.create_command_buffer(Some("prompt logits"), None).unwrap();
    let token_ids = command_buffer
        .allocate_constant_from_slice(&tokens.iter().map(|token| *token as u32).collect::<Box<[u32]>>())
        .unwrap();
    let trie = TrieNode::flat(0, tokens, &state.prng);
    let nodes = trie.linearize().token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
    let batch = BatchTopology::new(&nodes, true);
    let output = model
        .decoder
        .encode(&token_ids, &batch, Some((0..count).into()), None, &mut state.transformer_state, &mut command_buffer)
        .unwrap();
    let logits = output.logits.unwrap();
    let mut copy = context.create_buffer_cpu_gpu(logits.size()).unwrap().handoff();
    command_buffer.encode_copy(&logits, &mut copy);
    drop((logits, output.hidden_features, output.final_hidden, token_ids));
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let vocab = model.vocab_size;
    let values: Vec<f32> = if copy.size() == tokens.len() * vocab * 4 {
        copy.as_slice::<f32>().to_vec()
    } else {
        assert_eq!(copy.size(), tokens.len() * vocab * 2, "unexpected logits size");
        copy.as_slice::<bf16>().iter().map(|value| value.to_f32()).collect()
    };
    values.chunks(vocab).map(|row| row.to_vec()).collect()
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = logits.iter().map(|value| (*value as f64 - peak).exp()).sum();
    let log_total = total.ln() + peak;
    logits.iter().map(|value| *value as f64 - log_total).collect()
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// Fixed prompts from files in this repository (prose, Rust, Python, Metal), `tokens` long.
fn prompts(
    model: &LanguageModel<Metal>,
    tokens: usize,
) -> Vec<(String, Vec<u64>)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        "../../README.md",
        "src/backends/metal/command_buffer.rs",
        "../../scripts/ane_prefill/numerics/pipeline_v1.py",
        "src/backends/metal/kernel/ane/ane_ffn.metal",
    ]
    .iter()
    .map(|path| {
        let text = std::fs::read_to_string(root.join(path)).unwrap();
        let mut ids: Vec<u64> =
            model.tokenizer().encode(text.as_str(), true).unwrap().get_ids().iter().map(|id| u64::from(*id)).collect();
        assert!(ids.len() >= tokens, "{path} has only {} tokens", ids.len());
        ids.truncate(tokens);
        (path.to_string(), ids)
    })
    .collect()
}

// Ignored: needs UZU_PREFILL_MODEL; loads the model twice.
#[uzu_test]
#[ignore]
fn prefill_kl() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let tokens: usize = std::env::var("UZU_KL_TOKENS").map_or(1024, |value| value.parse().unwrap());
    // SAFETY: test-only; the variable is read when a Metal context is created.
    unsafe { std::env::set_var("UZU_ANE", "0") };
    let gpu_engine = block_on(Engine::<Metal>::new()).unwrap();
    let gpu = block_on(gpu_engine.load_language_model(&model_path)).unwrap();
    unsafe { std::env::remove_var("UZU_ANE") };
    let ane_engine = block_on(Engine::<Metal>::new()).unwrap();
    let ane = block_on(ane_engine.load_language_model(&model_path)).unwrap();

    let (mut kl_sum, mut positions, mut agree, mut worst) = (0.0f64, 0usize, 0usize, 0.0f64);
    for (name, ids) in prompts(&gpu, tokens) {
        let reference = prompt_logits(&gpu, &ids);
        let candidate = prompt_logits(&ane, &ids);
        let (mut prompt_kl, mut prompt_agree) = (0.0f64, 0usize);
        for (p, q) in reference.iter().zip(&candidate) {
            let (log_p, log_q) = (log_softmax(p), log_softmax(q));
            let kl: f64 = log_p.iter().zip(&log_q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum();
            prompt_kl += kl;
            worst = worst.max(kl);
            prompt_agree += usize::from(argmax(p) == argmax(q));
        }
        println!(
            "{name}: mean KL {:.2e}, top-1 agreement {:.1}% over {} positions",
            prompt_kl / ids.len() as f64,
            100.0 * prompt_agree as f64 / ids.len() as f64,
            ids.len()
        );
        kl_sum += prompt_kl;
        positions += ids.len();
        agree += prompt_agree;
    }
    println!(
        "OVERALL: mean KL {:.2e}, max KL {:.2e}, top-1 agreement {:.2}% ({positions} positions)",
        kl_sum / positions as f64,
        worst,
        100.0 * agree as f64 / positions as f64
    );
}
