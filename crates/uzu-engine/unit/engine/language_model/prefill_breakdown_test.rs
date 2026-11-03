//! Prefill breakdown harness (ANE prefill experiment, branch `ane-prefill`).
//!
//! Run:
//!   UZU_PREFILL_MODEL=~/models/Qwen3.5-9B-M UZU_PREFILL_LENGTHS=1024,2048,4096 \
//!     cargo test --release -p uzu-engine --lib prefill_breakdown -- --ignored --nocapture
//!
//! Reports wall-clock prefill tok/s (without timestamps) and a GPU-time breakdown per
//! module class (with timestamps), aggregated over layers.

use std::{collections::BTreeMap, path::PathBuf, sync::mpsc::channel, time::Instant};

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{common::TimestampSpan, metal::Metal},
    encodable_block::sampling::SamplingMethod,
    engine::{Engine, language_model::LanguageModel},
};

const RUNS: usize = 3;

fn prompt(length: usize) -> Vec<u64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            1000 + state % 100_000
        })
        .collect()
}

fn prefill(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
    timestamps: bool,
) -> (f64, Vec<TimestampSpan>) {
    let mut state = model.create_empty_state(Some(tokens.len() as u32 + 64), 0).unwrap();
    let (sender, receiver) = channel();
    let mut options = model.default_stream_options();
    if timestamps {
        options.timestamps = Some(sender);
    }
    let started = Instant::now();
    let mut stream = model.stream(tokens, &mut state, options).unwrap();
    block_on(stream.generate()).unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    block_on(stream.close()).unwrap();
    drop(stream);
    let spans = receiver.try_iter().flat_map(|spans| spans.into_vec()).collect();
    (elapsed, spans)
}

/// "decoder/transformer layer 7/mlp (dense)/linear (rht)/Matmul" -> "layer/mlp (dense)/linear (rht)/Matmul"
fn class_of(name: &str) -> String {
    name.split('/')
        .filter(|part| *part != "decoder")
        .map(|part| {
            if part.starts_with("transformer layer ") {
                "layer"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn report(
    length: usize,
    spans: &[TimestampSpan],
) {
    let Some(first) = spans.iter().map(|span| span.start).min() else {
        println!("  no timestamps recorded");
        return;
    };
    let last = spans.iter().map(|span| span.end).max().unwrap();
    let gpu_total = last.duration_since(first).as_secs_f64();

    let mut per_class: BTreeMap<String, f64> = BTreeMap::new();
    for span in spans {
        *per_class.entry(class_of(&span.name)).or_default() += span.end.duration_since(span.start).as_secs_f64();
    }

    let layer_total = per_class.get("layer").copied().unwrap_or(0.0);
    println!(
        "  GPU span total {:.1} ms ({:.0} tok/s), layers {:.1} ms",
        gpu_total * 1e3,
        length as f64 / gpu_total,
        layer_total * 1e3
    );
    let mut rows: Vec<_> = per_class.iter().filter(|(class, _)| class.split('/').count() <= 3).collect();
    rows.sort_by(|a, b| b.1.total_cmp(a.1));
    for (class, seconds) in rows {
        println!("    {:>9.1} ms {:>5.1}%  {class}", seconds * 1e3, 100.0 * seconds / gpu_total);
    }
}

// Ignored: needs UZU_PREFILL_MODEL; manual perf harness.
#[uzu_test]
#[ignore]
fn prefill_breakdown() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let lengths: Vec<usize> = std::env::var("UZU_PREFILL_LENGTHS")
        .unwrap_or_else(|_| "1024,2048,4096".to_string())
        .split(',')
        .map(|length| length.trim().parse().unwrap())
        .collect();

    let engine = block_on(Engine::<Metal>::new()).unwrap();
    let model = block_on(engine.load_language_model(&model_path)).unwrap();

    // Warm up pipelines.
    prefill(&model, &prompt(lengths[0]), false);

    for length in lengths {
        let tokens = prompt(length);
        let best = (0..RUNS).map(|_| prefill(&model, &tokens, false).0).fold(f64::INFINITY, f64::min);
        println!(
            "prefill {length} tokens: wall {:.1} ms -> {:.0} tok/s (best of {RUNS})",
            best * 1e3,
            length as f64 / best
        );
        let (_, spans) = prefill(&model, &tokens, true);
        report(length, &spans);
    }
}

/// Greedy-decodes a short continuation of a fixed text prompt; prints it for eyeballing.
pub fn greedy_continuation(
    model: &LanguageModel<Metal>,
    text: &str,
    max_tokens: usize,
) -> (Vec<u64>, String) {
    let tokens: Vec<u64> =
        model.tokenizer().encode(text, true).unwrap().get_ids().iter().map(|id| u64::from(*id)).collect();
    let mut state = model.create_empty_state(Some(tokens.len() as u32 + max_tokens as u32 + 64), 0).unwrap();
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(&tokens, &mut state, options).unwrap();
    let mut output = Vec::new();
    while output.len() < max_tokens {
        if let Some(token) = block_on(stream.generate()).unwrap() {
            output.push(token);
        } else {
            break;
        }
    }
    block_on(stream.close()).unwrap();
    drop(stream);
    let ids: Vec<u32> = output.iter().map(|id| *id as u32).collect();
    let text = model.tokenizer().decode(&ids, false).unwrap();
    (output, text)
}

// Ignored: needs UZU_PREFILL_MODEL; manual coherence check.
#[uzu_test]
#[ignore]
fn prefill_coherence() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let engine = block_on(Engine::<Metal>::new()).unwrap();
    let model = block_on(engine.load_language_model(&model_path)).unwrap();
    let long_context = "The quick brown fox jumps over the lazy dog. ".repeat(150);
    for text in [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        &format!("{long_context}\nQuestion: which animal jumps over the dog? Answer:"),
    ] {
        let (_, continuation) = greedy_continuation(&model, text, 32);
        println!("--- prompt ({} chars) ---\n{continuation:?}", text.len());
    }
}
