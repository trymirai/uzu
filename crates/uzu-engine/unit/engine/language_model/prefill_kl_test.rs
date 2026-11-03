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
    encodable_block::{batch_topology::BatchTopology, sampling::SamplingMethod},
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
    .filter_map(|path| {
        let text = std::fs::read_to_string(root.join(path)).unwrap();
        let mut ids: Vec<u64> =
            model.tokenizer().encode(text.as_str(), true).unwrap().get_ids().iter().map(|id| u64::from(*id)).collect();
        if ids.len() < tokens {
            println!("skipping {path}: {} tokens, need {tokens}", ids.len());
            return None;
        }
        ids.truncate(tokens);
        Some((path.to_string(), ids))
    })
    .collect()
}

// Ignored: needs UZU_PREFILL_MODEL.
//
// Relative RMS of each ANE-run linear layer (DeltaNet in/out, attention in/out) against the GPU
// linear on the same real activations: the ANE and the GPU path run on each projection's input,
// and the GPU output continues through the model, so errors do not accumulate across layers.
#[uzu_test]
#[ignore]
fn ane_linear_numerics() {
    use std::collections::BTreeMap;

    use crate::{backends::metal::ane::linear::comparison, encodable_block::linear::LinearRole};

    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let tokens: usize = std::env::var("UZU_KL_TOKENS").map_or(1024, |value| value.parse().unwrap());
    let engine = block_on(Engine::<Metal>::new()).unwrap();
    let model = block_on(engine.load_language_model(&model_path)).unwrap();
    comparison::enable(1);

    for (name, ids) in prompts(&model, tokens) {
        comparison::RECORDS.lock().clear();
        prompt_logits(&model, &ids);
        let records = std::mem::take(&mut *comparison::RECORDS.lock());
        let mut per_role: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for record in &records {
            let ane = record.ane.as_slice::<bf16>();
            let gpu = record.gpu.as_slice::<bf16>();
            let (mut error, mut energy) = (0.0f64, 0.0f64);
            for (a, g) in ane.iter().zip(gpu) {
                let (a, g) = (f64::from(a.to_f32()), f64::from(g.to_f32()));
                error += (a - g).powi(2);
                energy += g * g;
            }
            let role = match record.role {
                LinearRole::DeltaNetInput => "deltanet in ",
                LinearRole::DeltaNetOutput => "deltanet out",
                LinearRole::AttentionInput => "attention in ",
                LinearRole::AttentionOutput => "attention out",
            };
            per_role.entry(format!("{role} (N={})", record.output)).or_default().push((error / energy).sqrt());
        }
        println!("{name}:");
        for (role, values) in per_role {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let worst = values.iter().copied().fold(0.0, f64::max);
            println!("  {role}: rel RMS mean {mean:.4}, max {worst:.4} over {} layers", values.len());
        }
    }
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

/// Logits of the last prefill chunk's positions, `[chunk tokens][vocab]`, from a prefill encoded
/// the way the stream does it: one command buffer, `chunk` tokens per decoder pass, the chunks
/// pipelined across lanes when the model allows it.
fn chunked_prompt_logits(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
    chunk: usize,
) -> Vec<Vec<f32>> {
    try_chunked_prompt_logits(model, tokens, chunk).unwrap()
}

/// [`chunked_prompt_logits`] that reports a failed execution instead of panicking.
fn try_chunked_prompt_logits(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
    chunk: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let mut state = model.create_empty_state(Some(tokens.len() as u32 + 64), 0).unwrap();
    let chunk_count = tokens.len().div_ceil(chunk);
    state.transformer_state.prepare(((chunk_count - 1) * chunk) as u32, chunk.min(tokens.len()) as u32).unwrap();
    let context = model.engine.context.clone();
    let mut command_buffer = context.create_command_buffer(Some("chunked prompt logits"), None).unwrap();
    let pipelined = chunk_count > 1 && model.decoder.supports_layerwise_prefill_pipelining();

    let mut token_ids = Vec::new();
    let mut last = None;
    for (index, ids) in tokens.chunks(chunk).enumerate() {
        if pipelined && index > 0 {
            command_buffer.begin_independent_chunk().unwrap();
        }
        token_ids.push(
            command_buffer
                .allocate_constant_from_slice(&ids.iter().map(|token| *token as u32).collect::<Box<[u32]>>())
                .unwrap(),
        );
        let trie = TrieNode::flat(index * chunk, ids, &state.prng);
        let nodes = trie.linearize().token_subtrie_ranges().collect::<Box<[GpuTrieNode]>>();
        let batch = BatchTopology::new(&nodes, true);
        let is_last = index == chunk_count - 1;
        let output = model
            .decoder
            .encode(
                token_ids.last().unwrap(),
                &batch,
                is_last.then(|| (0..ids.len() as u32).into()),
                None,
                &mut state.transformer_state,
                &mut command_buffer,
            )
            .unwrap();
        state
            .transformer_state
            .encode_accept(&(0..ids.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer)
            .unwrap();
        if is_last {
            last = Some((output.logits.unwrap(), ids.len()));
        }
    }
    let (logits, rows) = last.unwrap();
    let mut copy = context.create_buffer_cpu_gpu(logits.size()).unwrap().handoff();
    command_buffer.encode_copy(&logits, &mut copy);
    drop((logits, token_ids));
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).map_err(|error| error.to_string())?;

    let vocab = model.vocab_size;
    let values: Vec<f32> = if copy.size() == rows * vocab * 4 {
        copy.as_slice::<f32>().to_vec()
    } else {
        assert_eq!(copy.size(), rows * vocab * 2, "unexpected logits size");
        copy.as_slice::<bf16>().iter().map(|value| value.to_f32()).collect()
    };
    Ok(values.chunks(vocab).map(|row| row.to_vec()).collect())
}

/// Greedy continuation through the stream, i.e. the real prefill driver.
fn greedy_tokens(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
    count: usize,
) -> Vec<u64> {
    let mut state = model.create_empty_state(Some((tokens.len() + count + 64) as u32), 0).unwrap();
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(tokens, &mut state, options).unwrap();
    let mut output = Vec::new();
    while output.len() < count {
        if let Some(token) = block_on(stream.generate()).unwrap() {
            output.push(token);
        } else {
            break;
        }
    }
    block_on(stream.close()).unwrap();
    output
}

fn kl_divergence(
    reference: &[f32],
    candidate: &[f32],
) -> f64 {
    let (log_p, log_q) = (log_softmax(reference), log_softmax(candidate));
    log_p.iter().zip(&log_q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
}

// Ignored: needs UZU_PREFILL_MODEL; loads the model twice.
//
// A multi-chunk prefill (default 3 chunks of 1024), where the ANE engine pipelines the chunks across
// two command queues: KL of the last chunk's logits against the GPU-only engine, and agreement of
// the greedy continuation produced through the stream.
#[uzu_test]
#[ignore]
fn prefill_kl_pipelined() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let tokens: usize = std::env::var("UZU_KL_TOKENS").map_or(3072, |value| value.parse().unwrap());
    let continuation: usize = 16;
    const CHUNK: usize = 1024;
    // SAFETY: test-only; the variable is read when a Metal context is created.
    unsafe { std::env::set_var("UZU_ANE", "0") };
    let gpu_engine = block_on(Engine::<Metal>::new()).unwrap();
    let gpu = block_on(gpu_engine.load_language_model(&model_path)).unwrap();
    unsafe { std::env::remove_var("UZU_ANE") };
    let ane_engine = block_on(Engine::<Metal>::new()).unwrap();
    let ane = block_on(ane_engine.load_language_model(&model_path)).unwrap();
    assert!(ane.decoder.supports_layerwise_prefill_pipelining(), "the model must allow pipelining");

    let (mut kl_sum, mut positions, mut agree, mut worst) = (0.0f64, 0usize, 0usize, 0.0f64);
    let (mut final_agree, mut prompts_run, mut tokens_agree, mut tokens_total) = (0usize, 0usize, 0usize, 0usize);
    for (name, ids) in prompts(&gpu, tokens) {
        let reference = chunked_prompt_logits(&gpu, &ids, CHUNK);
        let candidate = chunked_prompt_logits(&ane, &ids, CHUNK);
        let mut prompt_kl = 0.0;
        for (p, q) in reference.iter().zip(&candidate) {
            let kl = kl_divergence(p, q);
            prompt_kl += kl;
            worst = worst.max(kl);
            agree += usize::from(argmax(p) == argmax(q));
        }
        let (last_reference, last_candidate) = (reference.last().unwrap(), candidate.last().unwrap());
        final_agree += usize::from(argmax(last_reference) == argmax(last_candidate));

        let reference_tokens = greedy_tokens(&gpu, &ids, continuation);
        let candidate_tokens = greedy_tokens(&ane, &ids, continuation);
        let matching = reference_tokens.iter().zip(&candidate_tokens).filter(|(a, b)| a == b).count();
        println!(
            "{name}: last-chunk mean KL {:.2e}, final-position KL {:.2e}, greedy continuation {matching}/{continuation} equal",
            prompt_kl / reference.len() as f64,
            kl_divergence(last_reference, last_candidate),
        );
        kl_sum += prompt_kl;
        positions += reference.len();
        prompts_run += 1;
        tokens_agree += matching;
        tokens_total += continuation;
    }
    println!(
        "OVERALL ({tokens} tokens, {CHUNK}-token chunks): last-chunk mean KL {:.2e}, max KL {worst:.2e}, top-1 agreement {:.2}% ({positions} positions), final-position top-1 {final_agree}/{prompts_run}, greedy continuation {tokens_agree}/{tokens_total}",
        kl_sum / positions as f64,
        100.0 * agree as f64 / positions as f64,
    );
}
// Ignored: needs UZU_PREFILL_MODEL; loads the model twice.
//
// A prompt of one chunk's worth (default 1024 tokens) is split by the stream into two pipelined
// chunks (768 + 256) on the ANE engine: KL of the final logits of the split prefill against the
// GPU-only single-chunk prefill, and agreement of the greedy continuation produced through the
// stream (single chunk on the GPU-only engine, split chunks on the ANE engine).
#[uzu_test]
#[ignore]
fn prefill_kl_split_single_chunk() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let tokens: usize = std::env::var("UZU_KL_TOKENS").map_or(1024, |value| value.parse().unwrap());
    let continuation: usize = 16;
    // The first chunk of the stream's two-chunk split of 1024 tokens (768 + 256).
    const SPLIT_CHUNK: usize = 768;
    // SAFETY: test-only; the variable is read when a Metal context is created.
    unsafe { std::env::set_var("UZU_ANE", "0") };
    let gpu_engine = block_on(Engine::<Metal>::new()).unwrap();
    let gpu = block_on(gpu_engine.load_language_model(&model_path)).unwrap();
    unsafe { std::env::remove_var("UZU_ANE") };
    let ane_engine = block_on(Engine::<Metal>::new()).unwrap();
    let ane = block_on(ane_engine.load_language_model(&model_path)).unwrap();
    assert!(ane_engine.context.pipelines_prefill_chunks(), "the ANE engine must pipeline prefill chunks");
    assert!(!gpu_engine.context.pipelines_prefill_chunks(), "the GPU-only engine must not");

    let (mut kl_sum, mut positions, mut agree, mut worst) = (0.0f64, 0usize, 0usize, 0.0f64);
    let (mut final_agree, mut prompts_run, mut tokens_agree, mut tokens_total) = (0usize, 0usize, 0usize, 0usize);
    for (name, ids) in prompts(&gpu, tokens) {
        let reference = chunked_prompt_logits(&gpu, &ids, 1024);
        let candidate = chunked_prompt_logits(&ane, &ids, SPLIT_CHUNK);
        // Rows of the candidate's last (shorter or equal) chunk line up with the reference's tail.
        let offset = reference.len() - candidate.len();
        let mut prompt_kl = 0.0;
        for (p, q) in reference[offset..].iter().zip(&candidate) {
            let kl = kl_divergence(p, q);
            prompt_kl += kl;
            worst = worst.max(kl);
            agree += usize::from(argmax(p) == argmax(q));
        }
        final_agree += usize::from(argmax(reference.last().unwrap()) == argmax(candidate.last().unwrap()));

        let reference_tokens = greedy_tokens(&gpu, &ids, continuation);
        let candidate_tokens = greedy_tokens(&ane, &ids, continuation);
        let matching = reference_tokens.iter().zip(&candidate_tokens).filter(|(a, b)| a == b).count();
        println!(
            "{name}: tail mean KL {:.2e}, final-position KL {:.2e}, greedy continuation {matching}/{continuation} equal",
            prompt_kl / candidate.len() as f64,
            kl_divergence(reference.last().unwrap(), candidate.last().unwrap()),
        );
        kl_sum += prompt_kl;
        positions += candidate.len();
        prompts_run += 1;
        tokens_agree += matching;
        tokens_total += continuation;
    }
    println!(
        "OVERALL ({tokens} tokens, split in two): tail mean KL {:.2e}, max KL {worst:.2e}, top-1 agreement {:.2}% ({positions} positions), final-position top-1 {final_agree}/{prompts_run}, greedy continuation {tokens_agree}/{tokens_total}",
        kl_sum / positions as f64,
        100.0 * agree as f64 / positions as f64,
    );
}

/// [`greedy_tokens`] that reports errors instead of panicking.
fn try_greedy_tokens(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
    count: usize,
) -> Result<Vec<u64>, String> {
    let mut state = model.create_empty_state(Some((tokens.len() + count + 64) as u32), 0).unwrap();
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(tokens, &mut state, options).map_err(|error| format!("{error:?}"))?;
    let mut output = Vec::new();
    while output.len() < count {
        match block_on(stream.generate()).map_err(|error| format!("{error:?}"))? {
            Some(token) => output.push(token),
            None => break,
        }
    }
    let _ = block_on(stream.close());
    Ok(output)
}

/// Runs `work`, failing the process (a stuck GPU queue cannot be unwound) if it takes over 3 minutes.
fn within_deadline<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let _ = sender.send(work());
        });
        receiver.recv_timeout(std::time::Duration::from_secs(180)).unwrap_or_else(|_| {
            println!("HANG: no result within 180 s");
            std::process::exit(2);
        })
    })
}

// Ignored: needs UZU_PREFILL_MODEL; loads the model repeatedly.
//
// Failure injection (`UZU_ANE_FAULT=enqueue:K` / `completion:K`, see `AneFfn`) for a single-chunk
// (1024 tokens) and a pipelined (3072 tokens) prefill, driven both through the decoder directly and
// through the stream. Per case: no hang, the failed prefill reports an error (not silently wrong
// logits), the ANE retired, and the next prefill on the same engine matches the GPU-only engine.
#[uzu_test]
#[ignore]
fn prefill_fault() {
    let model_path = PathBuf::from(std::env::var("UZU_PREFILL_MODEL").expect("set UZU_PREFILL_MODEL"));
    let continuation = 16;
    // SAFETY: test-only; the variables are read when a Metal context / the ANE executor is created.
    unsafe { std::env::set_var("UZU_ANE", "0") };
    let gpu_engine = block_on(Engine::<Metal>::new()).unwrap();
    let gpu = block_on(gpu_engine.load_language_model(&model_path)).unwrap();
    unsafe { std::env::remove_var("UZU_ANE") };

    let kinds = std::env::var("UZU_FAULT_KINDS").unwrap_or_else(|_| "enqueue,completion".to_string());
    let lengths: Vec<usize> = std::env::var("UZU_FAULT_LENGTHS")
        .unwrap_or_else(|_| "1024,3072".to_string())
        .split(',')
        .map(|length| length.parse().unwrap())
        .collect();
    let (_, ids) = prompts(&gpu, 3072).into_iter().next().expect("a prompt with 3072 tokens");
    for kind in kinds.split(',') {
        for length in &lengths {
            let ids = &ids[..*length];
            // 32 MLPs per chunk: fail in the middle of the first (single) or second (pipelined) chunk.
            let at = if *length > 1024 {
                32 + 10
            } else {
                10
            };
            let reference = chunked_prompt_logits(&gpu, ids, 1024);
            let reference_tokens = greedy_tokens(&gpu, ids, continuation);
            let mean_kl = |candidate: &[Vec<f32>]| {
                reference.iter().zip(candidate).map(|(p, q)| kl_divergence(p, q)).sum::<f64>() / reference.len() as f64
            };
            let equal = |tokens: &Result<Vec<u64>, String>| match tokens {
                Ok(tokens) => {
                    let matching = tokens.iter().zip(&reference_tokens).filter(|(a, b)| a == b).count();
                    format!("{matching}/{continuation} equal")
                },
                Err(error) => format!("error ({error})"),
            };
            let load = || {
                unsafe { std::env::set_var("UZU_ANE_FAULT", format!("{kind}:{at}")) };
                let engine = block_on(Engine::<Metal>::new()).unwrap();
                let model = block_on(engine.load_language_model(&model_path)).unwrap();
                unsafe { std::env::remove_var("UZU_ANE_FAULT") };
                (engine, model)
            };

            // Through the decoder: the fault fires in the first prefill.
            let (engine, model) = load();
            let failed = within_deadline(|| try_chunked_prompt_logits(&model, ids, 1024));
            let retired = !engine.context.ane_pipelining_available();
            let after = within_deadline(|| try_chunked_prompt_logits(&model, ids, 1024));
            println!(
                "{kind}:{at} {length} tokens, decoder: failed prefill {}, ANE retired {retired}, next prefill {}",
                failed.map_or_else(
                    |error| format!("error ({error})"),
                    |logits| format!("OK, KL {:.2e}", mean_kl(&logits))
                ),
                after.map_or_else(
                    |error| format!("error ({error})"),
                    |logits| format!("OK, KL {:.2e}", mean_kl(&logits))
                ),
            );
            drop((model, engine));

            // Through the stream.
            let (engine, model) = load();
            let first = within_deadline(|| try_greedy_tokens(&model, ids, continuation));
            let retired = !engine.context.ane_pipelining_available();
            let second = within_deadline(|| try_greedy_tokens(&model, ids, continuation));
            println!(
                "{kind}:{at} {length} tokens, stream: first request {}, ANE retired {retired}, second request {}",
                equal(&first),
                equal(&second)
            );
        }
    }
}
