//! Env-driven greedy eval for the LFM3 port: drives the engine exactly like the
//! MLX reference (raw prompt text, temperature 0, fixed token budget) and prints
//! a JSON record with token ids, text and timings.
//!
//! Env: LFM3_MODEL (bundle path), LFM3_PROMPT (@file or text),
//! LFM3_MAX_TOKENS (default 128), LFM3_RUNS (default 1).

use std::time::Instant;

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::backends::{
    cpu::Cpu,
    metal::Metal,
    Backend,
    common::{Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
};
use crate::encodable_block::batch_topology::BatchTopology;
use crate::engine::{
    language_model::stream::{LanguageModelStream, LanguageModelStreamOptions},
    Engine,
};
use crate::encodable_block::sampling::SamplingMethod;
use crate::safetensors::ParameterLoader;

fn probe_logits<B: Backend + std::fmt::Debug>(
    model: &crate::engine::language_model::LanguageModel<B>,
    model_path: &std::path::Path,
    prompt: &str,
) {
    let context = model.context();
    let tokenizer = model.tokenizer().clone();
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let input: Vec<u64> = encoding.get_ids().iter().map(|&t| t as u64).collect();
    let max_context = model.recommended_context_length();
    let mut state = model.decoder().create_empty_state(max_context, context).unwrap();
    let mut command_buffer = context.create_command_buffer(Some("lfm3_probe"), None).unwrap();
    let input_u32: Vec<u32> = input.iter().map(|&x| x as u32).collect();
    let token_ids = command_buffer.allocate_constant_from_slice(&input_u32).unwrap();
    let batch_dim = BatchTopology::flat(input.len() as u32, true);
    let layer_indices: Vec<u32> = (0..24).collect();
    let output = model
        .decoder()
        .encode(
            &token_ids,
            &batch_dim,
            Some((input.len() as u32 - 1..input.len() as u32).into()),
            Some(&layer_indices),
            &mut state,
            &mut command_buffer,
        )
        .unwrap();
    let offset = (input.len() - 1) * 2048;
    let loader_dump;
    {
        let loader = block_on(ParameterLoader::<B>::new(std::path::Path::new(&model_path).join("model.safetensors").as_path(), context)).unwrap();
        let tree = loader.tree();
        let emb = tree
            .subtree("decoder")
            .subtree("embedding")
            .subtree("embedding");
        let spec = emb
            .metadata::<crate::config::weight_matrix::AnyWeightMatrixSpec>("spec")
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("ERR {e}"));
        let weights = block_on(emb.read_buffer("weights", &[128000, 2048], crate::data_type::DataType::BF16)).unwrap();
        let weights_v: Vec<f32> = crate::tests::helpers::buffer_to_vec::<B, half::bf16>(context, &weights)
            .into_iter()
            .map(f32::from)
            .collect();
        let row = 124901;
        loader_dump = Some(serde_json::json!({
            "spec": spec,
            "weights_row_first8": &weights_v[row * 2048..row * 2048 + 8],
        }));
    }

    let mut bufs: Vec<(&str, B::GlobalGpuBuffer)> = Vec::new();
    {
        let embedding = model.decoder().embedding();
        let emb_out = embedding
            .encode_lookup(&token_ids, input.len() as u32, &mut command_buffer)
            .unwrap();
        let mut buf = context.create_buffer_gpu(emb_out.size()).unwrap();
        command_buffer.encode_copy(&emb_out, &mut buf);
        bufs.push(("embed", buf));
    }
    if let Some(features) = &output.hidden_features {
        for f in features.iter() {
            let mut buf = context.create_buffer_gpu(f.size()).unwrap();
            command_buffer.encode_copy(f, &mut buf);
            bufs.push(("layer", buf));
        }
    }

    let logits_buf = if let Some(logits) = output.logits.as_ref() {
        let mut buf = context.create_buffer_gpu(logits.size()).unwrap();
        command_buffer.encode_copy(logits, &mut buf);
        Some(buf)
    } else {
        None
    };
    drop(token_ids);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    fn stats(values: &[f32]) -> serde_json::Value {
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let rms = (values.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
        serde_json::json!({
            "mean": (mean * 1000.0).round() / 1000.0,
            "rms": (rms * 1000.0).round() / 1000.0,
            "first4": values[..4].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        })
    }

    let mut layer_stats = Vec::new();
    let mut embed_stats = None;
    for (name, buf) in &bufs {
        let v: Vec<f32> = crate::tests::helpers::buffer_to_vec::<B, half::bf16>(context, buf)
            .into_iter()
            .map(f32::from)
            .collect();
        let s = stats(&v[offset..offset + 2048]);
        if *name == "embed" {
            embed_stats = Some(s);
        } else {
            layer_stats.push(s);
        }
    }

    let values: Vec<f32> = logits_buf
        .map(|buf| {
            crate::tests::helpers::buffer_to_vec::<B, half::bf16>(context, &buf)
                .into_iter()
                .map(f32::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let nan_count = values.iter().filter(|v| v.is_nan()).count();
    let mut pairs: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top5: Vec<(usize, f32)> = pairs.into_iter().take(5).collect();
    println!(
        "{}",
        serde_json::json!({
            "probe": "lfm3",
            "prompt_tokens": input.len(),
            "loader": loader_dump,
            "input_ids": input,
            "embed": embed_stats,
            "layer_stats": layer_stats,
            "nan_count": nan_count,
            "top5": top5.iter().map(|(t, s)| (*t, (s * 1000.0).round() / 1000.0)).collect::<Vec<_>>(),
        })
    );
}

fn eval_model<B: Backend + std::fmt::Debug>(
    model: &crate::engine::language_model::LanguageModel<B>,
    prompt: &str,
    max_tokens: usize,
    runs: usize,
    stop_token_ids: &[u64],
) -> Vec<serde_json::Value> {
    let tokenizer = model.tokenizer().clone();
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let input: Vec<u64> = encoding.get_ids().iter().map(|&t| t as u64).collect();
    let max_context = model.recommended_context_length();
    println!("{{\"prompt_tokens\": {}, \"backend\": \"{}\"}}", input.len(), std::any::type_name::<B>());

    let mut responses = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut state = model.create_empty_state(max_context, 0).unwrap();
        let options = LanguageModelStreamOptions {
            sampling_method: SamplingMethod::Greedy,
            timestamps: None,
            grammar: None,
        };
        let started = Instant::now();
        let mut stream = LanguageModelStream::new(model, &input, &mut state, options).unwrap();
        let mut generated: Vec<u64> = Vec::with_capacity(max_tokens);
        let mut first_token: Option<Instant> = None;
        loop {
            let token = block_on(stream.generate()).unwrap();
            let Some(token) = token else { break };
            first_token.get_or_insert(Instant::now());
            generated.push(token);
            if stop_token_ids.contains(&token) || generated.len() >= max_tokens {
                break;
            }
        }
        let finished = Instant::now();
        block_on(stream.close()).unwrap();
        let metrics = stream.metrics();
        let decode_tps = match first_token {
            Some(first) if generated.len() > 1 => {
                (generated.len() - 1) as f64 / (finished - first).as_secs_f64()
            },
            _ => 0.0,
        };
        let generated_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&generated_u32, false).unwrap();
        let record = serde_json::json!({
            "token_ids": generated,
            "text": text,
            "ttft_s": first_token.map(|t| t.duration_since(started).as_secs_f64()),
            "decode_tps": decode_tps,
            "prefill_passes": metrics.num_prefill_forward_passes,
            "decode_passes": metrics.num_decode_forward_passes,
        });
        println!("{record}");
        responses.push(record);
    }
    responses
}

#[uzu_test]
fn lfm3_eval() {
    let Some(model_path) = std::env::var_os("LFM3_MODEL") else {
        println!("{{\"skipped\": \"LFM3_MODEL not set\"}}");
        return;
    };

    let raw_prompt = match std::env::var("LFM3_PROMPT") {
        Ok(text) if !text.is_empty() => text,
        _ => panic!("LFM3_PROMPT is required"),
    };
    let prompt = match raw_prompt.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).unwrap(),
        None => raw_prompt,
    };
    let max_tokens: usize = std::env::var("LFM3_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let runs: usize = std::env::var("LFM3_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let use_cpu = std::env::var("LFM3_BACKEND").as_deref() == Ok("Cpu");
    let probe = std::env::var("LFM3_PROBE").is_ok();

    let responses = if use_cpu {
        let engine = block_on(Engine::<Cpu>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        if probe {
            probe_logits(&model, std::path::Path::new(&model_path), &prompt);
            return;
        }
        let stop_token_ids: Vec<u64> = model.generation_config().stop_token_ids.to_vec();
        eval_model(&model, &prompt, max_tokens, runs, &stop_token_ids)
    } else {
        let engine = block_on(Engine::<Metal>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        if probe {
            probe_logits(&model, std::path::Path::new(&model_path), &prompt);
            return;
        }
        let stop_token_ids: Vec<u64> = model.generation_config().stop_token_ids.to_vec();
        eval_model(&model, &prompt, max_tokens, runs, &stop_token_ids)
    };
    println!("{}", serde_json::to_string(&responses).unwrap());
}
