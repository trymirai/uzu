//! Env-driven greedy eval for the MiniCPM JANGH2 port: drives the engine exactly
//! like the MLX reference (raw prompt text, temperature 0, fixed token budget)
//! and prints a JSON record with token ids, text, timings and forward-pass counts.
//!
//! Env: MINICPM_MODEL (bundle path), MINICPM_PROMPT (@file or text),
//! MINICPM_MAX_TOKENS (default 128), MINICPM_RUNS (default 3),
//! MINICPM_BACKEND (Metal, default; or Cpu).

use std::time::Instant;

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        cpu::Cpu,
        metal::Metal,
        Backend,
        common::{Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context},
    },
    encodable_block::{batch_topology::BatchTopology, sampling::SamplingMethod},
    engine::{
        language_model::stream::{LanguageModelStream, LanguageModelStreamOptions},
        Engine,
    },
};

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

    if std::env::var("MINICPM_PROBE").is_ok() {
        let context = model.context();
        let mut state = model.decoder().create_empty_state(max_context, context).unwrap();
        let mut command_buffer = context.create_command_buffer(Some("probe"), None).unwrap();
        let input_u32: Vec<u32> = input.iter().map(|&x| x as u32).collect();
        let token_ids = command_buffer.allocate_constant_from_slice(&input_u32).unwrap();
        let batch_dim = BatchTopology::flat(input.len() as u32, true);
        let layer_indices: Vec<u32> = (0..40).collect();
        let output = model.decoder().encode(
            &token_ids,
            &batch_dim,
            Some((input.len() as u32 - 1..input.len() as u32).into()),
            Some(&layer_indices),
            &mut state,
            &mut command_buffer,
        ).unwrap();
        let mut logits_buf = None;
        if let Some(logits) = &output.logits {
            let mut buf = context.create_buffer_gpu(logits.size()).unwrap();
            command_buffer.encode_copy(logits, &mut buf);
            logits_buf = Some(buf);
        }
        let mut layer_bufs = Vec::new();
        if let Some(features) = &output.hidden_features {
            for f in features.iter() {
                let mut buf = context.create_buffer_gpu(f.size()).unwrap();
                command_buffer.encode_copy(f, &mut buf);
                layer_bufs.push(buf);
            }
        }
        drop(token_ids);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        for (i, buf) in layer_bufs.iter().enumerate() {
            let v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, buf);
            let v: Vec<f32> = v_bf16.into_iter().map(f32::from).collect();
            let offset = (input.len() - 1) * 2048;
            let last_tok = &v[offset..offset + 2048];
            let mean = last_tok.iter().sum::<f32>() / 2048.0;
            let rms = (last_tok.iter().map(|x| x * x).sum::<f32>() / 2048.0).sqrt();
            let first4 = format!("{:+.4}, {:+.4}, {:+.4}, {:+.4}", last_tok[0], last_tok[1], last_tok[2], last_tok[3]);
            println!("UZU_LAYER {:2}: mean={:+.5} rms={:.5} first4=[{}]", i, mean, rms, first4);
        }
        if let Some(buf) = logits_buf {
            let v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, &buf);
            let v: Vec<f32> = v_bf16.into_iter().map(f32::from).collect();
            let mut pairs: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
            let n_nans = v.iter().filter(|x| x.is_nan()).count();
            let n_infs = v.iter().filter(|x| x.is_infinite()).count();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("PROBE LOGITS: len={}, nans={}, infs={}", v.len(), n_nans, n_infs);
            println!("PROBE TOP8: {}", serde_json::to_string(&pairs[..8.min(pairs.len())]).unwrap());
            if let Some(pos) = pairs.iter().position(|&(idx, _)| idx == 21502) {
                println!("token 21502: rank={}, logit={}", pos, pairs[pos].1);
            }
            if let Some(pos) = pairs.iter().position(|&(idx, _)| idx == 9) {
                println!("token 9: rank={}, logit={}", pos, pairs[pos].1);
            }
            if let Some(pos) = pairs.iter().position(|&(idx, _)| idx == 332) {
                println!("token 332: rank={}, logit={}", pos, pairs[pos].1);
            }
        }
    }

    let mut responses = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut state = model.create_empty_state(max_context, 0).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let enable_profile = std::env::var("MINICPM_PROFILE").is_ok();
        let options = LanguageModelStreamOptions {
            sampling_method: SamplingMethod::Greedy,
            timestamps: if enable_profile { Some(tx) } else { None },
            grammar: None,
        };
        let started = Instant::now();
        let mut stream = LanguageModelStream::new(model, &input, &mut state, options).unwrap();
        let mut generated: Vec<u64> = Vec::with_capacity(max_tokens);
        let mut first_token: Option<Instant> = None;
        let mut span_batches: Vec<(bool, Box<[crate::backends::common::TimestampSpan]>)> = Vec::new();
        loop {
            let is_prefill = generated.is_empty();
            let token = block_on(stream.generate()).unwrap();
            let Some(token) = token else { break };
            if enable_profile {
                while let Ok(spans) = rx.try_recv() {
                    span_batches.push((is_prefill, spans));
                }
            }
            first_token.get_or_insert(Instant::now());
            generated.push(token);
            if stop_token_ids.contains(&token) || generated.len() >= max_tokens {
                break;
            }
        }
        let finished = Instant::now();
        block_on(stream.close()).unwrap();
        if enable_profile {
            while let Ok(spans) = rx.try_recv() {
                span_batches.push((false, spans));
            }
            let mut prefill_times: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            let mut decode_times: std::collections::HashMap<String, (f64, usize)> = std::collections::HashMap::new();
            let decode_steps = span_batches.iter().filter(|(p, _)| !*p).count().max(1);

            for (is_prefill, spans) in &span_batches {
                for (i, span) in spans.iter().enumerate() {
                    let has_child = spans.iter().enumerate().any(|(j, other)| j != i && other.name.starts_with(&span.name) && other.name.len() > span.name.len());
                    if !has_child {
                        let dur = span.end.duration_since(span.start).as_secs_f64();
                        let base_name = span.name.rsplit('/').next().unwrap_or(&span.name).to_string();
                        let tagged_name = if base_name == "Gemv" {
                            if span.name.contains("embedding readout") {
                                "Gemv_LMHead".to_string()
                            } else if span.name.contains("mlp (dense)") {
                                "Gemv_SharedExpert".to_string()
                            } else if span.name.contains("delta net") {
                                "Gemv_DeltaNet".to_string()
                            } else if span.name.contains("attention") {
                                "Gemv_Attention".to_string()
                            } else if span.name.contains("mlp (moe)") {
                                "Gemv_Router_SharedGate".to_string()
                            } else {
                                format!("Gemv_Other({})", span.name)
                            }
                        } else if base_name == "Gemm" {
                            format!("Gemm({})", span.name)
                        } else {
                            base_name
                        };

                        if *is_prefill {
                            *prefill_times.entry(tagged_name).or_insert(0.0) += dur;
                        } else {
                            let entry = decode_times.entry(tagged_name).or_insert((0.0, 0));
                            entry.0 += dur;
                            entry.1 += 1;
                        }
                    }
                }
            }

            let mut sorted_decode: Vec<(String, (f64, usize))> = decode_times.into_iter().collect();
            sorted_decode.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total_decode: f64 = sorted_decode.iter().map(|x| x.1.0).sum();
            println!("\n=== DECODE GPU PROFILE ({} steps, Total: {:.2} ms, {:.2} ms/tok, ~{:.1} tok/s) ===",
                decode_steps, total_decode * 1000.0, (total_decode / decode_steps as f64) * 1000.0, decode_steps as f64 / total_decode);
            for (name, (dur, count)) in sorted_decode.iter().take(20) {
                let ms_per_tok = (dur / decode_steps as f64) * 1000.0;
                let calls_per_tok = *count as f64 / decode_steps as f64;
                println!("{:25} : {:6.2} ms/tok ({:4.1}%) | calls/tok: {:4.1} | total: {:6.2} ms",
                    name, ms_per_tok, (dur / total_decode) * 100.0, calls_per_tok, dur * 1000.0);
            }
        }
        let metrics = stream.metrics();
        let decode_tps = match first_token {
            Some(first) if generated.len() > 1 => (generated.len() - 1) as f64 / (finished - first).as_secs_f64(),
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

fn eval_multimodal<B: Backend + std::fmt::Debug>(
    model: &crate::engine::language_model::LanguageModel<B>,
    manifest_path: &str,
    max_tokens: usize,
    stop_token_ids: &[u64],
) -> serde_json::Value {
    let manifest_str = std::fs::read_to_string(manifest_path).expect("Failed to read manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_str).expect("Failed to parse manifest");

    let prompt_tokens: Vec<u64> = manifest["prompt_tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let visual_token_indices: Vec<usize> = manifest["visual_token_indices"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let delta = manifest["delta"].as_i64().unwrap();
    let mrope_raw = manifest["mrope_positions"].as_array().unwrap();
    let mrope_t: Vec<u32> = mrope_raw[0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mrope_h: Vec<u32> = mrope_raw[1].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mrope_w: Vec<u32> = mrope_raw[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let mrope_3d: Vec<[u32; 3]> = (0..prompt_tokens.len())
        .map(|i| [mrope_t[i], mrope_h[i], mrope_w[i]])
        .collect();

    // Group visual_token_indices into contiguous runs (start, count)
    let mut visual_slices: Vec<(usize, usize)> = Vec::new();
    if !visual_token_indices.is_empty() {
        let mut start = visual_token_indices[0];
        let mut count = 1;
        for &idx in &visual_token_indices[1..] {
            if idx == start + count {
                count += 1;
            } else {
                visual_slices.push((start, count));
                start = idx;
                count = 1;
            }
        }
        visual_slices.push((start, count));
    }

    // Read features binary
    let features_path = manifest["features_path"].as_str().unwrap();
    let features_raw = std::fs::read(features_path).expect("Failed to read features binary");
    let features_bf16: Vec<half::bf16> = features_raw
        .as_chunks::<2>().0.iter()
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    let context = model.context();
    let max_context = model.recommended_context_length();
    let mut state = model.decoder().create_empty_state(max_context, context).unwrap();
    state.prepare(0, prompt_tokens.len() as u32).unwrap();

    let started = Instant::now();
    let mut command_buffer = context.create_command_buffer(Some("vlm_prefill"), None).unwrap();
    let input_u32: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
    let token_ids = command_buffer.allocate_constant_from_slice(&input_u32).unwrap();
    let features_gpu = command_buffer.allocate_constant_from_slice(&features_bf16).unwrap();

    let batch_dim = BatchTopology::flat(prompt_tokens.len() as u32, true);
    let output = model.decoder().encode_multimodal(
        &token_ids,
        Some(&features_gpu),
        Some(&visual_slices),
        Some(&mrope_3d),
        delta,
        &batch_dim,
        Some((prompt_tokens.len() as u32 - 1..prompt_tokens.len() as u32).into()),
        None,
        &mut state,
        &mut command_buffer,
    ).unwrap();

    let mut logits_buf = context.create_buffer_gpu(output.logits.as_ref().unwrap().size()).unwrap();
    command_buffer.encode_copy(output.logits.as_ref().unwrap(), &mut logits_buf);
    state.encode_accept(&(0..prompt_tokens.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer).unwrap();
    drop(token_ids);
    drop(features_gpu);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let ttft = started.elapsed().as_secs_f64();

    // Read first token from logits
    let v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, &logits_buf);
    let v: Vec<f32> = v_bf16.into_iter().map(f32::from).collect();
    let first_token = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;

    let mut generated: Vec<u64> = vec![first_token];
    let mut curr_token = first_token;
    let decode_start = Instant::now();

    while generated.len() < max_tokens && !stop_token_ids.contains(&curr_token) {
        state.prepare(state.context_length, 1).unwrap();
        let mut command_buffer = context.create_command_buffer(Some("vlm_decode"), None).unwrap();
        let next_u32 = [curr_token as u32];
        let token_ids = command_buffer.allocate_constant_from_slice(&next_u32).unwrap();
        let batch_dim = BatchTopology::flat(1, true);

        let output = model.decoder().encode(
            &token_ids,
            &batch_dim,
            Some((0..1).into()),
            None,
            &mut state,
            &mut command_buffer,
        ).unwrap();

        let mut next_logits_buf = context.create_buffer_gpu(output.logits.as_ref().unwrap().size()).unwrap();
        command_buffer.encode_copy(output.logits.as_ref().unwrap(), &mut next_logits_buf);
        state.encode_accept(&[0], &mut command_buffer).unwrap();
        drop(token_ids);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

        let next_v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, &next_logits_buf);
        let next_v: Vec<f32> = next_v_bf16.into_iter().map(f32::from).collect();
        curr_token = next_v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;
        generated.push(curr_token);
    }

    let decode_dur = decode_start.elapsed().as_secs_f64();
    let decode_tokens_count = generated.len().saturating_sub(1);
    let decode_tps = if decode_dur > 0.0 { decode_tokens_count as f64 / decode_dur } else { 0.0 };

    let tokenizer = model.tokenizer();
    let generated_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
    let text = tokenizer.decode(&generated_u32, false).unwrap_or_default();

    println!("VLM RESULT: \"{}\"", text);
    println!("TTFT: {:.1} ms | Decode TPS: {:.2} tok/s | Tokens: {}", ttft * 1000.0, decode_tps, generated.len());

    serde_json::json!({
        "token_ids": generated,
        "text": text,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "total_tokens": generated.len(),
    })
}

fn eval_native_vlm<B: Backend + std::fmt::Debug>(
    model: &crate::engine::language_model::LanguageModel<B>,
    model_dir: &std::path::Path,
    image_path: &str,
    prompt_text: &str,
    max_tokens: usize,
    stop_token_ids: &[u64],
) -> serde_json::Value {
    use crate::vision::{
        preprocess_image, build_placeholder_tokens_with_config, compute_canvas_mrope_with_config,
        VisionConfig, CanvasConfig, IMAGE_PAD,
    };

    let img = image::open(image_path).expect("Failed to open image");
    let vision_config = VisionConfig::default();
    let canvas_config = CanvasConfig::default();

    let t0 = Instant::now();
    let preprocessed = preprocess_image(&img, &vision_config, &canvas_config, 9, 448);
    println!("Native image preprocessing took: {:?}", t0.elapsed());

    let t_tower = Instant::now();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let visual_tokens = {
        let tower = crate::vision::MetalVisionTower::load_from_model_dir(model_dir).expect("Failed to load MetalVisionTower");
        let vt = tower.forward(&preprocessed).expect("MetalVisionTower forward pass failed");
        println!("MetalVisionTower forward took: {:?}", t_tower.elapsed());
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in &vt {
            if v < min { min = v; }
            if v > max { max = v; }
        }
        println!("Metal visual_tokens len: {}, min: {}, max: {}, first 5: {:?}", vt.len(), min, max, &vt[..5]);
        vt
    };
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let visual_tokens = {
        let tower = VisionTower::load_from_model_dir(model_dir).expect("Failed to load VisionTower");
        let vt = tower.forward(&preprocessed).expect("VisionTower forward pass failed");
        println!("VisionTower forward took: {:?}", t_tower.elapsed());
        vt
    };

    let features_bf16: Vec<half::bf16> = visual_tokens.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let tokenizer = model.tokenizer();
    let placeholders = build_placeholder_tokens_with_config(
        &preprocessed.grids,
        preprocessed.slice_grid,
        0,
        true,
        16,
        &CanvasConfig::default(),
    );
    let t_prefix: Vec<u64> = tokenizer.encode("<|im_start|>user\n", false).unwrap().get_ids().iter().map(|&t| t as u64).collect();
    let t_placeholders: Vec<u64> = placeholders.into_iter().map(|t| t as u64).collect();
    let t_sep: Vec<u64> = vec![198];
    let t_query: Vec<u64> = tokenizer.encode(prompt_text, false).unwrap().get_ids().iter().map(|&t| t as u64).collect();
    let enable_thinking = std::env::var("MINICPM_THINKING").map(|v| v == "1" || v == "true").unwrap_or(false);
    let suffix_str = if enable_thinking {
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    };
    let t_suffix: Vec<u64> = tokenizer.encode(suffix_str, false).unwrap().get_ids().iter().map(|&t| t as u64).collect();

    let prompt_tokens: Vec<u64> = [t_prefix, t_placeholders, t_sep, t_query, t_suffix].concat();
    println!("Total prompt tokens: {}", prompt_tokens.len());

    let visual_token_indices: Vec<usize> = prompt_tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == IMAGE_PAD as u64)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(visual_token_indices.len(), visual_tokens.len() / 2048);

    let (mrope_3d, delta) = compute_canvas_mrope_with_config(&prompt_tokens, &preprocessed.grids, 4, &CanvasConfig::default());

    let mut visual_slices: Vec<(usize, usize)> = Vec::new();
    if !visual_token_indices.is_empty() {
        let mut start = visual_token_indices[0];
        let mut count = 1;
        for &idx in &visual_token_indices[1..] {
            if idx == start + count {
                count += 1;
            } else {
                visual_slices.push((start, count));
                start = idx;
                count = 1;
            }
        }
        visual_slices.push((start, count));
    }

    let context = model.context();
    let max_context = model.recommended_context_length();
    let mut state = model.decoder().create_empty_state(max_context, context).unwrap();
    state.prepare(0, prompt_tokens.len() as u32).unwrap();

    let started = Instant::now();
    let mut command_buffer = context.create_command_buffer(Some("native_vlm_prefill"), None).unwrap();
    let input_u32: Vec<u32> = prompt_tokens.iter().map(|&x| x as u32).collect();
    let token_ids = command_buffer.allocate_constant_from_slice(&input_u32).unwrap();
    let features_gpu = command_buffer.allocate_constant_from_slice(&features_bf16).unwrap();

    let batch_dim = BatchTopology::flat(prompt_tokens.len() as u32, true);
    let output = model.decoder().encode_multimodal(
        &token_ids,
        Some(&features_gpu),
        Some(&visual_slices),
        Some(&mrope_3d),
        delta,
        &batch_dim,
        Some((prompt_tokens.len() as u32 - 1..prompt_tokens.len() as u32).into()),
        None,
        &mut state,
        &mut command_buffer,
    ).unwrap();

    let mut logits_buf = context.create_buffer_gpu(output.logits.as_ref().unwrap().size()).unwrap();
    command_buffer.encode_copy(output.logits.as_ref().unwrap(), &mut logits_buf);
    state.encode_accept(&(0..prompt_tokens.len() as u32).collect::<Box<[u32]>>(), &mut command_buffer).unwrap();
    drop(token_ids);
    drop(features_gpu);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let ttft = started.elapsed().as_secs_f64();

    // Read first token from logits
    let v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, &logits_buf);
    let v: Vec<f32> = v_bf16.into_iter().map(f32::from).collect();
    let first_token = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;

    let mut generated: Vec<u64> = vec![first_token];
    let mut curr_token = first_token;
    let decode_start = Instant::now();

    while generated.len() < max_tokens && !stop_token_ids.contains(&curr_token) {
        state.prepare(state.context_length, 1).unwrap();
        let mut command_buffer = context.create_command_buffer(Some("native_vlm_decode"), None).unwrap();
        let next_u32 = [curr_token as u32];
        let token_ids = command_buffer.allocate_constant_from_slice(&next_u32).unwrap();
        let batch_dim = BatchTopology::flat(1, true);

        let output = model.decoder().encode(
            &token_ids,
            &batch_dim,
            Some((0..1).into()),
            None,
            &mut state,
            &mut command_buffer,
        ).unwrap();

        let mut next_logits_buf = context.create_buffer_gpu(output.logits.as_ref().unwrap().size()).unwrap();
        command_buffer.encode_copy(output.logits.as_ref().unwrap(), &mut next_logits_buf);
        state.encode_accept(&[0], &mut command_buffer).unwrap();
        drop(token_ids);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

        let next_v_bf16: Vec<half::bf16> = crate::tests::helpers::buffer_to_vec::<B, _>(context, &next_logits_buf);
        let next_v: Vec<f32> = next_v_bf16.into_iter().map(f32::from).collect();
        curr_token = next_v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u64;
        generated.push(curr_token);
    }

    let decode_dur = decode_start.elapsed().as_secs_f64();
    let decode_tokens_count = generated.len().saturating_sub(1);
    let decode_tps = if decode_dur > 0.0 { decode_tokens_count as f64 / decode_dur } else { 0.0 };

    let generated_u32: Vec<u32> = generated.iter().map(|&x| x as u32).collect();
    let text = tokenizer.decode(&generated_u32, false).unwrap_or_default();

    println!("NATIVE VLM RESULT: \"{}\"", text);
    println!("TTFT: {:.1} ms | Decode TPS: {:.2} tok/s | Tokens: {}", ttft * 1000.0, decode_tps, generated.len());

    serde_json::json!({
        "token_ids": generated,
        "text": text,
        "ttft_s": ttft,
        "decode_tps": decode_tps,
        "total_tokens": generated.len(),
    })
}

#[uzu_test]
fn minicpm_eval() {
    let Some(model_path) = std::env::var_os("MINICPM_MODEL") else {
        println!("{{\"skipped\": \"MINICPM_MODEL not set\"}}");
        return;
    };

    if let Ok(image_path) = std::env::var("MINICPM_IMAGE") {
        let max_tokens: usize = std::env::var("MINICPM_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let prompt_text = std::env::var("MINICPM_PROMPT").unwrap_or_else(|_| "Describe this image in detail.".to_string());
        let engine = block_on(Engine::<Metal>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        let stop_token_ids: Vec<u64> = model.generation_config().stop_token_ids.to_vec();
        let res = eval_native_vlm(&model, std::path::Path::new(&model_path), &image_path, &prompt_text, max_tokens, &stop_token_ids);
        println!("{}", serde_json::to_string(&res).unwrap());
        return;
    }

    if let Ok(manifest_path) = std::env::var("MINICPM_VLM_MANIFEST") {
        let max_tokens: usize = std::env::var("MINICPM_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let engine = block_on(Engine::<Metal>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        let stop_token_ids: Vec<u64> = model.generation_config().stop_token_ids.to_vec();
        let res = eval_multimodal(&model, &manifest_path, max_tokens, &stop_token_ids);
        println!("{}", serde_json::to_string(&res).unwrap());
        return;
    }
    let raw_prompt = match std::env::var("MINICPM_PROMPT") {
        Ok(text) if !text.is_empty() => text,
        _ => panic!("MINICPM_PROMPT is required"),
    };
    let prompt = match raw_prompt.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).unwrap(),
        None => raw_prompt,
    };
    let max_tokens: usize = std::env::var("MINICPM_MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let runs: usize = std::env::var("MINICPM_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let cpu = std::env::var("MINICPM_BACKEND").as_deref() == Ok("Cpu");

    let responses = if cpu {
        let stop_token_ids: Vec<u64> = vec![248046, 248044];
        let engine = block_on(Engine::<Cpu>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        eval_model(&model, &prompt, max_tokens, runs.min(1), &stop_token_ids)
    } else {
        let engine = block_on(Engine::<Metal>::new()).unwrap();
        let model = block_on(engine.load_language_model(std::path::Path::new(&model_path))).unwrap();
        let stop_token_ids: Vec<u64> = model.generation_config().stop_token_ids.to_vec();
        eval_model(&model, &prompt, max_tokens, runs, &stop_token_ids)
    };
    println!("{}", serde_json::to_string(&responses).unwrap());
}
