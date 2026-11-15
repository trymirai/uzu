use std::{io::Write, path::PathBuf};
use anyhow::{Context, Result};
use image::GenericImageView;
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};

pub async fn run_vlm_cli(
    model_name: String,
    image_paths: Vec<String>,
    video_path: Option<String>,
    prompt: Option<String>,
    interactive: bool,
    max_tokens: usize,
    max_context_tokens: usize,
    temperature: f32,
) -> Result<()> {
    eprintln!("🚀 Initializing VLM CLI for model: {model_name}");
    let mut images = Vec::new();
    for (idx, path) in image_paths.iter().enumerate() {
        eprintln!("📷 Loading image [{idx}]: {path}");
        let img = image::open(path)
            .with_context(|| format!("Failed to open image file: {path}"))?;
        let (iw, ih) = img.dimensions();
        eprintln!("📐 Image [{idx}] dimensions: {iw}x{ih}");
        images.push(img);
    }

    let video_frames: Option<Vec<image::DynamicImage>> = match &video_path {
        Some(path) => {
            eprintln!("🎞️  Loading video frames: {path}");
            let frames = crate::common::video::decode_video_source(path)
                .with_context(|| format!("Failed to decode video: {path}"))?;
            eprintln!("🎞️  Video frames sampled: {}", frames.len());
            Some(frames)
        },
        None => None,
    };

    let is_interactive = interactive || prompt.is_none();

    let engine = Engine::new(EngineConfig::default()).await?;
    let resolved = crate::common::hf::resolve_model_or_prompt_hf(&engine, &model_name, true).await?;

    let model_path = match &resolved.accessibility {
        uzu::types::model::ModelAccessibility::OnDevice {
            source: uzu::types::model::ModelSource::Filesystem { path },
        } => PathBuf::from(path),
        _ => anyhow::bail!("Model accessibility is not local filesystem"),
    };

    eprintln!("🧠 Initializing language model engine...");
    let session = engine.chat(resolved, ChatConfig::default()).await
        .context("Failed to create chat session")?;
    let backend_inst = session.chat_token_backend_instance().await
        .context("No backend instance available")?;
    let uzu_backend = backend_inst.as_any()
        .downcast_ref::<uzu::engine::bridge::UzuChatTokenBackendInstance<uzu_engine::backends::metal::Metal>>()
        .context("Backend is not Metal")?;
    let lm = uzu_backend.language_model().clone();

    let vlm = uzu_engine::vision::AnyVlmPipeline::load_from_model_dir(&model_path, lm.tokenizer())
        .context("Failed to load VLM pipeline from model directory")?;

    let mut sampler = uzu_engine::vision::VlmSampler::new(uzu_engine::vision::VlmSamplingConfig {
        temperature: if temperature > 0.0 { Some(temperature) } else { None },
        ..Default::default()
    });

    let tokenizer = lm.tokenizer();
    sampler.set_vocab_limit(tokenizer.get_vocab_size(true));
    let mut stop_token_ids: Vec<u64> = lm.generation_config().stop_token_ids.to_vec();
    if let Ok(enc) = tokenizer.encode("<|im_end|>", false)
        && let Some(&id) = enc.get_ids().first() {
            stop_token_ids.push(id as u64);
        }
    if let Ok(enc) = tokenizer.encode("<|endoftext|>", false)
        && let Some(&id) = enc.get_ids().first() {
            stop_token_ids.push(id as u64);
        }

    let initial_query = match prompt {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            print!("\nUser >>> ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            let trimmed = line.trim().to_string();
            if trimmed.is_empty() {
                if images.is_empty() {
                    "Hello!".to_string()
                } else {
                    "Describe this image in detail.".to_string()
                }
            } else {
                trimmed
            }
        }
    };

    if video_frames.is_some() {
        eprintln!("🎞️  Encoding video frames via {} vision tower...", vlm.vision_backend());
    } else if images.is_empty() {
        eprintln!("⚡ Encoding text tokens (text-only prompt)...");
    } else if vlm.vision_backend() == "metal" {
        eprintln!("⚡ Preprocessing {} image(s) and encoding visual tokens via Metal vision tower...", images.len());
    } else {
        eprintln!("⚡ Preprocessing {} image(s) and encoding visual tokens via CPU vision tower...", images.len());
    }
    let t0 = std::time::Instant::now();
    let (prepared_res, (state_res, t_state)) = std::thread::scope(|s| {
        let state_handle = s.spawn(|| {
            let t_s0 = std::time::Instant::now();
            let st = vlm.acquire_state(&lm, None);
            (st, t_s0.elapsed())
        });
        let prep = match &video_frames {
            Some(frames) => vlm.prepare_video_inputs(frames, &initial_query, tokenizer, false),
            None => vlm.prepare_inputs(&images, &initial_query, tokenizer, false),
        };
        let (st, t_s) = state_handle.join().unwrap();
        (prep, (st, t_s))
    });
    let t_prep = t0.elapsed();

    let prepared = prepared_res.context("VLM input preparation failed")?;
    let mut state = state_res.context("Failed to acquire TransformerState")?;

    eprintln!("🚀 Executing prefill ({} tokens)...", prepared.prompt_tokens().len());
    let t2 = std::time::Instant::now();
    let first_token = vlm.prefill(&lm, &mut state, &prepared, &mut sampler)
        .context("VLM prefill failed")?;
    let t_prefill = t2.elapsed();
    let ttft = t0.elapsed();

    eprintln!("💬 Generation start:\n----------------------------------------");
    let mut decoder = uzu_engine::vision::StreamingTokenDecoder::new(tokenizer, false);
    if let Ok(Some(text)) = decoder.step(first_token as u32) {
        print!("{text}");
        std::io::stdout().flush()?;
    }

    let t_decode_start = std::time::Instant::now();
    let mut curr_token = first_token;
    let mut generated_tokens = 1;

    while generated_tokens < max_tokens && !stop_token_ids.contains(&curr_token) {
        let next_tok = vlm.decode_step_with_state(&lm, &mut state, curr_token, &mut sampler)?;
        curr_token = next_tok;
        if stop_token_ids.contains(&curr_token) {
            break;
        }
        if let Ok(Some(text)) = decoder.step(curr_token as u32) {
            print!("{text}");
            std::io::stdout().flush()?;
        }
        generated_tokens += 1;
    }
    if let Ok(Some(tail)) = decoder.finish() {
        print!("{tail}");
        std::io::stdout().flush()?;
    }
    let t_decode = t_decode_start.elapsed();
    println!("\n----------------------------------------");

    let tok_per_sec = if t_decode.as_secs_f64() > 0.0 && generated_tokens > 1 {
        (generated_tokens - 1) as f64 / t_decode.as_secs_f64()
    } else {
        0.0
    };

    eprintln!("📊 [Turn 1 Benchmark Summary]");
    eprintln!("   - prepare_inputs (Vision Tower): {:.3}s ({} images)", t_prep.as_secs_f64(), images.len());
    eprintln!("   - acquire_state (Parallel Init): {:.3}s (overlapped behind vision)", t_state.as_secs_f64());
    eprintln!("   - prefill ({} tokens):           {:.3}s", prepared.prompt_tokens().len(), t_prefill.as_secs_f64());
    eprintln!("   - Time To First Token (TTFT):    {:.3}s", ttft.as_secs_f64());
    eprintln!("   - Decode Speed:                  {:.2} tokens/s ({} tokens in {:.2}s)", tok_per_sec, generated_tokens, t_decode.as_secs_f64());
    eprintln!("   - Current Context Length:        {} tokens", state.context_length());

    // Interactive multi-turn chat loop
    if is_interactive {
        eprintln!("\n💬 Interactive Multi-turn Chat Active (KV Cache & Visual Features Preserved)");
        eprintln!("   Type follow-up question and press Enter. Type '/exit' or 'quit' to end.");
        eprintln!("-------------------------------------------------------------------------");

        let mut turn_idx = 2;
        loop {
            print!("\nUser >>> ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            let query = line.trim();
            if query.is_empty() {
                continue;
            }
            if query == "/exit" || query == "quit" || query == "exit" {
                eprintln!("👋 Exiting VLM session.");
                break;
            }

            if state.context_length() as usize >= max_context_tokens {
                eprintln!("⚠️  [KV Cache Limit] Context length ({}) has reached max limit ({}).", state.context_length(), max_context_tokens);
                eprintln!("   Ending session to avoid memory exhaustion.");
                break;
            }

            eprintln!("⚡ Reusing KV Cache & Visual Features (Vision Tower: 0.000s)...");
            let t_turn_start = std::time::Instant::now();
            let first_token = vlm.prefill_followup_turn(&lm, &mut state, query, tokenizer, &mut sampler, false)
                .context("VLM follow-up prefill failed")?;
            let ttft_turn = t_turn_start.elapsed();

            eprintln!("💬 Generation start:\n----------------------------------------");
            let mut decoder = uzu_engine::vision::StreamingTokenDecoder::new(tokenizer, false);
            if let Ok(Some(text)) = decoder.step(first_token as u32) {
                print!("{text}");
                std::io::stdout().flush()?;
            }

            let t_turn_decode = std::time::Instant::now();
            let mut curr_token = first_token;
            let mut turn_tokens = 1;

            while turn_tokens < max_tokens && !stop_token_ids.contains(&curr_token) {
                let next_tok = vlm.decode_step_with_state(&lm, &mut state, curr_token, &mut sampler)?;
                curr_token = next_tok;
                if stop_token_ids.contains(&curr_token) {
                    break;
                }
                if let Ok(Some(text)) = decoder.step(curr_token as u32) {
                    print!("{text}");
                    std::io::stdout().flush()?;
                }
                turn_tokens += 1;
            }
            if let Ok(Some(tail)) = decoder.finish() {
                print!("{tail}");
                std::io::stdout().flush()?;
            }
            let t_turn_dec = t_turn_decode.elapsed();
            println!("\n----------------------------------------");

            let turn_tok_per_sec = if t_turn_dec.as_secs_f64() > 0.0 && turn_tokens > 1 {
                (turn_tokens - 1) as f64 / t_turn_dec.as_secs_f64()
            } else {
                0.0
            };

            eprintln!("📊 [Turn {turn_idx} Benchmark Summary]");
            eprintln!("   - Vision Tower (Reused):         0.000s");
            eprintln!("   - Time To First Token (TTFT):    {:.3}s (⚡ Instant)", ttft_turn.as_secs_f64());
            eprintln!("   - Decode Speed:                  {:.2} tokens/s ({} tokens in {:.2}s)", turn_tok_per_sec, turn_tokens, t_turn_dec.as_secs_f64());
            eprintln!("   - Total Context Length:          {} tokens", state.context_length());
            turn_idx += 1;
        }
    }

    vlm.release_state(state);
    eprintln!("✅ Clean shutdown and memory released.");

    Ok(())
}
