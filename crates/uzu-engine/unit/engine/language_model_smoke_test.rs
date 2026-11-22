use std::{
    collections::HashMap,
    path::PathBuf,
    sync::mpsc::{Receiver, channel},
    time::{Duration, Instant},
};

use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::TimestampSpan,
    encodable_block::sampling::SamplingMethod,
    engine::Engine,
    tests::helpers::for_each_non_cpu_backend,
};

/// Sums GPU time by span name across layers, and prints the largest spans per decoded token.
fn print_profile(
    spans: &Receiver<Box<[TimestampSpan]>>,
    tokens: usize,
) {
    let mut totals = HashMap::<String, (Duration, usize)>::new();
    for batch in spans.try_iter() {
        for span in batch.iter() {
            let name = span
                .name
                .split('/')
                .map(|part| if part.starts_with("transformer layer ") { "layer" } else { part })
                .collect::<Vec<_>>()
                .join("/");
            let total = totals.entry(name).or_default();
            total.0 += span.end.saturating_duration_since(span.start);
            total.1 += 1;
        }
    }
    let mut totals = totals.into_iter().collect::<Vec<_>>();
    totals.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (name, (total, count)) in totals.iter().take(40) {
        eprintln!(
            "{:>9.1} us/token {:>6.1} calls/token  {name}",
            total.as_secs_f64() * 1e6 / tokens as f64,
            *count as f64 / tokens as f64
        );
    }
}

/// Generates from a converted model on the GPU with the model's own sampling: `UZU_SMOKE_MODEL` is its directory,
/// `UZU_SMOKE_PROMPT` the raw prompt with its special tokens, `UZU_SMOKE_TOKENS` the most tokens to generate and
/// `UZU_SMOKE_SEED` the sampling seed. `UZU_SMOKE_GREEDY` samples greedily instead, and with `UZU_SMOKE_PROFILE` the
/// test prints the GPU time of each debug group per decoded token. Without a model directory it passes without
/// running.
#[uzu_test]
fn smoke_generate_from_local_model() {
    let Ok(model_path) = std::env::var("UZU_SMOKE_MODEL") else {
        return;
    };
    let prompt = std::env::var("UZU_SMOKE_PROMPT").expect("UZU_SMOKE_PROMPT holds the raw prompt");
    let max_tokens = std::env::var("UZU_SMOKE_TOKENS").map_or(64, |tokens| tokens.parse::<usize>().unwrap());
    for_each_non_cpu_backend!(|B| {
        block_on(async {
            let engine = Engine::<B>::new().await.expect("create engine");
            let load_start = Instant::now();
            let model = engine.load_language_model(&PathBuf::from(&model_path)).await.expect("load model");
            let load_time = load_start.elapsed();
            let input = model
                .tokenizer()
                .encode(prompt.as_str(), false)
                .expect("tokenize prompt")
                .get_ids()
                .iter()
                .map(|&id| u64::from(id))
                .collect::<Vec<u64>>();
            let stop_token_ids = model.generation_config().stop_token_ids.clone();
            let seed = std::env::var("UZU_SMOKE_SEED").map_or(0, |seed| seed.parse::<u64>().unwrap());
            let mut state = model.create_empty_state(Some(8192), seed).expect("create state");
            let mut options = model.default_stream_options();
            if std::env::var("UZU_SMOKE_GREEDY").is_ok() {
                options.sampling_method = SamplingMethod::Greedy;
            }
            let profile = std::env::var("UZU_SMOKE_PROFILE").is_ok().then(channel::<Box<[TimestampSpan]>>);
            let (timestamps, spans) = profile.unzip();
            options.timestamps = timestamps;
            let start = Instant::now();
            let mut stream = model.stream(&input, &mut state, options).expect("start stream");
            let mut output = Vec::new();
            let mut first_token_time = None;
            // UZU_SMOKE_IGNORE_STOP keeps generating past the stop tokens, to time long contexts.
            let ignore_stop = std::env::var("UZU_SMOKE_IGNORE_STOP").is_ok();
            let mut segment_start = None;
            #[allow(unused_mut, unused_assignments)]
            let mut decode_counters = (0u64, 0u64);
            while output.len() < max_tokens {
                let Some(token) = stream.generate().await.expect("generate") else {
                    break;
                };
                if first_token_time.is_none() {
                    first_token_time = Some(start.elapsed());
                    // Leave the prefill out of the profile.
                    spans.iter().for_each(|spans| spans.try_iter().for_each(drop));
                    #[cfg(backend = "metal")]
                    {
                        decode_counters = crate::backends::metal::dispatch_counters();
                    }
                }
                output.push(token);
                if output.len() % 250 == 0 {
                    let now = Instant::now();
                    if let Some(segment_start) = segment_start {
                        let seconds = now.duration_since(segment_start).as_secs_f64();
                        eprintln!("tokens {}..{}: {:.1} tok/s", output.len() - 250, output.len(), 250.0 / seconds);
                    }
                    segment_start = Some(now);
                }
                if stop_token_ids.contains(&token) && !ignore_stop {
                    break;
                }
            }
            let total_time = start.elapsed();
            let metrics = stream.metrics().clone();
            stream.close().await.expect("close stream");
            if metrics.num_decode_forward_passes > 0 {
                eprintln!(
                    "speculation: {} decode passes, {:.2} tokens per pass, {} of {} proposed tokens accepted",
                    metrics.num_decode_forward_passes,
                    metrics.num_tokens_returned as f64 / metrics.num_decode_forward_passes as f64,
                    metrics.num_tokens_accepted,
                    metrics.num_tokens_proposed,
                );
            }
            let first_token_time = first_token_time.unwrap_or(total_time);
            let decode_tokens = output.len().saturating_sub(1);
            let ids = output.iter().map(|&token| token as u32).collect::<Vec<u32>>();
            let text = model.tokenizer().decode(&ids, false).expect("detokenize");
            eprintln!(
                "load {load_time:.2?} | prompt {} tokens, first token {first_token_time:.2?} ({:.0} tok/s) | decode {} \
                 tokens at {:.1} tok/s",
                input.len(),
                input.len() as f64 / first_token_time.as_secs_f64(),
                decode_tokens,
                decode_tokens as f64 / (total_time - first_token_time).as_secs_f64(),
            );
            if let Some(spans) = &spans {
                print_profile(spans, decode_tokens.max(1));
            }
            #[cfg(backend = "metal")]
            {
                let (dispatches, barriers) = crate::backends::metal::dispatch_counters();
                let tokens = decode_tokens.max(1) as f64;
                eprintln!(
                    "decode: {:.0} dispatches and {:.0} barriers per token",
                    (dispatches - decode_counters.0) as f64 / tokens,
                    (barriers - decode_counters.1) as f64 / tokens,
                );
            }
            eprintln!("ids: {output:?}");
            eprintln!("text: {text}");
        });
    });
}
