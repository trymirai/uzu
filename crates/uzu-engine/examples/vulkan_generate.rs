//! Greedy generation on the Vulkan backend from a model directory, without the catalog or a network: time to first
//! token, prefill and decode speed. Usage: vulkan_generate <model directory> <prompt file> [tokens] [runs]

use std::{
    error::Error,
    hash::{DefaultHasher, Hasher},
    path::Path,
    time::Instant,
};

use futures::executor::block_on;
use uzu_engine::{
    backends::vulkan::Vulkan,
    engine::{Engine, language_model::stream::SamplingMethod},
};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let usage = "usage: vulkan_generate <model directory> <prompt file> [tokens] [runs]";
    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or(usage)?;
    let prompt = std::fs::read_to_string(args.next().ok_or(usage)?)?;
    let tokens: usize = args.next().map_or(Ok(128), |value| value.parse())?;
    let runs: usize = args.next().map_or(Ok(3), |value| value.parse())?;

    let engine = block_on(Engine::<Vulkan>::new())?;
    let load = Instant::now();
    let model = block_on(engine.load_language_model(Path::new(&model_path)))?;
    println!("loaded {model_path} in {:.2} s", load.elapsed().as_secs_f64());
    let input = model.tokenizer().encode(prompt, false)?.get_ids().iter().map(|&id| u64::from(id)).collect::<Vec<_>>();

    for run in 0..runs {
        let mut state = model.create_empty_state(model.recommended_context_length(), 7)?;
        let mut options = model.default_stream_options();
        options.sampling_method = SamplingMethod::Greedy;
        let start = Instant::now();
        let mut stream = model.stream(&input, &mut state, options)?;
        let mut first_token = None;
        let mut generated = 0;
        let mut output_digest = DefaultHasher::new();
        while generated < tokens {
            let Some(token) = block_on(stream.generate())? else {
                break;
            };
            output_digest.write_u64(token);
            generated += 1;
            first_token.get_or_insert_with(|| start.elapsed().as_secs_f64());
        }
        let total = start.elapsed().as_secs_f64();
        let metrics = stream.metrics();
        let (prefill_passes, decode_passes) = (metrics.num_prefill_forward_passes, metrics.num_decode_forward_passes);
        block_on(stream.close())?;
        let first_token = first_token.ok_or("the model generated no token")?;
        println!(
            "run {run}: {} prompt tokens, time to first token {first_token:.3} s, prefill {:.1} t/s, {generated} tokens \
             at {:.2} t/s, {prefill_passes} prefill and {decode_passes} decode passes, peak memory {:.2} GB, output {:016x}",
            input.len(),
            input.len() as f64 / first_token,
            (generated - 1) as f64 / (total - first_token),
            engine.peak_memory_usage().unwrap_or(0) as f64 / 1e9,
            output_digest.finish(),
        );
    }
    Ok(())
}
