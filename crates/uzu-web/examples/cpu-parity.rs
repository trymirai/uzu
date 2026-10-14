//! Native reference for the browser numerical harness.
//! cargo run -p uzu-web --release --example cpu-parity -- MODEL_DIR PROMPT 64
#[allow(dead_code)]
#[path = "../src/chat.rs"]
mod chat;

use std::{path::PathBuf, time::Instant};

use serde_json::json;
use uzu_engine::{
    backends::cpu::Cpu,
    engine::{Engine, language_model::stream::SamplingMethod},
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut arguments = std::env::args().skip(1);
    let path = PathBuf::from(arguments.next().ok_or("expected model directory")?);
    let prompt = arguments.next().unwrap_or_else(|| "Explain why the sky is blue in detail.".into());
    let limit: usize = arguments.next().as_deref().unwrap_or("64").parse()?;
    let config: chat::ModelConfig = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let params: chat::RunParams =
        serde_json::from_value(json!({"messages":[{"role":"user","content":prompt}],"maxOutputTokens":limit}))?;
    let rendered = config.token_codec_config.render(&params, chrono::Utc::now())?;
    let engine = Engine::<Cpu>::new()?;
    let model = engine.load_language_model(&path)?;
    let input: Vec<u64> = model.tokenizer().encode(rendered, false)?.get_ids().iter().copied().map(u64::from).collect();
    let mut state = model.create_empty_state(Some(256), 42)?;
    let started = Instant::now();
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let mut stream = model.stream(&input, &mut state, options)?;
    let mut ids = Vec::new();
    for token in (&mut stream).take(limit) {
        let token = token?;
        if model.generation_config().stop_token_ids.contains(&token) {
            break;
        }
        ids.push(token as u32);
    }
    println!(
        "{}",
        json!({"backend":"cpu","prompt":prompt,"inputTokens":input,"tokenIds":ids,"text":model.tokenizer().decode(&ids,false)?,"duration":started.elapsed().as_secs_f64()})
    );
    Ok(())
}
