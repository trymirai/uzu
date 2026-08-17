use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatReplyStats},
};

fn print_stats(stats: &ChatReplyStats) {
    const MISSING: &str = "—";

    let time_to_first_token =
        stats.time_to_first_token.map(|value| format!("{value:.2} s")).unwrap_or_else(|| MISSING.to_string());
    let prefill_speed = stats
        .prefill_tokens_per_second
        .map(|value| format!("{value:.2} t/s"))
        .unwrap_or_else(|| MISSING.to_string());
    let generation_speed = stats
        .generate_tokens_per_second
        .map(|value| format!("{value:.2} t/s"))
        .unwrap_or_else(|| MISSING.to_string());
    let tokens_per_forward_pass = stats
        .speculator_stats
        .as_ref()
        .map(|value| format!("{:.2} t/f", value.tokens_per_forward_pass))
        .unwrap_or_else(|| MISSING.to_string());
    let memory_used = stats
        .memory_used_bytes
        .map(|bytes| format!("{:.2} GB", bytes.max(0) as f64 / 1024.0 / 1024.0 / 1024.0))
        .unwrap_or_else(|| MISSING.to_string());
    let power = stats
        .power_stats
        .as_ref()
        .map(|value| format!("{:.2} W avg", value.average_total_watts))
        .unwrap_or_else(|| MISSING.to_string());
    let energy = stats
        .power_stats
        .as_ref()
        .map(|value| format!("{:.2} J", value.energy_joules))
        .unwrap_or_else(|| MISSING.to_string());
    let energy_per_token =
        stats.joules_per_token().map(|value| format!("{value:.3} J/tok")).unwrap_or_else(|| MISSING.to_string());

    println!("time to first token: {time_to_first_token}");
    println!("prefill speed: {prefill_speed}");
    println!("generation speed: {generation_speed}");
    println!("tokens per forward pass: {tokens_per_forward_pass}");
    println!("memory used: {memory_used}");
    println!("average power: {power}");
    println!("total energy: {energy}");
    println!("energy per token: {energy_per_token}");
    println!("duration: {:.2} s", stats.duration);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or("usage: run <model-path> <prompt>")?;
    let prompt = args.next().ok_or("usage: run <model-path> <prompt>")?;

    let model_path = model_path.trim_end_matches('/').to_string();
    let parent_path = std::path::PathBuf::from(&model_path)
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();

    let engine = Engine::new(EngineConfig::default().with_local_path(parent_path)).await?;
    let model =
        engine.model_by_path(model_path.clone()).await?.ok_or(format!("Model not found at path: {model_path}"))?;

    let messages = vec![ChatMessage::user().with_text(prompt)];
    let session = engine.chat(model, ChatConfig::default()).await?;
    let stream = session.reply_with_stream(messages, ChatReplyConfig::default()).await;

    let mut stats = None;
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            chunk = stream.next() => {
                let Some(chunk) = chunk else {
                    break;
                };
                match chunk {
                    ChatSessionStreamChunk::Replies {
                        replies,
                    } => {
                        for reply in replies {
                            stats = Some(reply.stats);
                        }
                    },
                    ChatSessionStreamChunk::Error {
                        error,
                    } => {
                        eprintln!("Error: {error}");
                    },
                }
            },
        }
    }

    if let Some(stats) = stats {
        print_stats(&stats);
    }

    Ok(())
}
