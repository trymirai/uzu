use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

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

    while let Some(chunk) = stream.next().await {
        if let ChatSessionStreamChunk::Error {
            error,
        } = chunk
        {
            eprintln!("Error: {error}");
        }
    }

    Ok(())
}
