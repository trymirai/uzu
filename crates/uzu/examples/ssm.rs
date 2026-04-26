use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("cartesia-ai/Llamba-1B-4bit-mlx".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let messages = vec![ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string())];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.first() {
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
