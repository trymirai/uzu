use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let session = engine.chat(model, ChatConfig::default()).await?;

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];

    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
