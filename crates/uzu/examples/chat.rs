use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
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

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];
    let session = engine.chat(model, ChatConfig::default()).await?;
    let stream = session.reply_with_stream(messages, ChatReplyConfig::default()).await;
    let mut last_message: Option<ChatMessage> = None;
    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.first() {
                    last_message = Some(reply.message.clone());
                    println!("Generated tokens: {}", reply.stats.tokens_count_output.unwrap_or_default());
                }
            },
            ChatSessionStreamChunk::Error {
                error,
            } => {
                println!("Error: {error}");
            },
        }
    }
    if let Some(message) = last_message {
        println!("Reasoning: {}", message.reasoning().unwrap_or_default());
        println!("Text: {}", message.text().unwrap_or_default());
    }

    Ok(())
}
