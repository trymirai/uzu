use std::{io, io::Write};

use shoji::types::{
    basic::ReasoningEffort,
    session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig},
};
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let model = engine.model("trymirai/Qwen3.5-9B-M-Vision".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
        io::stdout().flush()?;
    }
    println!();

    let session = engine.chat(model, ChatConfig::default()).await?;
    let messages = vec![
        ChatMessage::system()
            .with_text("You are a helpful assistant".to_string())
            .with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user()
            .with_block(ChatContentBlock::Image {
                url: "https://assets.trymirai.com/images/artifacts/photo-cat.jpeg".to_string(),
            })
            .with_block(ChatContentBlock::Image {
                url: "https://assets.trymirai.com/images/artifacts/photo-london.jpeg".to_string(),
            })
            .with_text("What's on the images?".to_string()),
    ];
    let replies = session
        .reply(
            messages,
            ChatReplyConfig {
                token_limit: Some(256),
                ..Default::default()
            },
        )
        .await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
