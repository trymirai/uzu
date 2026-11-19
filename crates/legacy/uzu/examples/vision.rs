use std::{io, io::Write};

use shoji::types::{
    basic::ReasoningEffort,
    session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig},
};
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = "/Users/glk/Downloads/cat.jpg".to_string();
    if image_path.is_empty() {
        eprintln!("Empty image_path");
        return Ok(());
    }

    let engine = Engine::new(EngineConfig::default()).await?;
    let mut model = engine.model("/Users/glk/.cache/huggingface/hub/models--trymirai--Qwen3.5-9B-M-Vision/snapshots/6d3ea8cd5e0c00afca77554ab02c8381ed7f59fc".to_string()).await?.ok_or("Model not found")?;
    // TODO agolokoz: remove when model will appear in registry
    model.encoding = Some(serde_json::json!({"type": "hanashi", "name": "qwen3.5"}).into());

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
                url: image_path,
            })
            .with_text("What's on the image?".to_string()),
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
