use std::io::{self, Write};

use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::ReasoningEffort,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default().with_openai_api_key("OPENAI_API_KEY".to_string());
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {}", update.progress());
        io::stdout().flush()?;
    }
    println!();

    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Low),
        ChatMessage::user().with_text("How LLMs work".to_string()),
    ];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.first() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
