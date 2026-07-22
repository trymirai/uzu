use nagare::tool::func_def::ToolFunctionDefinition;
use shoji::types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig};
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let model = engine.model("Qwen/Qwen3.5-0.8B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let mut session = engine.chat(model, ChatConfig::default()).await?;
    session
        .add_tool_functions(vec![ToolFunctionDefinition::new(
            "get_current_time".to_string(),
            "Get the current time.".to_string(),
            None,
            Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Current time."
                        }
                    },
                    "required": ["time"]
                })
                .into(),
            ),
            Box::new(|_args| {
                Box::pin(async {
                    Ok(serde_json::json!({
                        "time": "17:03",
                    })
                    .into())
                })
            }),
        )])
        .await?;

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("What time is it now?".to_string()),
    ];
    let replies = session.reply(messages.clone(), ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
