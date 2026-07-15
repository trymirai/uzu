use nagare::tool::func_def::ToolFunctionDefinition;
use shoji::types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig};
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let model = engine.model("mlx-community/Qwen3.5-9B-MLX-8bit".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let mut session = engine.chat(model, ChatConfig::default()).await?;
    session
        .add_tool_functions(vec![
            ToolFunctionDefinition::new(
                "get_current_location".to_string(),
                "Get the user's current geographic location as latitude and longitude coordinates.".to_string(),
                None,
                Some(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude in decimal degrees."
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude in decimal degrees."
                            }
                        },
                        "required": ["latitude", "longitude"]
                    })
                    .into(),
                ),
                Box::new(|_args| {
                    Box::pin(async {
                        Ok(serde_json::json!({
                            "latitude": 59.938784,
                            "longitude": 30.314997,
                        })
                        .into())
                    })
                }),
            ),
            ToolFunctionDefinition::new(
                "get_current_temperature".to_string(),
                "Get the current temperature at the given geographic coordinates.".to_string(),
                Some(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "number",
                                "description": "Latitude in decimal degrees."
                            },
                            "longitude": {
                                "type": "number",
                                "description": "Longitude in decimal degrees."
                            }
                        },
                        "required": ["latitude", "longitude"]
                    })
                    .into(),
                ),
                Some(
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "temperature": {
                                "type": "number",
                                "description": "Current temperature in degrees Celsius."
                            }
                        },
                        "required": ["temperature"]
                    })
                    .into(),
                ),
                Box::new(|_args| {
                    Box::pin(async {
                        Ok(serde_json::json!({
                            "temperature": 25.9,
                        })
                        .into())
                    })
                }),
            ),
        ])
        .await?;

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("What is the weather now?".to_string()),
    ];

    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
