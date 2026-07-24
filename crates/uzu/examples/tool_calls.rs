use nagare::tool::{func_def::FutureError, schema::UzuToolSchema, uzu_tool_closure, uzu_tool_function};
use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::{SamplingMethod, SamplingPolicy},
    session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};
use uzu::engine::{Engine, EngineConfig};

/// A geographic coordinate.
#[derive(Serialize, Deserialize, UzuToolSchema)]
struct Coordinate {
    /// Latitude in decimal degrees.
    latitude: f64,
    /// Longitude in decimal degrees.
    longitude: f64,
}

/// Returns current location in coordinates
#[uzu_tool_function]
fn get_current_location() -> Result<Coordinate, FutureError> {
    Ok(Coordinate {
        latitude: 51.5074,
        longitude: -0.1278,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let model = engine.model("mlx-community/Qwen3.5-9B-MLX-8bit".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let mut session = engine.chat(model, ChatConfig::default()).await?;
    session.add_tool_function(get_current_location).await?;
    session
        .add_tool_function_definition(uzu_tool_closure! {
            /// Returns temperature in provided location
            get_current_temperature: |
                /// Latitude in decimal degrees.
                _latitude: f64,
                /// Longitude in decimal degrees.
                _longitude: f64
            | -> Result<f64, FutureError> {
                Ok(25.0)
            }
        })
        .await?;

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("What temperature is it now at my location?".to_string()),
    ];

    let config = ChatReplyConfig {
        sampling_policy: SamplingPolicy::Custom {
            method: SamplingMethod::Greedy {},
        },
        ..ChatReplyConfig::default()
    };
    let replies = session.reply(messages.clone(), config).await?;
    if let Some(reply) = replies.last() {
        println!("Reasoning: {}", reply.message.reasoning().unwrap_or_default());
        println!("Text: {}", reply.message.text().unwrap_or_default());
    }

    Ok(())
}
