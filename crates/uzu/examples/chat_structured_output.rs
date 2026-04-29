use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{Grammar, ReasoningEffort},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Country {
    name: String,
    capital: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CountryList {
    countries: Vec<Country>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let schema_string = serde_json::to_string(&schema_for!(CountryList))?;
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields"
                .to_string(),
        ),
    ];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let chat_reply_config = ChatReplyConfig::default().with_grammar(Some(Grammar::JsonSchema {
        schema: schema_string,
    }));
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        if let Some(text) = reply.message.text() {
            let parsed: CountryList = serde_json::from_str(&text)?;
            println!("{parsed:#?}");
        }
    }

    Ok(())
}
