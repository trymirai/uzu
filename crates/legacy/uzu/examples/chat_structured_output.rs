use std::io::{self, Write};

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

    let model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
        io::stdout().flush()?;
    }
    println!();

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
    if let Some(reply) = replies.first()
        && let Some(text) = reply.message.text()
    {
        let parsed: CountryList = serde_json::from_str(&text)?;
        println!("{parsed:#?}");
    }

    Ok(())
}
