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

    let model = engine.model("gpt-5".to_string()).await?.ok_or("Model not found")?;

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
