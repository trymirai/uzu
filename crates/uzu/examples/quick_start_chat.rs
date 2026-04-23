use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let engine = Engine::new(config).await?;

    let model = engine.model("alibaba:qwen3:0.6b".to_string()).await?.ok_or("Model not found")?;
    while let Some(update) = engine.download(&model).await?.next().await {
        println!("Downloading: {}", update.progress());
    }

    let session = engine.chat(model, ChatConfig::default()).await?;
    let outputs = session
        .reply(vec![ChatMessage::user().with_text("Tell about London".to_string())], ChatReplyConfig::default())
        .await?;
    for output in outputs {
        println!("{}", output.message.text().unwrap_or_default());
    }

    Ok(())
}
