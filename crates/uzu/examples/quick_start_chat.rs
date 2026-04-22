use uzu::{
    engine::{Config as EngineConfig, Engine},
    types::session::chat::{Config, Message, StreamConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let engine = Engine::new(config).await?;

    let model = engine.model("alibaba:qwen3:0.6b").await?.ok_or("Model not found")?;

    let downloader = engine.downloader(&model);
    downloader.resume().await?;
    if let Some(stream) = downloader.progress().await {
        while let Some(update) = stream.next().await {
            println!("Downloading: {}", update.progress());
        }
    }

    let session = engine.chat(model, Config::default()).await?;
    let outputs = session
        .reply(vec![Message::user().with_text("Tell about London".to_string())], StreamConfig::default())
        .await?;
    for output in outputs {
        println!("{}", output.message.text().unwrap_or_default());
    }

    Ok(())
}
