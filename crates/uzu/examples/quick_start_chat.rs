use uzu::{
    engine::{Engine, EngineConfig},
    storage::types::DownloadPhase,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let engine = Engine::new(config).await?;

    let model = engine.model("alibaba:qwen3:0.6b".to_string()).await?.ok_or("Model not found")?;

    let downloader = engine.downloader(&model);
    let state = downloader.state().await.ok_or("Unable to get download state")?;
    if !matches!(state.phase, DownloadPhase::Downloaded {}) {
        downloader.resume().await?;
        let stream = downloader.progress().await?;
        while let Some(update) = stream.next().await {
            println!("Downloading: {}", update.progress());
        }
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
