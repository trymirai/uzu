use std::io::{self, Write};

use uzu::{
    engine::{Engine, EngineConfig},
    types::session::classification::ClassificationMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("trymirai/chat-moderation-router".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
        io::stdout().flush()?;
    }
    println!();

    let messages = vec![ClassificationMessage::user("Hi".to_string())];

    let session = engine.classification(model).await?;
    let output = session.classify(messages).await?;
    println!("Output: {:?}", output.probabilities.values);

    Ok(())
}
