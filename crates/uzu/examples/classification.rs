use std::io::{self, Write};

use uzu::{
    engine::{Engine, EngineConfig},
    types::session::classification::ClassificationMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        print!("\r\u{001B}[2KDownload progress: {}", update.progress());
        io::stdout().flush()?;
    }
    println!();

    let messages = vec![ClassificationMessage::user("Hi".to_string())];

    let session = engine.classification(model).await?;
    let output = session.classify(messages).await?;
    println!("Output: {:?}", output.probabilities.values);

    Ok(())
}
