use anyhow::Result;
use uzu::engine::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Engine::new(EngineConfig::default()).await?;
    let application = engine.cli();
    application.run().await?;
    Ok(())
}
