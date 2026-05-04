use anyhow::Result;
use uzu::{cli::CliApplication, engine::EngineConfig};

const APPLICATION_IDENTIFIER: &str = "com.trymirai.cli";

#[tokio::main]
async fn main() -> Result<()> {
    let engine_config = EngineConfig::default().with_application_identifier(APPLICATION_IDENTIFIER.to_string());
    let application = CliApplication::create(engine_config).await?;
    application.run().await?;
    Ok(())
}
