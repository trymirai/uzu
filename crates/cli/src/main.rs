use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(feature = "capability-cli")]
    {
        use uzu::{cli::CliApplication, engine::EngineConfig};
        let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
        let application = CliApplication::create(engine_config).await?;
        application.run().await?;
    }
    Ok(())
}
