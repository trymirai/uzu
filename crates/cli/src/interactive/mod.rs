use uzu::engine::EngineConfig;

use crate::interactive::app::CliApplication;

mod app;
mod components;
mod flows;
mod helpers;
mod sessions;

pub async fn run_interactive(model: Option<String>) -> anyhow::Result<()> {
    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let application = CliApplication::create(engine_config).await?;
    application.run_with_model(model).await?;
    Ok(())
}
