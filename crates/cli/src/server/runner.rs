use std::{net::IpAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rocket::{Config, config::LogLevel, routes};
use tokio::sync::Mutex;
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};

use crate::server::{ServerState, handle_chat_completions, handle_models};

pub async fn run_server(
    model: String,
    host: String,
    port: u16,
) -> Result<()> {
    let engine_config = EngineConfig::default().with_application_identifier("com.trymirai.cli".to_string());
    let engine = Engine::new(engine_config).await.context("Failed to create engine")?;

    let resolved = match engine.model(model.clone()).await? {
        Some(model) => model,
        None => engine.model_by_path(model.clone()).await?.with_context(|| format!("Model not found: {model}"))?,
    };

    let spinner = ProgressBar::new_spinner();
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
    spinner.set_message(format!("Preparing model: {}", resolved.identifier));
    let downloader = engine.download(&resolved).await.context("Failed to start model download")?;
    while let Some(update) = downloader.next().await {
        spinner.set_message(format!("Downloading {}: {:.0}%", resolved.identifier, update.progress() * 100.0));
    }
    spinner.finish_with_message(format!("Loaded: {}", resolved.identifier));

    let session =
        engine.chat(resolved.clone(), ChatConfig::default()).await.context("Failed to create chat session")?;
    let model_name = resolved.identifier.clone();
    let state = ServerState {
        model_name: model_name.clone(),
        session: Arc::new(Mutex::new(session)),
    };

    let address: IpAddr = host.parse().with_context(|| format!("Invalid host: {host}"))?;
    let config = Config {
        port,
        address,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    println!("🚀 OpenAI-compatible server for model: {model_name}");
    println!("🌐 Available at: http://{host}:{port}");
    println!("📝 Endpoints:");
    println!("   POST /v1/chat/completions (or /chat/completions)");
    println!("   GET  /v1/models           (or /models)");

    let rocket = rocket::custom(config)
        .manage(state)
        .mount("/", routes![handle_chat_completions, handle_models])
        .mount("/v1", routes![handle_chat_completions, handle_models]);

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
