use std::{net::IpAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rocket::{Config, config::LogLevel, routes};
use tokio::sync::Mutex;
use uzu::{
    engine::{Engine, EngineConfig},
    types::{model::ModelSpecialization, session::chat::ChatConfig},
};

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        ChatState, DecisionState, ServerState, handle_chat_completions, handle_decisions, handle_models,
        handle_system_one,
        log::{self, FileLogger, RequestLoggingFairing},
    },
};

pub async fn run_server(
    model: String,
    host: String,
    port: u16,
    prefix_cache: bool,
    verbose_file_logs: bool,
) -> Result<()> {
    let file_logger = if verbose_file_logs {
        let cache_dir = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
            .context("Failed to resolve cache directory")?;
        let logs_dir_path = cache_dir.join("mirai").join("server").join("logs");
        Some(FileLogger::new(logs_dir_path)?)
    } else {
        None
    };
    log::init(tracing::Level::INFO, file_logger.clone(), tracing::Level::DEBUG)?;

    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await.context("Failed to create engine")?;

    let resolved = engine.model(model.clone()).await?.with_context(|| format!("Model not found: {model}"))?;

    let spinner = ProgressBar::new_spinner();
    spinner.enable_steady_tick(Duration::from_millis(100));
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg}").unwrap());
    spinner.set_message(format!("Preparing model: {}", resolved.identifier));
    let downloader = engine.download(&resolved).await.context("Failed to start model download")?;
    while let Some(update) = downloader.next().await {
        spinner.set_message(format!("Downloading {}: {:.0}%", resolved.identifier, update.progress() * 100.0));
    }
    spinner.finish_with_message(format!("Loaded: {}", resolved.identifier));

    let model_name = resolved.identifier.clone();
    let address: IpAddr = host.parse().with_context(|| format!("Invalid host: {host}"))?;
    let config = Config {
        port,
        address,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    let rocket = rocket::custom(config)
        .manage(ServerState {
            model_name: model_name.clone(),
        })
        .attach(RequestLoggingFairing);

    let (rocket, title, details) = if resolved.specializations.contains(&ModelSpecialization::Decision {}) {
        let session = engine.decision(resolved).await.context("Failed to create decision session")?;
        let state = DecisionState {
            session: Arc::new(Mutex::new(session)),
        };
        let rocket = rocket
            .manage(state)
            .mount("/", routes![handle_system_one, handle_decisions, handle_models])
            .mount("/v1", routes![handle_system_one, handle_decisions, handle_models]);
        let details = "📝 Endpoints:\n   POST /v1/systemone        (or /v1/decisions, /systemone, /decisions)\n";
        (rocket, "System One server", details.to_string())
    } else {
        let thinking_support = ThinkingSupport::for_model(&resolved);
        let session = engine.chat(resolved, ChatConfig::default()).await.context("Failed to create chat session")?;
        let state = ChatState {
            session: Arc::new(Mutex::new(session)),
            thinking_support,
            prefix_cache,
        };
        let rocket = rocket
            .manage(state)
            .mount("/", routes![handle_chat_completions, handle_models])
            .mount("/v1", routes![handle_chat_completions, handle_models]);
        let prefix_cache_enabled = if prefix_cache {
            "enabled"
        } else {
            "disabled"
        };
        let details = format!(
            concat!(
                "🗄️  Prefix cache: {prefix_cache_enabled}\n",
                "📝 Endpoints:\n",
                "   POST /v1/chat/completions (or /chat/completions)\n",
            ),
            prefix_cache_enabled = prefix_cache_enabled,
        );
        (rocket, "OpenAI-compatible server", details)
    };

    if let Some(ref file_logger) = file_logger {
        tracing::info!("🗃️  Logs will be written to file: {}", file_logger.file_path().display())
    }
    tracing::info!(
        concat!(
            "🚀 {title} v{version} for model: {model_name}\n",
            "🌐 Available at: http://{host}:{port}\n",
            "{details}",
            "   GET  /v1/models           (or /models)\n",
        ),
        title = title,
        version = Engine::version(),
        model_name = model_name,
        host = host,
        port = port,
        details = details,
    );

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
