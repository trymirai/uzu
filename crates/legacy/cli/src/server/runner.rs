use std::{net::IpAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rocket::{
    Config, Responder, State,
    config::LogLevel,
    fairing::AdHoc,
    get,
    http::{Header, Status},
    routes,
};
use tokio::sync::Mutex;
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        ServerState, handle_chat_completions, handle_models, log::Logger, request_info::RequestInfo,
        response_logger::ResponseBodyLogger,
    },
};

#[derive(Responder)]
#[response(content_type = "application/zip")]
struct LogArchive {
    file: std::fs::File,
    content_disposition: Header<'static>,
}

#[get("/logs")]
async fn handle_logs(logger: &State<Logger>) -> std::result::Result<LogArchive, Status> {
    match logger.get_file_archive().await {
        Ok(Some(file)) => Ok(LogArchive {
            file,
            content_disposition: Header::new("Content-Disposition", "attachment; filename=\"mirai-server-logs.zip\""),
        }),
        Ok(None) => Err(Status::NotFound),
        Err(error) => {
            logger.err(format!("Failed to create log archive: {error}"));
            Err(Status::InternalServerError)
        },
    }
}

pub async fn run_server(
    model: String,
    host: String,
    port: u16,
    prefix_cache: bool,
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

    let thinking_support = ThinkingSupport::for_model(&resolved);
    let session =
        engine.chat(resolved.clone(), ChatConfig::default()).await.context("Failed to create chat session")?;
    let model_name = resolved.identifier.clone();
    let state = ServerState {
        model_name: model_name.clone(),
        session: Arc::new(Mutex::new(session)),
        thinking_support,
        prefix_cache,
    };

    let address: IpAddr = host.parse().with_context(|| format!("Invalid host: {host}"))?;
    let config = Config {
        port,
        address,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    let cache_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
        .context("Failed to resolve cache directory")?;
    let logs_dir_path = cache_dir.join("mirai").join("server").join("logs");
    let logger = Logger::new(true, Some(logs_dir_path))?;

    let prefix_cache_enabled = if prefix_cache {
        "enabled"
    } else {
        "disabled"
    };
    logger.msg(format!(
        concat!(
            "🚀 OpenAI-compatible server v{version} for model: {model_name}\n",
            "🌐 Available at: http://{host}:{port}\n",
            "🗄️  Prefix cache: {prefix_cache_enabled}\n",
            "📝 Endpoints:\n",
            "   POST /v1/chat/completions (or /chat/completions)\n",
            "   GET  /v1/models           (or /models)\n",
            "   GET  /logs\n",
        ),
        version = Engine::version(),
        model_name = model_name,
        host = host,
        port = port,
        prefix_cache_enabled = prefix_cache_enabled,
    ));

    let rocket = rocket::custom(config)
        .manage(state)
        .manage(logger)
        .attach(AdHoc::on_request("Request logger", |req, _data| {
            Box::pin(async move {
                let _ = req.local_cache(|| RequestInfo::new(req.method(), req.uri().to_string()));
            })
        }))
        .attach(AdHoc::on_response("Response logger", |req, response| {
            Box::pin(async move {
                if matches!(req.uri().path().as_str(), "/logs") {
                    return;
                }

                let req_info = req.local_cache(|| RequestInfo::new(req.method(), req.uri().to_string()));
                let logger = req.rocket().state::<Logger>().expect("managed Logger");
                let prefix = format!("[{}] <-- {} {}", req_info.id_short(), response.status(), req.uri());
                if response.body().is_none() {
                    logger.msg(format!("{prefix} body=<empty>\n"));
                } else {
                    let is_json = response.content_type().is_some_and(|content_type| content_type.is_json());
                    let body = response.body_mut().take();
                    response.set_streamed_body(ResponseBodyLogger::new(body, logger, prefix, is_json));
                }
            })
        }))
        .mount("/", routes![handle_chat_completions, handle_models, handle_logs])
        .mount("/v1", routes![handle_chat_completions, handle_models]);

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
