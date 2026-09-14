use std::{io, net::IpAddr, path::PathBuf, pin::Pin, sync::Arc, task::Poll, time::Duration};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rocket::{
    Config, Request, Response,
    config::LogLevel,
    data::Data,
    fairing::{Fairing, Info, Kind},
    response::Body,
    routes,
};
use tokio::{
    io::{AsyncRead, ReadBuf},
    sync::Mutex,
};
use uzu::{
    engine::{Engine, EngineConfig},
    types::session::chat::ChatConfig,
};

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        ServerState, handle_chat_completions, handle_models,
        log::{self, FileLogger},
        request_info::RequestInfo,
    },
};

pub struct ResponseBodyLogger<'r> {
    body: Body<'r>,
    prefix: String,
    bytes: Vec<u8>,
    is_json: bool,
    logged: bool,
    span: tracing::Span,
}

impl<'r> ResponseBodyLogger<'r> {
    pub fn new(
        body: Body<'r>,
        prefix: String,
        is_json: bool,
        span: tracing::Span,
    ) -> Self {
        Self {
            body,
            prefix,
            bytes: Vec::new(),
            is_json,
            logged: false,
            span,
        }
    }

    fn log(
        &mut self,
        suffix: &str,
    ) {
        if self.logged {
            return;
        }
        self.logged = true;

        let body = if self.is_json {
            serde_json::from_slice::<serde_json::Value>(&self.bytes)
                .map(|value| value.to_string())
                .unwrap_or_else(|_| format!("{:?}", String::from_utf8_lossy(&self.bytes)))
        } else {
            format!("{:?}", String::from_utf8_lossy(&self.bytes))
        };
        self.span.in_scope(|| tracing::debug!("{} body={}{}\n", self.prefix, body, suffix));
    }
}

impl AsyncRead for ResponseBodyLogger<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let span = this.span.clone();
        let _entered = span.enter();
        let filled_before = buffer.filled().len();

        match Pin::new(&mut this.body).poll_read(cx, buffer) {
            Poll::Ready(Ok(())) => {
                let filled_after = buffer.filled().len();
                if filled_after == filled_before {
                    this.log("");
                } else {
                    this.bytes.extend_from_slice(&buffer.filled()[filled_before..filled_after]);
                }
                Poll::Ready(Ok(()))
            },
            Poll::Ready(Err(error)) => {
                this.log(" [read error]");
                Poll::Ready(Err(error))
            },
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for ResponseBodyLogger<'_> {
    fn drop(&mut self) {
        self.log(" [incomplete]");
    }
}

struct RequestLoggingFairing;

#[rocket::async_trait]
impl Fairing for RequestLoggingFairing {
    fn info(&self) -> Info {
        Info {
            name: "Request and response logger",
            kind: Kind::Request | Kind::Response,
        }
    }

    async fn on_request(
        &self,
        request: &mut Request<'_>,
        _data: &mut Data<'_>,
    ) {
        let _ = request.local_cache(|| RequestInfo::new(request.method(), request.uri().to_string()));
    }

    async fn on_response<'r>(
        &self,
        request: &'r Request<'_>,
        response: &mut Response<'r>,
    ) {
        if matches!(request.uri().path().as_str(), "/logs") {
            return;
        }

        let request_info = request.local_cache(|| RequestInfo::new(request.method(), request.uri().to_string()));
        let _entered = request_info.span.enter();
        let prefix = format!("<-- {} {}", response.status(), request.uri());
        if response.body().is_none() {
            tracing::debug!("{prefix} body=<empty>\n");
        } else {
            let is_json = response.content_type().is_some_and(|content_type| content_type.is_json());
            let body = response.body_mut().take();
            response.set_streamed_body(ResponseBodyLogger::new(body, prefix, is_json, request_info.span.clone()));
        }
    }
}

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

    let prefix_cache_enabled = if prefix_cache {
        "enabled"
    } else {
        "disabled"
    };
    if let Some(ref file_logger) = file_logger {
        tracing::info!("🗃️  Logs will be written to file: {}", file_logger.file_path().display())
    }
    tracing::info!(
        concat!(
            "🚀 OpenAI-compatible server v{version} for model: {model_name}\n",
            "🌐 Available at: http://{host}:{port}\n",
            "🗄️  Prefix cache: {prefix_cache_enabled}\n",
            "📝 Endpoints:\n",
            "   POST /v1/chat/completions (or /chat/completions)\n",
            "   GET  /v1/models           (or /models)\n",
        ),
        version = Engine::version(),
        model_name = model_name,
        host = host,
        port = port,
        prefix_cache_enabled = prefix_cache_enabled,
    );

    let rocket = rocket::custom(config)
        .manage(state)
        .attach(RequestLoggingFairing)
        .mount("/", routes![handle_chat_completions, handle_models])
        .mount("/v1", routes![handle_chat_completions, handle_models]);

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
