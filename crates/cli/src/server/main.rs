use std::path::PathBuf;

use log::LevelFilter;
use rocket::{Config, config::LogLevel, log::private as log, routes};

use crate::server::{
    SessionState, SessionWrapper, handle_chat_completions, load_session,
};

struct SilentLogger;
static SILENT_LOGGER: SilentLogger = SilentLogger;
impl log::Log for SilentLogger {
    fn enabled(
        &self,
        _: &log::Metadata<'_>,
    ) -> bool {
        false
    }
    fn log(
        &self,
        _record: &log::Record<'_>,
    ) {
    }
    fn flush(&self) {}
}
// -------------------------------------------------------------------------------

pub async fn run_server(
    model_path: String,
    prefill_step_size: Option<usize>,
) {
    // Install the silent logger **before** Rocket initializes its own logger.
    let _ = log::set_logger(&SILENT_LOGGER)
        .map(|_| log::set_max_level(LevelFilter::Off));

    let config = Config {
        workers: 1,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    let model_name = PathBuf::from(model_path.clone())
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    println!("🚀 Starting server with model: {}", model_name);
    println!("📂 Model path: {}", model_path);
    println!(
        "🌐 Server will be available at: http://localhost:{}",
        config.port
    );
    println!(
        "📝 Endpoints:\n   POST /chat/completions - Chat completions API\n"
    );

    let session = load_session(model_path, prefill_step_size, None);
    let state = SessionState {
        model_name,
        session_wrapper: SessionWrapper::new(session),
    };

    let _ = rocket::custom(config)
        .manage(state)
        .mount("/", routes![handle_chat_completions])
        .launch()
        .await;
}
