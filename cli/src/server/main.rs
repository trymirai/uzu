use std::path::PathBuf;

use rocket::{Config, config::LogLevel, routes};

use crate::server::{
    SessionState, SessionWrapper, handle_chat_completions, load_session,
};

pub async fn run_server(model_path: String) {
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
    let session = load_session(model_path);
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
