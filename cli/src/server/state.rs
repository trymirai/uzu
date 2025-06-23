use std::{path::PathBuf, sync::Mutex};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::{
    generator::config::{ContextLength, SamplingSeed},
    session::{
        session::Session,
        session_config::{SessionConfig, SessionPreset},
    },
};

pub struct SessionWrapper(Mutex<Session>);
unsafe impl Send for SessionWrapper {}
unsafe impl Sync for SessionWrapper {}
impl SessionWrapper {
    pub fn new(session: Session) -> Self {
        Self(Mutex::new(session))
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, Session> {
        self.0.lock().unwrap()
    }
}

pub struct SessionState {
    pub model_name: String,
    pub session_wrapper: SessionWrapper,
}

pub fn load_session(model_path: String) -> Session {
    let style_bold = Style::new().bold();

    let model_path_buf = PathBuf::from(model_path);
    let model_name = style_bold
        .apply_to(
            model_path_buf.file_name().unwrap().to_str().unwrap().to_string(),
        )
        .to_string();

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    progress_bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} Loading: {msg}")
            .unwrap(),
    );
    progress_bar.set_message(model_name.clone());

    let config = SessionConfig::new(
        SessionPreset::General,
        SamplingSeed::Default,
        ContextLength::Default,
    );
    let mut session =
        Session::new(model_path_buf).expect("Failed to create session");
    session.load(config).expect("Failed to load session");

    progress_bar.set_style(
        ProgressStyle::default_spinner().template("Loaded: {msg}").unwrap(),
    );
    progress_bar.finish_with_message(model_name.clone());

    session
}
