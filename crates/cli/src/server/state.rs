use std::{path::PathBuf, sync::Mutex};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::{
    ModelMetadata, ModelType,
    prelude::SamplingSeed,
    session::{
        ClassificationSession, Session, config::DecodingConfig,
        parameter::PrefillStepSize,
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

unsafe impl Send for SessionState {}
unsafe impl Sync for SessionState {}

pub fn load_model_type(model_path: &str) -> ModelType {
    let config_path = PathBuf::from(model_path).join("config.json");
    let config_file = std::fs::File::open(&config_path)
        .expect("Failed to open config.json");
    let metadata: ModelMetadata =
        serde_json::from_reader(std::io::BufReader::new(config_file))
            .expect("Failed to parse config.json");
    metadata.model_type
}

fn make_loading_progress_bar(model_path: &str) -> (ProgressBar, String) {
    let style_bold = Style::new().bold();
    let model_name = style_bold
        .apply_to(
            PathBuf::from(model_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
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
    (progress_bar, model_name)
}

fn finish_loading_progress_bar(
    progress_bar: &ProgressBar,
    model_name: String,
) {
    progress_bar.set_style(
        ProgressStyle::default_spinner()
            .template("Loaded: {msg}")
            .unwrap(),
    );
    progress_bar.finish_with_message(model_name);
}

pub fn load_session(
    model_path: String,
    prefill_step_size: Option<usize>,
    seed: Option<u64>,
) -> Session {
    let (progress_bar, model_name) = make_loading_progress_bar(&model_path);

    let model_path_buf = PathBuf::from(model_path);
    let prefill_step_size_config = match prefill_step_size {
        Some(value) => PrefillStepSize::Custom(value),
        None => PrefillStepSize::Default,
    };

    let decoding_config = DecodingConfig::default()
        .with_prefill_step_size(prefill_step_size_config)
        .with_sampling_seed(match seed {
            Some(seed) => SamplingSeed::Custom(seed),
            None => SamplingSeed::Default,
        });
    let session = Session::new(model_path_buf, decoding_config)
        .expect("Failed to create session");

    finish_loading_progress_bar(&progress_bar, model_name);
    session
}

pub fn load_classification_session(
    model_path: String,
) -> ClassificationSession {
    let (progress_bar, model_name) = make_loading_progress_bar(&model_path);

    let model_path_buf = PathBuf::from(model_path);
    let session = ClassificationSession::new(model_path_buf)
        .expect("Failed to create classification session");

    finish_loading_progress_bar(&progress_bar, model_name);
    session
}
