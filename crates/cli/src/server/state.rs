use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    rc::Rc,
    sync::Mutex,
};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::session::{
    Session,
    config::{DecodingConfig, SpeculatorConfig},
    helpers::Context,
    parameter::{ContextLength, PrefillStepSize, SamplingSeed},
};

pub struct ContextCache {
    pub map: HashMap<String, Rc<Context>>,
    pub order: VecDeque<String>,
    pub capacity: usize,
}

impl ContextCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn insert(
        &mut self,
        key: String,
        context: Rc<Context>,
    ) -> Option<(String, Rc<Context>)> {
        if self.map.contains_key(&key) {
            return None;
        }
        self.map.insert(key.clone(), context);
        self.order.push_back(key.clone());

        if self.order.len() > self.capacity {
            if let Some(old_key) = self.order.pop_front() {
                if let Some(old_context) = self.map.remove(&old_key) {
                    return Some((old_key, old_context));
                }
            }
        }
        None
    }

    pub fn get(
        &self,
        key: &str,
    ) -> Option<Rc<Context>> {
        self.map.get(key).cloned()
    }
}

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
    pub cache: Mutex<ContextCache>,
}

unsafe impl Send for SessionState {}
unsafe impl Sync for SessionState {}

pub fn load_session(
    model_path: String,
    prefill_step_size: Option<usize>,
) -> Session {
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

    let prefill_step_size_config: PrefillStepSize;
    if let Some(value) = prefill_step_size {
        prefill_step_size_config = PrefillStepSize::Custom(value);
    } else {
        prefill_step_size_config = PrefillStepSize::Default;
    }

    let decoding_config = DecodingConfig::new(
        prefill_step_size_config,
        ContextLength::default(),
        SpeculatorConfig::default(),
        SamplingSeed::default(),
        true,
    );
    let session = Session::new(model_path_buf, decoding_config)
        .expect("Failed to create session");

    progress_bar.set_style(
        ProgressStyle::default_spinner().template("Loaded: {msg}").unwrap(),
    );
    progress_bar.finish_with_message(model_name.clone());

    session
}
