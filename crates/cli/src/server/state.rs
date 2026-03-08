use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::{
    prelude::SamplingSeed,
    session::{
        Session,
        config::{DecodingConfig, RunConfig},
        parameter::PrefillStepSize,
        types::{Error, Input, Output},
    },
};

use crate::speculator_args::SpeculatorArgs;

pub trait RunSession {
    fn run(
        &mut self,
        input: Input,
        config: RunConfig,
        progress: Option<Box<dyn Fn(Output) -> bool>>,
    ) -> Result<Output, Error>;
}

impl RunSession for Session {
    fn run(
        &mut self,
        input: Input,
        config: RunConfig,
        progress: Option<Box<dyn Fn(Output) -> bool>>,
    ) -> Result<Output, Error> {
        Session::run(self, input, config, progress)
    }
}

pub struct SessionWrapper(Mutex<Box<dyn RunSession>>);

unsafe impl Send for SessionWrapper {}
unsafe impl Sync for SessionWrapper {}

impl SessionWrapper {
    pub fn new(runner: impl RunSession + 'static) -> Self {
        Self(Mutex::new(Box::new(runner)))
    }

    pub fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn RunSession>> {
        self.0.lock().unwrap()
    }
}

pub struct SessionState {
    pub model_name: String,
    pub session_wrapper: Arc<SessionWrapper>,
}

pub fn load_session(
    model_path: String,
    prefill_step_size: Option<usize>,
    seed: Option<u64>,
    speculator_args: SpeculatorArgs,
) -> Session {
    let style_bold = Style::new().bold();

    let model_path_buf = PathBuf::from(model_path);
    let model_name = style_bold.apply_to(model_path_buf.file_name().unwrap().to_str().unwrap().to_string()).to_string();

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    progress_bar.set_style(ProgressStyle::default_spinner().template("{spinner:.green} Loading: {msg}").unwrap());
    progress_bar.set_message(model_name.clone());

    let prefill_step_size_config = match prefill_step_size {
        Some(value) => PrefillStepSize::Custom(value),
        None => PrefillStepSize::Default,
    };

    let decoding_config = DecodingConfig::default()
        .with_prefill_step_size(prefill_step_size_config)
        .with_sampling_seed(match seed {
            Some(seed) => SamplingSeed::Custom(seed),
            None => SamplingSeed::Default,
        })
        .with_speculator_config(speculator_args.build_speculator_config(&model_path_buf));
    let session = Session::new(model_path_buf, decoding_config).expect("Failed to create session");

    progress_bar.set_style(ProgressStyle::default_spinner().template("Loaded: {msg}").unwrap());
    progress_bar.finish_with_message(model_name.clone());

    session
}
