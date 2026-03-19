use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use console::Style;
use indicatif::{ProgressBar, ProgressStyle};
use uzu::{
    prelude::{SamplingSeed, SpeculatorConfig},
    session::{
        Session,
        config::{DecodingConfig, RunConfig},
        parameter::PrefillStepSize,
        types::{Error, Input, Output},
    },
};

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
    speculator: Option<String>,
) -> Session {
    let style_bold = Style::new().bold();

    let model_path_buf = PathBuf::from(model_path);
    let model_name = style_bold.apply_to(model_path_buf.file_name().unwrap().to_str().unwrap().to_string()).to_string();

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    progress_bar.set_style(ProgressStyle::default_spinner().template("{spinner:.green} Loading: {msg}").unwrap());
    progress_bar.set_message(model_name.clone());

    let prefill_step_size_config: PrefillStepSize;
    if let Some(value) = prefill_step_size {
        prefill_step_size_config = PrefillStepSize::Custom(value);
    } else {
        prefill_step_size_config = PrefillStepSize::Default;
    }

    let decoding_config = DecodingConfig::default()
        .with_prefill_step_size(prefill_step_size_config)
        .with_sampling_seed(match seed {
            Some(seed) => SamplingSeed::Custom(seed),
            None => SamplingSeed::Default,
        })
        .with_speculator_config(match speculator {
            Some(speculator) => {
                let parts: Vec<&str> = speculator.splitn(3, ':').collect();
                let speculator_path = parts[0];
                let number_of_speculated_tokens: usize = parts.get(1).unwrap_or(&"1").parse().unwrap();
                let temperature: Option<f32> = parts.get(2).map(|s| s.parse().unwrap());

                let speculator = Arc::new(uzu::speculators::ngram_speculator::NGramSpeculator::load_with_temperature(
                    speculator_path,
                    temperature,
                ));

                SpeculatorConfig {
                    number_of_speculated_tokens,
                    speculator,
                }
            },
            None => SpeculatorConfig::default(),
        });
    let session = Session::new(model_path_buf, decoding_config).expect("Failed to create session");

    progress_bar.set_style(ProgressStyle::default_spinner().template("Loaded: {msg}").unwrap());
    progress_bar.finish_with_message(model_name.clone());

    session
}
