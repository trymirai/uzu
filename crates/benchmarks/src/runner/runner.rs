use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use sysinfo::System;
use uzu::{
    ConfigDataType, VERSION,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        parameter::{ContextLength, PrefillStepSize, SamplingMethod, SamplingPolicy},
        types::{Input, Output},
    },
};

use crate::runner::{
    helpers::get_memory_usage,
    types::{BenchRunMode, Device, Result as TaskResult, Task},
};

pub struct Runner {
    pub task: Task,
    pub model_path: String,
    pub prefill_step_size: Option<usize>,
}

impl Runner {
    pub fn new(
        task: Task,
        model_path: String,
        prefill_step_size: Option<usize>,
    ) -> Self {
        Self {
            task,
            model_path,
            prefill_step_size,
        }
    }

    fn minimal_context_length(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let input = Input::Messages(self.task.messages.clone());
        let mut session = ChatSession::new(PathBuf::from(self.model_path.clone()), DecodingConfig::default())?;
        let run_config = RunConfig::default().tokens_limit(1);
        let output = session.run(
            input.clone(),
            run_config,
            Some(|_: Output| {
                return true;
            }),
        )?;
        return Ok(output.stats.total_stats.tokens_count_input as usize + self.task.tokens_limit as usize);
    }

    pub fn run<F>(
        &self,
        progress: Option<F>,
    ) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        let context_length = self.minimal_context_length()?;
        let device = self.get_device_info();
        let input = Input::Messages(self.task.messages.clone());
        match self.task.run_mode {
            BenchRunMode::WarmedProcess => self.run_warmed_process(context_length, device, input, progress),
            BenchRunMode::FreshSession => self.run_fresh_session(context_length, device, input, progress),
            BenchRunMode::FreshProcess => {
                panic!("FreshProcess must be orchestrated by the CLI handler, not Runner")
            },
        }
    }
}

impl Runner {
    fn run_warmed_process<F>(
        &self,
        context_length: usize,
        device: Device,
        input: Input,
        mut progress: Option<F>,
    ) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        let mut session = self.new_session(context_length)?;
        if self.task.warmup_tokens > 0 {
            let warmup_config = RunConfig::default().tokens_limit(self.task.warmup_tokens);
            session.run(input.clone(), warmup_config, Some(|_: Output| true))?;
        }
        let precision = session
            .model_metadata
            .model_config
            .as_language_model()
            .map(|config| config.model_config.transformer_config.output_norm_config.scale_precision);
        let mut results = Vec::with_capacity(self.task.number_of_runs as usize);
        for run_index in 0..self.task.number_of_runs {
            let result = self.run_once(&mut session, &device, precision, &input)?;
            results.push(result);
            if let Some(progress) = progress.as_mut() {
                progress((run_index + 1) as f64 / self.task.number_of_runs as f64);
            }
        }
        Ok(results)
    }

    fn run_fresh_session<F>(
        &self,
        context_length: usize,
        device: Device,
        input: Input,
        mut progress: Option<F>,
    ) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        let mut results = Vec::with_capacity(self.task.number_of_runs as usize);
        for run_index in 0..self.task.number_of_runs {
            let mut session = self.new_session(context_length)?;
            if self.task.warmup_tokens > 0 {
                let warmup_config = RunConfig::default().tokens_limit(self.task.warmup_tokens);
                session.run(input.clone(), warmup_config, Some(|_: Output| true))?;
            }
            let precision = session
                .model_metadata
                .model_config
                .as_language_model()
                .map(|config| config.model_config.transformer_config.output_norm_config.scale_precision);
            let result = self.run_once(&mut session, &device, precision, &input)?;
            results.push(result);
            if let Some(progress) = progress.as_mut() {
                progress((run_index + 1) as f64 / self.task.number_of_runs as f64);
            }
        }
        Ok(results)
    }

    fn new_session(
        &self,
        context_length: usize,
    ) -> Result<ChatSession, Box<dyn std::error::Error>> {
        let mut decoding_config = DecodingConfig::default().with_context_length(ContextLength::Custom(context_length));
        if let Some(prefill_step_size) = self.prefill_step_size {
            decoding_config = decoding_config.with_prefill_step_size(PrefillStepSize::Custom(prefill_step_size));
        }
        Ok(ChatSession::new(PathBuf::from(self.model_path.clone()), decoding_config)?)
    }

    fn run_once(
        &self,
        session: &mut ChatSession,
        device: &Device,
        precision: Option<ConfigDataType>,
        input: &Input,
    ) -> Result<TaskResult, Box<dyn std::error::Error>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let mut run_config = RunConfig::default().tokens_limit(self.task.tokens_limit);
        if self.task.greedy {
            run_config = run_config.sampling_policy(SamplingPolicy::Custom {
                value: SamplingMethod::Greedy,
            });
        }
        let output = if let Some(forced_token_path) = self.task.forced_token_path.as_deref() {
            session.run_forced_token_path(
                input.clone(),
                run_config,
                forced_token_path,
                Some(|_: Output| {
                    return true;
                }),
            )?
        } else {
            session.run(
                input.clone(),
                run_config,
                Some(|_: Output| {
                    return true;
                }),
            )?
        };
        let text = output.text.original.clone();
        let text_blake3 = blake3::hash(text.as_bytes()).to_hex().to_string();
        Ok(TaskResult {
            task: self.task.clone(),
            device: device.clone(),
            engine_version: VERSION.to_string(),
            timestamp,
            precision,
            memory_used: get_memory_usage(),
            kv_storage_bytes: session.kv_storage_bytes(),
            tokens_count_input: output.stats.total_stats.tokens_count_input,
            tokens_count_output: output.stats.total_stats.tokens_count_output,
            time_to_first_token: output.stats.prefill_stats.duration,
            prompt_tokens_per_second: output.stats.prefill_stats.processed_tokens_per_second,
            generate_tokens_per_second: output.stats.generate_stats.map(|stats| stats.tokens_per_second),
            text_blake3: text_blake3.clone(),
            matches_reference_output: self
                .task
                .reference_text_blake3
                .as_ref()
                .map(|reference| *reference == text_blake3),
            text,
        })
    }

    fn get_device_info(&self) -> Device {
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let os_name = System::long_os_version();
        let cpu_name = system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();

        let device = Device {
            os_name,
            cpu_name,
            memory_total,
        };

        device
    }
}
