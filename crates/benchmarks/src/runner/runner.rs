use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use backend_uzu::{
    VERSION,
    prelude::PromptLookupSpeculator,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig, SpeculatorConfig},
        parameter::{PrefillStepSize, SamplingMethod, SamplingPolicy},
        types::{Input, Output},
    },
};
use sysinfo::System;

use crate::runner::types::{Device, Result as TaskResult, SpeculatorSpec, Task};

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

    pub fn run<F>(
        &self,
        mut progress: Option<F>,
    ) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        let mut decoding_config = DecodingConfig::default();
        if let Some(prefill_step_size) = self.prefill_step_size {
            decoding_config = decoding_config.with_prefill_step_size(PrefillStepSize::Custom(prefill_step_size));
        }
        if let Some(speculator) = &self.task.speculator {
            let speculator_config = match speculator {
                SpeculatorSpec::PromptLookup {
                    max_ngram_size,
                    number_of_speculated_tokens,
                } => SpeculatorConfig::new(
                    *number_of_speculated_tokens,
                    Arc::new(PromptLookupSpeculator::new_with_params(*max_ngram_size)),
                ),
            };
            decoding_config = decoding_config.with_speculator_config(speculator_config);
        }

        let mut session = ChatSession::new(PathBuf::from(self.model_path.clone()), decoding_config)?;
        let data_type = session.activation_data_type();

        let device = self.get_device_info();

        let input = Input::Messages(self.task.messages.clone());

        let warmup_config = RunConfig::default().tokens_limit(1);
        session.run(input.clone(), warmup_config, Option::<fn(Output) -> bool>::None)?;

        let mut results: Vec<TaskResult> = Vec::new();
        for run_index in 0..self.task.number_of_runs {
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

            let mut run_config = RunConfig::default().tokens_limit(self.task.tokens_limit);
            if self.task.greedy {
                run_config = run_config.sampling_policy(SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                });
            }
            let output = session.run(input.clone(), run_config, Option::<fn(Output) -> bool>::None)?;

            let memory_used = session.peak_memory_usage();

            let (speculator_proposed, speculator_accepted) = output
                .stats
                .generate_stats
                .as_ref()
                .map(|stats| (stats.speculator_proposed, stats.speculator_accepted))
                .unwrap_or((0, 0));

            let result = TaskResult {
                task: self.task.clone(),
                device: device.clone(),
                engine_version: VERSION.to_string(),
                timestamp,
                data_type,
                memory_used,
                tokens_count_input: output.stats.total_stats.tokens_count_input,
                tokens_count_output: output.stats.total_stats.tokens_count_output,
                time_to_first_token: output.stats.prefill_stats.duration,
                prompt_tokens_per_second: output.stats.prefill_stats.processed_tokens_per_second,
                generate_tokens_per_second: output.stats.generate_stats.map(|stats| stats.tokens_per_second),
                speculator_proposed,
                speculator_accepted,
                text: output.text.original,
            };
            results.push(result);

            if let Some(progress) = progress.as_mut() {
                progress((run_index + 1) as f64 / self.task.number_of_runs as f64);
            }
        }

        Ok(results)
    }
}

impl Runner {
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
