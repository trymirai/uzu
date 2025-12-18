use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use sysinfo::System;
use uzu::{
    VERSION,
    session::{
        ChatSession,
        config::{DecodingConfig, RunConfig},
        types::{Input, Output},
    },
};

use crate::runner::{
    helpers::get_memory_usage,
    types::{Device, Result as TaskResult, Task},
};

pub struct Runner {
    pub task: Task,
    pub model_path: String,
}

impl Runner {
    pub fn new(
        task: Task,
        model_path: String,
    ) -> Self {
        Self {
            task,
            model_path,
        }
    }

    pub fn run<F>(
        &self,
        mut progress: Option<F>,
    ) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>>
    where
        F: FnMut(f64),
    {
        let mut session = ChatSession::new(
            PathBuf::from(self.model_path.clone()),
            DecodingConfig::default(),
        )?;

        let precision =
            session.model_metadata.model_config.as_language_model().map(
                |config| {
                    config
                        .model_config
                        .transformer_config
                        .output_norm_config
                        .scale_precision
                },
            );

        let device = self.get_device_info();

        let mut results: Vec<TaskResult> = Vec::new();
        for run_index in 0..self.task.number_of_runs {
            let timestamp =
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

            let run_config =
                RunConfig::default().tokens_limit(self.task.tokens_limit);
            let output = session.run(
                Input::Messages(self.task.messages.clone()),
                run_config,
                Some(|_: Output| {
                    return true;
                }),
            )?;

            let memory_used = get_memory_usage();

            let result = TaskResult {
                task: self.task.clone(),
                device: device.clone(),
                engine_version: VERSION.to_string(),
                timestamp,
                precision,
                memory_used,
                tokens_count_input: output.stats.total_stats.tokens_count_input,
                tokens_count_output: output
                    .stats
                    .total_stats
                    .tokens_count_output,
                time_to_first_token: output.stats.prefill_stats.duration,
                prompt_tokens_per_second: output
                    .stats
                    .prefill_stats
                    .processed_tokens_per_second,
                generate_tokens_per_second: output
                    .stats
                    .generate_stats
                    .map(|stats| stats.tokens_per_second),
                text: output.text.original,
            };
            results.push(result);

            if let Some(progress) = progress.as_mut() {
                progress(
                    (run_index + 1) as f64 / self.task.number_of_runs as f64,
                );
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
        let cpu_name =
            system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();

        let device = Device {
            os_name,
            cpu_name,
            memory_total,
        };

        device
    }
}
