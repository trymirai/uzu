use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use backend_uzu::{VERSION, data_type::DataType};
use sysinfo::System;
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::SamplingMethod,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

use crate::bench::model::{BenchDevice, BenchResult, BenchTask};

pub struct BenchRunner {
    pub task: BenchTask,
    pub model_path: String,
    pub prefill_step_size: Option<usize>,
}

impl BenchRunner {
    pub fn new(
        task: BenchTask,
        model_path: String,
        prefill_step_size: Option<usize>,
    ) -> Self {
        Self {
            task,
            model_path,
            prefill_step_size,
        }
    }

    pub async fn run<F: FnMut(f64)>(
        &self,
        mut progress: Option<F>,
    ) -> Result<Vec<BenchResult>> {
        let engine_config = EngineConfig::default();
        let engine = Engine::new(engine_config).await.with_context(|| "Can not create engine".to_string())?;
        let model_path = self.model_path.trim_end_matches('/').to_string();
        let model = engine
            .model(model_path.clone())
            .await?
            .with_context(|| format!("Model not found at path: {model_path}"))?;
        let device = self.get_device_info();

        let messages: Vec<ChatMessage> = self.task.messages.iter().map(|msg| msg.to_chat_message()).collect();

        let session_config = ChatConfig::default();
        let session = engine.chat(model, session_config).await?;

        let warmup_config = ChatReplyConfig::default().with_token_limit(Some(1));
        let _ = session.reply(messages.clone(), warmup_config).await?;

        let mut results = Vec::<BenchResult>::new();
        for run_idx in 0..self.task.number_of_runs {
            session.reset().await?;

            let mut reply_config = ChatReplyConfig::default();
            if self.task.greedy {
                reply_config = reply_config.with_sampling_method(SamplingMethod::Greedy {})
            }

            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
            let replies = session.reply(messages.clone(), reply_config).await?;

            let mut tokens_count_input = 0u64;
            let mut tokens_count_output = 0u64;
            let mut time_to_first_token = 0.0f64;
            let mut prompt_tokens_per_second = 0.0f64;
            let mut generate_tokens_per_second = 0.0f64;
            for reply in replies.iter() {
                tokens_count_input += reply.stats.tokens_count_input.unwrap_or(0) as u64;
                tokens_count_output += reply.stats.tokens_count_output.unwrap_or(0) as u64;
                time_to_first_token += reply.stats.time_to_first_token.unwrap_or(0.0f64);
                prompt_tokens_per_second += reply.stats.prefill_tokens_per_second.unwrap_or(0.0f64);
                generate_tokens_per_second += reply.stats.generate_tokens_per_second.unwrap_or(0.0f64);
            }

            let mut text: Option<String> = None;
            if replies.len() > 0 {
                let replies_count = replies.len() as f64;
                time_to_first_token = time_to_first_token / replies_count;
                prompt_tokens_per_second = prompt_tokens_per_second / replies_count;
                generate_tokens_per_second = generate_tokens_per_second / replies_count;
                text = replies.last().unwrap().message.text();
            }

            let result = BenchResult {
                task: self.task.clone(),
                device: device.clone(),
                engine_version: VERSION.to_string(),
                timestamp,
                data_type: DataType::BF16,
                memory_used: session.peak_memory_usage().await,
                tokens_count_input,
                tokens_count_output,
                time_to_first_token,
                prompt_tokens_per_second,
                generate_tokens_per_second: Some(generate_tokens_per_second),
                text: text.unwrap_or("".to_string()),
            };
            results.push(result);

            if let Some(progress) = progress.as_mut() {
                progress((run_idx + 1) as f64 / self.task.number_of_runs as f64);
            }
        }

        Ok(results)
    }

    fn get_device_info(&self) -> BenchDevice {
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let os_name = System::long_os_version();
        let cpu_name = system_info.cpus().first().map(|cpu| cpu.brand().to_string());
        let memory_total = system_info.total_memory();

        BenchDevice {
            os_name,
            cpu_name,
            memory_total,
        }
    }
}
