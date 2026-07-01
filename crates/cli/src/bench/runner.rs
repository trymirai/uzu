use std::{
    fs::File,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use backend_uzu::{
    VERSION,
    backends::metal::Metal,
    data_type::DataType,
    engine::{Engine as BackendUzuEngine, language_model::stream::SamplingMethod as BackendSamplingMethod},
};
use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::{Value, json};
use sysinfo::System;
use tokenizers::Tokenizer;
use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::SamplingMethod,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

use crate::bench::model::{BenchDevice, BenchMessageRole, BenchResult, BenchTask};

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
        progress: Option<F>,
    ) -> Result<Vec<BenchResult>> {
        self.run_stream(progress).await
    }

    #[allow(dead_code)]
    async fn run_stream<F: FnMut(f64)>(
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
                time_to_first_token = time_to_first_token / (replies.len() as f64);
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

    #[allow(dead_code)]
    async fn run_backend<F: FnMut(f64)>(
        &self,
        mut progress: Option<F>,
    ) -> Result<Vec<BenchResult>> {
        let model_path = PathBuf::from(self.model_path.trim_end_matches('/'));

        // Load config, tokenizer and stop tokens (mirrors examples/engine.rs).
        let config_file = File::open(model_path.join("config.json"))
            .with_context(|| format!("Failed to open config.json at {}", model_path.display()))?;
        let config: Value = serde_json::from_reader(config_file).context("Failed to parse config.json")?;
        let codec = &config["token_codec_config"];

        let stop_token_ids: Vec<u64> = config["generation_config"]["stop_token_ids"]
            .as_array()
            .map(|ids| ids.iter().filter_map(|id| id.as_u64()).collect())
            .unwrap_or_default();

        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|error| anyhow!("Failed to load tokenizer: {error}"))?;

        // Render the chat template into a prompt and tokenize it once.
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment
            .add_template("chat_template", codec["prompt_template"].as_str().context("Missing prompt_template")?)
            .context("Failed to add chat template")?;

        let messages: Vec<Value> = self
            .task
            .messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    BenchMessageRole::System => codec["system_role_name"].as_str().unwrap_or("system"),
                    BenchMessageRole::User => codec["user_role_name"].as_str().unwrap_or("user"),
                    BenchMessageRole::Assistant => codec["assistant_role_name"].as_str().unwrap_or("assistant"),
                };
                json!({ "role": role, "content": message.content })
            })
            .collect();

        let prompt = environment
            .get_template("chat_template")
            .context("Chat template not found")?
            .render(context!(
                messages => messages,
                add_generation_prompt => true,
                bos_token => codec["bos_token"].as_str(),
                eos_token => codec["eos_token"].as_str(),
                enable_thinking => false,
            ))
            .context("Failed to render chat template")?;

        let prompt_tokens: Vec<u64> = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| anyhow!("Failed to encode prompt: {error}"))?
            .get_ids()
            .iter()
            .map(|&token| u64::from(token))
            .collect();

        // Set up the backend engine and load the language model directly.
        let engine =
            BackendUzuEngine::<Metal>::new().map_err(|error| anyhow!("Failed to create backend engine: {error}"))?;
        let model = engine
            .load_language_model(&model_path)
            .map_err(|error| anyhow!("Failed to load language model: {error}"))?;

        let sampling_method = if self.task.greedy {
            BackendSamplingMethod::Greedy
        } else {
            model.default_sampling_method()
        };
        let token_limit = (self.task.tokens_limit as usize).max(1);
        let device = self.get_device_info();

        // Warmup: compile prefill + single-token decode kernels before timing.
        {
            let mut state = model
                .create_empty_state(model.recommended_context_length())
                .map_err(|error| anyhow!("Failed to create model state: {error}"))?;
            let mut options = model.default_stream_options();
            options.sampling_method = sampling_method.clone();
            let mut stream = model
                .stream(&prompt_tokens, &mut state, options)
                .map_err(|error| anyhow!("Failed to start warmup stream: {error}"))?;
            for _ in 0..8 {
                match stream.next() {
                    Some(Ok(token)) => {
                        if stop_token_ids.contains(&token) {
                            break;
                        }
                    },
                    Some(Err(error)) => return Err(anyhow!("Warmup stream error: {error}")),
                    None => break,
                }
            }
        }

        let mut results = Vec::<BenchResult>::new();
        for run_idx in 0..self.task.number_of_runs {
            let mut state = model
                .create_empty_state(model.recommended_context_length())
                .map_err(|error| anyhow!("Failed to create model state: {error}"))?;
            let mut options = model.default_stream_options();
            options.sampling_method = sampling_method.clone();

            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

            let mut output = Vec::<u32>::new();

            let end_to_end_start = Instant::now();
            let mut stream = model
                .stream(&prompt_tokens, &mut state, options)
                .map_err(|error| anyhow!("Failed to start stream: {error}"))?;

            // First token: everything up to here is prefill (time-to-first-token).
            let Some(first_token) = stream.next() else {
                bail!("Stream produced no tokens");
            };
            let first_token = first_token.map_err(|error| anyhow!("Stream error: {error}"))?;
            let time_to_first_token = end_to_end_start.elapsed().as_secs_f64();
            output.push(first_token as u32);

            // Remaining tokens: pure decode.
            let decode_start = Instant::now();
            if !stop_token_ids.contains(&first_token) && output.len() < token_limit {
                for token in stream {
                    let token = token.map_err(|error| anyhow!("Stream error: {error}"))?;
                    output.push(token as u32);
                    if stop_token_ids.contains(&token) || output.len() >= token_limit {
                        break;
                    }
                }
            }
            let decode_elapsed = decode_start.elapsed().as_secs_f64();

            let output_count = output.len();
            let prompt_tokens_per_second = if time_to_first_token > 0.0 {
                prompt_tokens.len() as f64 / time_to_first_token
            } else {
                0.0
            };
            let generate_tokens_per_second = if output_count > 1 && decode_elapsed > 0.0 {
                (output_count - 1) as f64 / decode_elapsed
            } else {
                0.0
            };

            let text = tokenizer.decode(&output, false).map_err(|error| anyhow!("Failed to decode output: {error}"))?;

            let result = BenchResult {
                task: self.task.clone(),
                device: device.clone(),
                engine_version: VERSION.to_string(),
                timestamp,
                data_type: model.data_type(),
                memory_used: engine.peak_memory_usage(),
                tokens_count_input: prompt_tokens.len() as u64,
                tokens_count_output: output_count as u64,
                time_to_first_token,
                prompt_tokens_per_second,
                generate_tokens_per_second: Some(generate_tokens_per_second),
                text,
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
