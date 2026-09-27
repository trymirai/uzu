use std::{collections::HashMap, io, io::Write, str::FromStr, sync::Arc, time::Instant};

use anyhow::{Context, ensure};
use futures::StreamExt;
use hanashi::{Encoding as _, chat::hanashi::HanashiEncodingImpl};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatInstanceKind,
    traits::backend::chat_token::{Instance, TokenStreamOutput},
    types::{
        basic::{SamplingMethod, ToolCall, ToolDescription, ToolFunction, ToolNamespace},
        session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatRole},
    },
};

use crate::{
    bench::{BenchRequest, BenchResponse},
    common::InferenceEngine,
    memory::MemoryCounters,
};

pub struct UzuEngine {
    instance: Arc<dyn Instance>,
    encoding: HanashiEncodingImpl,
    stop_tokens: Box<[u64]>,
}

impl UzuEngine {
    pub async fn new(model: &str) -> anyhow::Result<Self> {
        let engine = Engine::new(EngineConfig::default()).await?;
        let model = engine.model(model.to_string()).await?.context("Model not found")?;
        ensure!(model.is_on_device(), "Uzu benchmarks require an on-device model");

        let downloader = engine.download(&model).await?;
        while let Some(update) = downloader.next().await {
            eprint!("\r\u{001B}[2KDownload progress: {:.2}%", update.progress() * 100.0);
            io::stderr().flush()?;
        }
        eprintln!();

        engine.model_path(&model).await.context("Model download did not complete")?;
        let encoding_config =
            serde_json::from_str(&model.encoding.as_ref().context("Model has no chat encoding")?.json)
                .context("Invalid Hanashi encoding config")?;
        let shared = engine.chat_instance(model, ChatConfig::default()).await?;
        let ChatInstanceKind::Token(instance) = shared.kind() else {
            anyhow::bail!("Uzu benchmarks require a token backend");
        };
        let encoding = HanashiEncodingImpl::new(encoding_config, instance.tokenizer())?;

        let stop_tokens = instance.stop_token_ids().context("Model has no stop token IDs")?;
        Ok(Self {
            instance,
            encoding,
            stop_tokens,
        })
    }

    fn tokenize(
        &mut self,
        request: &BenchRequest,
    ) -> anyhow::Result<Vec<u64>> {
        if let Some(text) = &request.prompt_text {
            let encoded = self.instance.tokenizer().encode(text.as_str(), true).map_err(anyhow::Error::msg)?;
            return Ok(encoded.get_ids().iter().copied().map(u64::from).collect());
        }

        let messages = get_chat_messages(request)?;
        self.encoding.reset()?;
        self.encoding.encode(messages)?;
        let tokens = self.encoding.state().tokens.iter().map(|token| u64::from(token.id)).collect();
        Ok(tokens)
    }

    async fn run_single(
        &self,
        tokens: &Vec<u64>,
        config: ChatReplyConfig,
    ) -> anyhow::Result<BenchResponse> {
        // share model weights, but give every run a fresh KV cache and sampling state.
        let mut state = self.instance.state().await.map_err(anyhow::Error::msg)?;
        let max_tokens = config.token_limit.context("Missing token limit")? as usize;
        let mut memory = MemoryCounters::collect()?;
        let mut output = Vec::new();
        let mut tokens_generated = 0;
        let mut first_token = None;

        let started = Instant::now();
        let mut stream = self.instance.stream(tokens, state.as_mut(), config, CancellationToken::new());
        while tokens_generated < max_tokens {
            let Some(event) = stream.next().await else {
                break;
            };
            let TokenStreamOutput::Token(token) = event.map_err(anyhow::Error::msg)? else {
                break;
            };
            first_token.get_or_insert_with(Instant::now);
            tokens_generated += 1;

            let current = MemoryCounters::collect()?;
            if current.graphics_total > memory.graphics_total {
                memory = current;
            }
            if self.stop_tokens.contains(&token) {
                break;
            }
            output.push(u32::try_from(token).context("Token ID exceeds u32")?);
        }
        let first_token = first_token.context("Generation did not return a token")?;
        let metrics = stream.metrics().context("Generation did not return metrics")?;
        let forward_passes = metrics.num_prefill_forward_passes + metrics.num_decode_forward_passes;
        drop(stream);

        let text = self.instance.tokenizer().decode(&output, false).map_err(anyhow::Error::msg)?;
        let finished = Instant::now();
        let time_to_first_token = first_token.duration_since(started).as_secs_f64();
        let decode_duration = finished.duration_since(first_token).as_secs_f64();
        Ok(BenchResponse {
            text,
            time_to_first_token,
            prompt_tps: rate(tokens.len(), time_to_first_token),
            decode_tps: rate(tokens_generated.saturating_sub(1), decode_duration),
            tokens_per_forward_pass: rate(tokens_generated, forward_passes as f64),
            duration: finished.duration_since(started).as_secs_f64(),
            memory_phys_footprint: memory.phys_footprint,
            memory_resident_peak: memory.resident_size_peak,
            memory_graphics_total: memory.graphics_total,
        })
    }
}

impl InferenceEngine for UzuEngine {
    async fn execute(
        &mut self,
        request: &BenchRequest,
    ) -> anyhow::Result<Vec<BenchResponse>> {
        let (num_runs, config) = get_config(request)?;
        let tokens = self.tokenize(request)?;
        ensure!(!tokens.is_empty(), "Prompt must contain at least one token");

        let context_length =
            tokens.len().checked_add(config.token_limit.unwrap() as usize).context("Context size overflow")?;
        if let Some(limit) = self.instance.max_context_length() {
            ensure!(
                context_length <= limit,
                "Prompt and generation length exceed the supported context size ({limit})"
            );
        }

        let mut responses = Vec::new();
        for _ in 0..num_runs {
            let response = self.run_single(&tokens, config.clone()).await?;
            responses.push(response);
        }

        Ok(responses)
    }
}

fn get_config(request: &BenchRequest) -> anyhow::Result<(usize, ChatReplyConfig)> {
    let num_runs = request.num_runs.unwrap_or(1);
    ensure!(num_runs > 0, "num_runs must be 1 or greater");
    ensure!(request.prompt_text.is_some() || request.prompt_chat.is_some(), "prompt_text and prompt_chat are absent");
    ensure!(
        request.speculative_depth.is_none(),
        "speculative_depth is not configurable through the Uzu API; omit it to use the model default"
    );

    let max_tokens = request.max_tokens.unwrap_or(256) as u32;
    ensure!(max_tokens > 0, "max_tokens must be 1 or greater");

    let sampling = match &request.sampling {
        None => SamplingMethod::Greedy {},
        Some(sampling) if sampling.temp.is_some_and(|temp| temp <= 0.0) => SamplingMethod::Greedy {},
        Some(sampling) => SamplingMethod::Stochastic {
            temperature: sampling.temp.map(f64::from),
            top_k: sampling.top_k.filter(|value| *value > 0).map(i64::from),
            top_p: sampling.top_p.filter(|value| *value > 0.0 && *value < 1.0).map(f64::from),
            min_p: sampling.min_p.filter(|value| *value > 0.0).map(f64::from),
            repetition_penalty: None,
            suffix_repetition_length: None,
        },
    };

    let config = ChatReplyConfig::default().with_token_limit(Some(max_tokens)).with_sampling_method(sampling);
    Ok((num_runs, config))
}

fn get_chat_messages(request: &BenchRequest) -> anyhow::Result<Vec<ChatMessage>> {
    let input = request.prompt_chat.as_ref().context("prompt_text and prompt_chat are absent")?;
    ensure!(!input.is_empty(), "prompt_chat must not be empty");

    let mut messages = Vec::new();
    let mut tool_names = HashMap::new();
    for message in input {
        let role = ChatRole::from_str(&message.role).map_err(anyhow::Error::msg)?;
        let mut converted = ChatMessage::for_role(role);
        if let Some(reasoning) = &message.reasoning_content {
            converted = converted.with_reasoning(reasoning.clone());
        }

        if let Some(identifier) = &message.tool_call_id {
            converted = converted.with_block(ChatContentBlock::ToolCallResult {
                identifier: Some(identifier.clone()),
                name: tool_names.get(identifier).cloned(),
                value: Value::String(message.content.clone().unwrap_or_default()).into(),
            });
        } else if let Some(content) = &message.content {
            converted = converted.with_text(content.clone());
        }

        for call in message.tool_calls.iter().flatten() {
            ensure!(call["type"] == "function", "Only function tool calls are supported");
            let function = &call["function"];
            let name = function["name"].as_str().context("Tool call has no function name")?.to_string();
            let arguments = match &function["arguments"] {
                Value::String(text) => serde_json::from_str(text).context("Invalid tool call arguments")?,
                value => value.clone(),
            };
            let identifier = call["id"].as_str().map(str::to_string);
            if let Some(identifier) = &identifier {
                tool_names.insert(identifier.clone(), name.clone());
            }

            converted = converted.with_tool_call(ToolCall {
                identifier,
                name,
                arguments: arguments.into(),
            });
        }
        messages.push(converted);
    }

    let mut tools = Vec::new();
    for tool in request.tools.iter().flatten() {
        let function = &tool["function"];
        tools.push(ToolDescription::Function {
            tool_function: ToolFunction {
                name: function["name"].as_str().context("Tool has no function name")?.to_string(),
                description: function["description"].as_str().unwrap_or_default().to_string(),
                parameters: function.get("parameters").filter(|value| !value.is_null()).cloned().map(Into::into),
                return_definition: None,
            },
        });
    }

    match request.tool_choice.as_ref().filter(|choice| !choice.is_null()) {
        None => {},
        Some(Value::String(mode)) if mode == "auto" => {},
        Some(Value::String(mode)) if mode == "none" => tools.clear(),
        Some(_) => anyhow::bail!("Uzu supports only auto or none for tool_choice"),
    }
    if !tools.is_empty() {
        let position =
            messages.iter().position(|message| message.role != (ChatRole::System {})).unwrap_or(messages.len());
        messages.insert(
            position,
            ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
                name: "functions".to_string(),
                description: None,
                tools,
            }]),
        );
    }

    Ok(messages)
}

fn rate(
    count: usize,
    duration: f64,
) -> f64 {
    if duration > 0.0 {
        count as f64 / duration
    } else {
        0.0
    }
}
