use async_openai::types::chat::{
    ChatCompletionStreamOptions, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, ToolChoiceOptions,
};
use shoji::types::{
    basic::{SamplingMethod, SamplingPolicy},
    session::chat::{ChatMessage, ChatStreamConfig, MessageList},
};

use crate::openai::{
    Error,
    bridging::{chat, reasoning_effort},
};

pub fn build(
    model: &str,
    config: &ChatStreamConfig,
    messages: Vec<ChatMessage>,
) -> Result<CreateChatCompletionRequest, Error> {
    let completion_messages = messages.iter().map(chat::message::build).collect::<Result<Vec<_>, _>>()?;

    let tools = messages
        .tool_namespaces()
        .into_iter()
        .flat_map(|namespace| namespace.tools.into_iter())
        .map(chat::tool::build)
        .collect::<Result<Vec<_>, _>>()?;

    let reasoning_effort = messages.reasoning_effort().map(reasoning_effort::build);

    let mut request = CreateChatCompletionRequest {
        messages: completion_messages,
        model: model.to_string(),
        stream: Some(true),
        stream_options: Some(ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        }),
        ..Default::default()
    };
    if !tools.is_empty() {
        request.tools = Some(tools);
        request.tool_choice = Some(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto));
    }
    if let Some(effort) = reasoning_effort {
        request.reasoning_effort = Some(effort);
    }
    if let Some(token_limit) = config.token_limit {
        request.max_completion_tokens = Some(token_limit as u32);
    }
    match &config.sampling_policy {
        SamplingPolicy::Default {} => {},
        SamplingPolicy::Custom {
            method,
        } => match method {
            SamplingMethod::Greedy {} => {},
            SamplingMethod::Stochastic {
                temperature,
                top_k: _,
                top_p,
                min_p: _,
            } => {
                request.temperature = temperature.map(|v| v as f32);
                request.top_p = top_p.map(|v| v as f32);
            },
        },
    }

    Ok(request)
}
