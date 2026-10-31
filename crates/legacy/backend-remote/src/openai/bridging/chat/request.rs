use async_openai::types::chat::{
    ChatCompletionStreamOptions, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, ToolChoiceOptions,
};
use shoji::types::{
    basic::{SamplingMethod, SamplingPolicy},
    session::chat::{ChatContentBlock, ChatMessage, ChatMessageList, ChatReplyConfig, ChatRole},
};

use crate::openai::{
    Error,
    bridging::{chat, reasoning_effort},
};

pub fn build(
    model: &str,
    config: &ChatReplyConfig,
    messages: Vec<ChatMessage>,
) -> Result<CreateChatCompletionRequest, Error> {
    let completion_messages = messages
        .iter()
        .filter(|message| !is_tool_definition_message(message))
        .map(chat::message::build)
        .collect::<Result<Vec<_>, _>>()?;

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
        request.max_completion_tokens = Some(token_limit);
    }
    match &config.sampling_policy {
        SamplingPolicy::Default {} => {},
        SamplingPolicy::Custom {
            method,
        } => match method {
            SamplingMethod::Greedy {} => {
                request.temperature = Some(0.0);
            },
            SamplingMethod::Stochastic {
                temperature,
                top_k: _,
                top_p,
                min_p: _,
                repetition_penalty: _,
                suffix_repetition_length: _,
            } => {
                request.temperature = temperature.map(|v| v as f32);
                request.top_p = top_p.map(|v| v as f32);
            },
        },
    }

    Ok(request)
}

fn is_tool_definition_message(message: &ChatMessage) -> bool {
    message.role == ChatRole::Developer {}
        && !message.content.is_empty()
        && message.content.iter().all(|content| matches!(content, ChatContentBlock::Tools { .. }))
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CreateChatCompletionRequest;
    use shoji::types::{basic::SamplingMethod, session::chat::ChatReplyConfig};

    use super::build;

    /// Builds the request through the crate's public builder, so the temperature
    /// put on the wire stays pinned to the sampling policy the caller chose.
    fn request_for(config: &ChatReplyConfig) -> CreateChatCompletionRequest {
        build("model", config, Vec::new()).expect("the request builds")
    }

    #[test]
    fn explicit_greedy_sends_temperature_zero() {
        let request = request_for(&ChatReplyConfig::default().with_sampling_method(SamplingMethod::Greedy {}));

        assert_eq!(request.temperature, Some(0.0));
    }

    #[test]
    fn default_policy_sends_no_temperature() {
        let request = request_for(&ChatReplyConfig::default());

        assert_eq!(request.temperature, None);
    }

    #[test]
    fn stochastic_forwards_temperature_unchanged() {
        let request = request_for(&ChatReplyConfig::default().with_sampling_method(SamplingMethod::Stochastic {
            temperature: Some(0.5),
            top_k: None,
            top_p: Some(0.75),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }));

        assert_eq!(request.temperature, Some(0.5));
        assert_eq!(request.top_p, Some(0.75));
    }

    #[test]
    fn stochastic_without_temperature_sends_none() {
        let request = request_for(&ChatReplyConfig::default().with_sampling_method(SamplingMethod::Stochastic {
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }));

        assert_eq!(request.temperature, None);
    }
}
