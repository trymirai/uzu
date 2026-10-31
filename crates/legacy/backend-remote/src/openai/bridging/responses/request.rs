use async_openai::types::responses::{
    CreateResponse, InputParam, Reasoning, ReasoningSummary, Tool as ResponseTool, ToolChoiceOptions, ToolChoiceParam,
};
use shoji::types::{
    basic::{SamplingMethod, SamplingPolicy},
    session::chat::{ChatMessage, ChatMessageList, ChatReplyConfig},
};

use crate::openai::{
    Error,
    bridging::{reasoning_effort, responses},
};

pub fn build(
    model: &str,
    config: &ChatReplyConfig,
    messages: Vec<ChatMessage>,
) -> Result<CreateResponse, Error> {
    let input_items = messages
        .iter()
        .map(responses::message::build)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let tools: Vec<ResponseTool> = messages
        .tool_namespaces()
        .into_iter()
        .flat_map(|namespace| namespace.tools.into_iter())
        .map(responses::tool::build)
        .collect::<Result<Vec<_>, _>>()?;

    let reasoning = messages.reasoning_effort().map(|effort| Reasoning {
        effort: Some(reasoning_effort::build(effort)),
        summary: Some(ReasoningSummary::Auto),
        ..Default::default()
    });

    let (temperature, top_p) = match &config.sampling_policy {
        SamplingPolicy::Default {} => (None, None),
        SamplingPolicy::Custom {
            method,
        } => match method {
            SamplingMethod::Greedy {} => (Some(0.0), None),
            SamplingMethod::Stochastic {
                temperature,
                top_k: _,
                top_p,
                min_p: _,
                repetition_penalty: _,
                suffix_repetition_length: _,
            } => (temperature.map(|value| value as f32), top_p.map(|value| value as f32)),
        },
    };

    let mut request = CreateResponse {
        input: InputParam::Items(input_items),
        model: Some(model.to_string()),
        stream: Some(true),
        max_output_tokens: config.token_limit,
        temperature,
        top_p,
        reasoning,
        ..Default::default()
    };
    if !tools.is_empty() {
        request.tools = Some(tools);
        request.tool_choice = Some(ToolChoiceParam::Option(ToolChoiceOptions::Auto));
    }

    Ok(request)
}

#[cfg(test)]
mod tests {
    use async_openai::types::responses::CreateResponse;
    use shoji::types::{basic::SamplingMethod, session::chat::ChatReplyConfig};

    use super::build;

    /// Builds the request through the crate's public builder, so the temperature
    /// put on the wire stays pinned to the sampling policy the caller chose.
    fn request_for(config: &ChatReplyConfig) -> CreateResponse {
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
