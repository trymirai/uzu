use async_openai::types::responses::{
    CreateResponse, InputParam, Reasoning, ReasoningSummary, Tool as ResponseTool, ToolChoiceOptions, ToolChoiceParam,
};
use shoji::{
    traits::backend::chat::StreamConfig,
    types::{
        encoding::{Message, MessageList},
        session::chat::{SamplingMethod, SamplingPolicy},
    },
};

use crate::openai::{
    Error,
    bridging::{reasoning_effort, responses},
};

pub fn build(
    model: &str,
    config: &StreamConfig,
    messages: Vec<Message>,
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
    });

    let (temperature, top_p) = match &config.sampling_policy {
        SamplingPolicy::Default {} => (None, None),
        SamplingPolicy::Custom {
            method,
        } => match method {
            SamplingMethod::Greedy {} => (None, None),
            SamplingMethod::Stochastic {
                temperature,
                top_k: _,
                top_p,
                min_p: _,
            } => (temperature.map(|value| value as f32), top_p.map(|value| value as f32)),
        },
    };

    let mut request = CreateResponse {
        input: InputParam::Items(input_items),
        model: Some(model.to_string()),
        stream: Some(true),
        max_output_tokens: config.token_limit.map(|value| value as u32),
        temperature,
        top_p,
        reasoning,
        ..Default::default()
    };
    if !tools.is_empty() {
        request.tools = Some(tools);
        request.tool_choice = Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto));
    }

    Ok(request)
}
