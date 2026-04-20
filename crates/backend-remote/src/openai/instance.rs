use std::{pin::Pin, sync::Arc};

use futures::{FutureExt, Stream, StreamExt, stream};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{
        ChatCompletionMessage, Content as OpenAIContent, MessageRole, ReasoningEffort as OpenAIReasoningEffort, Tool,
        ToolCall, ToolCallFunction, ToolChoiceType, ToolType,
        chat_completion_stream::{ChatCompletionStreamRequest, ChatCompletionStreamResponse},
    },
    types::{Function, FunctionParameters, JSONSchemaType},
};
use shoji::{
    traits::{
        Instance as InstanceTrait, State as StateTrait,
        backend::{
            Error as BackendError,
            chat::StreamConfig,
            chat_message::{Output, StreamInput, StreamOutput},
        },
    },
    types::{
        basic::Value,
        encoding::{Message, MessageList, ReasoningEffort, Role, ToolDescription},
        session::chat::{SamplingMethod, SamplingPolicy},
    },
};
use tokio_util::sync::CancellationToken;

use crate::openai::Error;

#[derive(Debug, Clone)]
pub struct State;

impl StateTrait for State {
    fn clone_boxed(&self) -> Box<dyn StateTrait> {
        Box::new(self.clone())
    }
}

pub struct Instance {
    client: Arc<OpenAIClient>,
    model_identifier: String,
}

impl Instance {
    pub fn new(
        client: Arc<OpenAIClient>,
        model_identifier: String,
    ) -> Self {
        Self {
            client,
            model_identifier,
        }
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = StreamConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        Box::pin(async move { Ok(Box::new(State) as Box<dyn StateTrait>) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        config: Self::StreamConfig,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        let client = self.client.clone();
        let request = build_request(&self.model_identifier, &config, input.clone());

        let stream = async move {
            let request = match request {
                Ok(request) => request,
                Err(error) => {
                    return stream::once(async move { Err(Box::new(error) as BackendError) }).boxed();
                },
            };
            match client.chat_completion_stream(request).await {
                Ok(completion_stream) => completion_stream
                    .filter_map(|response| {
                        let chunk = match process_chunk(response) {
                            Ok(Some(output)) => Some(Ok(output)),
                            Ok(None) => None,
                            Err(error) => Some(Err(Box::new(error) as BackendError)),
                        };
                        futures::future::ready(chunk)
                    })
                    .boxed(),
                Err(error) => stream::once(async move {
                    Err(Box::new(Error::Network {
                        message: error.to_string(),
                    }) as BackendError)
                })
                .boxed(),
            }
        };

        stream.flatten_stream().boxed()
    }
}

fn build_request(
    model_identifier: &str,
    config: &StreamConfig,
    messages: Vec<Message>,
) -> Result<ChatCompletionStreamRequest, Error> {
    let completion_messages = messages
        .iter()
        .map(|message| -> Result<ChatCompletionMessage, Error> {
            let (role, content, tool_calls, tool_call_id) = match message.role {
                Role::System {} => (MessageRole::system, message.text(), None, None),
                Role::User {} => (MessageRole::user, message.text(), None, None),
                Role::Assistant {} => (MessageRole::assistant, message.text(), Some(message.tool_calls()), None),
                Role::Tool {} => {
                    let tool_call_results = message.tool_call_results();
                    if tool_call_results.len() != 1 {
                        return Err(Error::ToolCallResultRequired);
                    }
                    let (tool_call_id, value) = tool_call_results.first().ok_or(Error::ToolCallResultRequired)?;
                    let content = serde_json::to_string(&value).map_err(|error| Error::Serialization {
                        message: error.to_string(),
                    })?;
                    (MessageRole::function, content, None, tool_call_id.clone())
                },
                Role::Developer {}
                | Role::Custom {
                    ..
                } => return Err(Error::UnsupportedRole),
            };
            let tool_calls: Option<Vec<ToolCall>> = tool_calls
                .map(|calls| {
                    calls
                        .into_iter()
                        .map(|tool_call| -> Result<ToolCall, Error> {
                            Ok(ToolCall {
                                id: tool_call.name.clone(),
                                r#type: "function".to_string(),
                                function: ToolCallFunction {
                                    name: Some(tool_call.name.clone()),
                                    arguments: Some(serde_json::to_string(&tool_call.arguments).map_err(|error| {
                                        Error::Serialization {
                                            message: error.to_string(),
                                        }
                                    })?),
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            Ok(ChatCompletionMessage {
                role,
                content: OpenAIContent::Text(content),
                name: tool_call_id.clone(),
                tool_calls,
                tool_call_id: tool_call_id.clone(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let tools: Vec<Tool> = messages
        .tool_namespaces()
        .into_iter()
        .flat_map(|namespace| namespace.tools.into_iter())
        .map(|description| -> Result<Tool, Error> {
            let ToolDescription::Function {
                function,
            } = description;
            let parameters = match function.parameters {
                Some(value) => {
                    serde_json::from_str::<FunctionParameters>(&value.json).map_err(|error| Error::Serialization {
                        message: error.to_string(),
                    })?
                },
                None => FunctionParameters {
                    schema_type: JSONSchemaType::Object,
                    properties: None,
                    required: None,
                },
            };
            Ok(Tool {
                r#type: ToolType::Function,
                function: Function {
                    name: function.name,
                    description: Some(function.description),
                    parameters,
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let reasoning_effort = messages.reasoning_effort().map(|effort| match effort {
        ReasoningEffort::Disabled => OpenAIReasoningEffort::None,
        ReasoningEffort::Default => OpenAIReasoningEffort::Medium,
        ReasoningEffort::Low => OpenAIReasoningEffort::Low,
        ReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
        ReasoningEffort::High => OpenAIReasoningEffort::High,
    });

    let mut request = ChatCompletionStreamRequest::new(model_identifier.to_string(), completion_messages);
    if tools.len() > 0 {
        request.tools = Some(tools);
        request.tool_choice = Some(ToolChoiceType::Auto);
    }
    if let Some(effort) = reasoning_effort {
        request.reasoning_effort = Some(effort);
    }
    if let Some(token_limit) = config.token_limit {
        request.max_tokens = Some(token_limit as i64);
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
                request.temperature = *temperature;
                request.top_p = *top_p;
            },
        },
    }

    Ok(request)
}

fn process_chunk(response: ChatCompletionStreamResponse) -> Result<Option<Output>, Error> {
    match response {
        ChatCompletionStreamResponse::Content(text) => Ok(Some(Output::Content(text))),
        ChatCompletionStreamResponse::Reasoning(text) => Ok(Some(Output::Reasoning(text))),
        ChatCompletionStreamResponse::ToolCall(calls) => {
            let values = calls
                .into_iter()
                .map(|call| {
                    serde_json::to_value(call).map(Value::from).map_err(|error| Error::Serialization {
                        message: error.to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(Output::ToolCalls(values)))
        },
        ChatCompletionStreamResponse::Done => Ok(None),
    }
}
