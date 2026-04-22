use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    FunctionCall,
};
use shoji::types::session::chat::{ChatMessage, ChatRole};

use crate::openai::Error;

enum ResolvedRole {
    System,
    User,
    Assistant,
    Tool,
}

pub fn build(message: &ChatMessage) -> Result<ChatCompletionRequestMessage, Error> {
    let (role, content, tool_calls, tool_call_id) = match message.role {
        ChatRole::System {} => (ResolvedRole::System, message.text(), None, None),
        ChatRole::User {} => (ResolvedRole::User, message.text(), None, None),
        ChatRole::Assistant {} => (ResolvedRole::Assistant, message.text(), Some(message.tool_calls()), None),
        ChatRole::Tool {} => {
            let tool_call_results = message.tool_call_results();
            if tool_call_results.len() != 1 {
                return Err(Error::ToolCallResultRequired);
            }
            let (tool_call_id, _, value) = tool_call_results.first().ok_or(Error::ToolCallResultRequired)?;
            let content = serde_json::to_string(&value).map_err(|error| Error::Serialization {
                message: error.to_string(),
            })?;
            (ResolvedRole::Tool, Some(content), None, tool_call_id.clone())
        },
        ChatRole::Developer {}
        | ChatRole::Custom {
            ..
        } => return Err(Error::UnsupportedRole),
    };

    let tool_calls: Option<Vec<ChatCompletionMessageToolCalls>> = tool_calls.map(|calls| {
        calls
            .into_iter()
            .map(|tool_call| {
                ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                    id: tool_call.identifier.unwrap_or_default(),
                    function: FunctionCall {
                        name: tool_call.name.clone(),
                        arguments: tool_call.arguments.json.clone(),
                    },
                })
            })
            .collect()
    });

    Ok(match role {
        ResolvedRole::System => ChatCompletionRequestMessage::System({
            let content = content.ok_or(Error::ContentRequired)?;
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(content),
                name: None,
            }
        }),
        ResolvedRole::User => ChatCompletionRequestMessage::User({
            let content = content.ok_or(Error::ContentRequired)?;
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(content),
                name: None,
            }
        }),
        ResolvedRole::Assistant => {
            let text_content = if let Some(content) = content {
                Some(ChatCompletionRequestAssistantMessageContent::Text(content))
            } else {
                None
            };
            #[allow(deprecated)]
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: text_content,
                refusal: None,
                name: None,
                audio: None,
                tool_calls,
                function_call: None,
            })
        },
        ResolvedRole::Tool => ChatCompletionRequestMessage::Tool({
            let content = content.ok_or(Error::ContentRequired)?;
            ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text(content),
                tool_call_id: tool_call_id.unwrap_or_default(),
            }
        }),
    })
}
