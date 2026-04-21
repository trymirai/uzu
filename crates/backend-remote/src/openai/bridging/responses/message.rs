use async_openai::types::responses::{
    EasyInputContent, EasyInputMessage, FunctionCallOutput, FunctionCallOutputItemParam, FunctionToolCall, InputItem,
    Item, MessageType, Role as ResponseRole,
};
use shoji::types::session::chat::{Message, Role};

use crate::openai::Error;

pub fn build(message: &Message) -> Result<Vec<InputItem>, Error> {
    let mut items = Vec::new();
    match message.role {
        Role::System {} | Role::Developer {} | Role::User {} | Role::Assistant {} => {
            let role = match message.role {
                Role::System {} => ResponseRole::System,
                Role::Developer {} => ResponseRole::Developer,
                Role::User {} => ResponseRole::User,
                Role::Assistant {} => ResponseRole::Assistant,
                _ => return Err(Error::UnsupportedRole),
            };
            let text = message.text();
            if let Some(text) = text {
                items.push(InputItem::EasyMessage(EasyInputMessage {
                    r#type: MessageType::default(),
                    role,
                    content: EasyInputContent::Text(text),
                    phase: None,
                }));
            }
            if matches!(message.role, Role::Assistant {}) {
                for tool_call in message.tool_calls() {
                    items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                        arguments: tool_call.arguments.json.clone(),
                        call_id: tool_call.identifier.unwrap_or_default(),
                        namespace: None,
                        name: tool_call.name,
                        id: None,
                        status: None,
                    })));
                }
            }
        },
        Role::Tool {} => {
            for (call_id, _name, value) in message.tool_call_results() {
                let content = serde_json::to_string(&value).map_err(|error| Error::Serialization {
                    message: error.to_string(),
                })?;
                items.push(InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                    call_id: call_id.unwrap_or_default(),
                    output: FunctionCallOutput::Text(content),
                    id: None,
                    status: None,
                })));
            }
        },
        Role::Custom {
            ..
        } => return Err(Error::UnsupportedRole),
    }
    Ok(items)
}
