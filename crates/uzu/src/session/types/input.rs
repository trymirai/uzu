use crate::session::types::{Message, Role};

#[derive(Debug)]
pub enum Input {
    Text(String),
    Messages(Vec<Message>),
}

impl Input {
    pub fn get_messages(&self) -> Vec<Message> {
        match self {
            Input::Text(content) => vec![Message {
                role: Role::User,
                content: content.clone(),
                reasoning_content: None,
            }],
            Input::Messages(messages) => messages.clone(),
        }
    }
}
