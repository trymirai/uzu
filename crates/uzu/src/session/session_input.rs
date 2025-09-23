use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use super::session_message::{SessionMessage, SessionMessageRole};
use crate::config::MessageProcessorConfig;

#[derive(Debug)]
pub enum SessionInput {
    Text(String),
    Messages(Vec<SessionMessage>),
}

impl SessionInput {
    fn get_messages(&self) -> Vec<SessionMessage> {
        match self {
            SessionInput::Text(content) => vec![SessionMessage {
                role: SessionMessageRole::User,
                content: content.clone(),
            }],
            SessionInput::Messages(messages) => messages.clone(),
        }
    }
}

pub trait SessionInputProcessor: Send + Sync {
    fn process(
        &self,
        input: &SessionInput,
        enable_thinking: bool,
    ) -> String;
}

pub struct SessionInputProcessorDefault {
    message_processing_config: MessageProcessorConfig,
}

impl SessionInputProcessorDefault {
    pub fn new(message_processing_config: MessageProcessorConfig) -> Self {
        Self {
            message_processing_config,
        }
    }
}

impl SessionInputProcessor for SessionInputProcessorDefault {
    fn process(
        &self,
        input: &SessionInput,
        enable_thinking: bool,
    ) -> String {
        let messages = input.get_messages();
        let template = self.message_processing_config.prompt_template.clone();
        let bos_token = self.message_processing_config.bos_token.clone();

        let template_name = "chat_template";
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_template(template_name, template.as_str()).unwrap();
        let template = environment.get_template(template_name).unwrap();
        let result = template
            .render(context!(
                messages => messages,
                add_generation_prompt => true,
                bos_token => bos_token,
                enable_thinking => enable_thinking
            ))
            .unwrap();
        result
    }
}
