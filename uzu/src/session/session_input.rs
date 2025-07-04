use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use super::{
    session_message::{SessionMessage, SessionMessageRole},
    tokenizer_config::TokenizerConfig,
};

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
    ) -> String;
}

pub struct SessionInputProcessorDefault {
    tokenizer_config: TokenizerConfig,
}

impl SessionInputProcessorDefault {
    pub fn new(tokenizer_config: TokenizerConfig) -> Self {
        Self {
            tokenizer_config,
        }
    }
}

impl SessionInputProcessor for SessionInputProcessorDefault {
    fn process(
        &self,
        input: &SessionInput,
    ) -> String {
        let messages = input.get_messages();
        let template = self.tokenizer_config.chat_template.clone().unwrap();

        let template_name = "chat_template";
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_template(template_name, template.as_str()).unwrap();
        let template = environment.get_template(template_name).unwrap();
        let result = template
            .render(
                context!(messages => messages, add_generation_prompt => true),
            )
            .unwrap();
        result
    }
}
