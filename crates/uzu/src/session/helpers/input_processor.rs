use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;

use crate::{
    config::MessageProcessorConfig,
    session::{
        parameter::ConfigResolvableValue,
        types::{Error, Input},
    },
};

pub trait InputProcessor: Send + Sync {
    fn process(
        &self,
        input: &Input,
        enable_thinking: bool,
    ) -> Result<String, Error>;
}

pub struct InputProcessorDefault {
    message_processing_config: MessageProcessorConfig,
}

impl InputProcessorDefault {
    pub fn new(message_processing_config: MessageProcessorConfig) -> Self {
        Self {
            message_processing_config,
        }
    }
}

impl InputProcessor for InputProcessorDefault {
    fn process(
        &self,
        input: &Input,
        enable_thinking: bool,
    ) -> Result<String, Error> {
        let messages = input
            .get_messages()
            .into_iter()
            .map(|message| message.resolve(&self.message_processing_config))
            .collect::<Vec<_>>();
        let template = self.message_processing_config.prompt_template.clone();
        let bos_token = self.message_processing_config.bos_token.clone();

        let template_name = "chat_template";
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment
            .add_template(template_name, template.as_str())
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;
        let template = environment
            .get_template(template_name)
            .map_err(|_| Error::UnableToLoadPromptTemplate)?;

        let result = template.render(context!(
            messages => messages,
            add_generation_prompt => true,
            bos_token => bos_token,
            enable_thinking => enable_thinking
        ));

        // For simple text inputs on base models, fall back to BOS + text if template fails
        match result {
            Ok(rendered) => Ok(rendered),
            Err(e) => {
                eprintln!(
                    "[Template Error] Failed to render prompt template: {:?}",
                    e
                );
                eprintln!(
                    "[Template Error] Falling back to simple format for single-message input"
                );
                // If it's a single user message (typical for Input::Text), use simple format
                if messages.len() == 1 {
                    if let Some(content_str) = messages[0].get("content") {
                        let prefix = bos_token
                            .as_ref()
                            .map(|s| s.as_str())
                            .unwrap_or("");
                        Ok(format!("{}{}", prefix, content_str))
                    } else {
                        Err(Error::UnableToRenderPromptTemplate)
                    }
                } else {
                    Err(Error::UnableToRenderPromptTemplate)
                }
            },
        }
    }
}
