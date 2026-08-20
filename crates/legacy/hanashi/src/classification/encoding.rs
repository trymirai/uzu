use std::{collections::HashMap, string::ToString, sync::Arc};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use shoji::types::session::classification::{
    ChatTokenCodecConfig, ClassificationMessage, ClassificationRole, TokenCodecConfig,
};
use tokenizers::Tokenizer;

use crate::chat::{
    hanashi::{
        Error,
        renderer::{
            RAISE_EXCEPTION_FUNCTION_NAME, STRFTIME_NOW_FUNCTION_NAME, TEMPLATE_NAME, TOJSON_FILTER_NAME,
            raise_exception, to_json,
        },
    },
    strftime_now,
};

pub struct ClassificationEncoding {
    tokenizer: Arc<Tokenizer>,
    config: TokenCodecConfig,
}

impl ClassificationEncoding {
    pub fn new(
        config: TokenCodecConfig,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            tokenizer,
            config,
        }
    }

    pub fn encode(
        &mut self,
        input: &[ClassificationMessage],
    ) -> Result<Vec<u32>, Error> {
        match &self.config {
            TokenCodecConfig::Chat(config) => Self::encode_chat(&self.tokenizer, config, input),
            TokenCodecConfig::RawText => Self::encode_raw_text(&self.tokenizer, input),
        }
    }

    fn encode_chat(
        tokenizer: &Tokenizer,
        config: &ChatTokenCodecConfig,
        input: &[ClassificationMessage],
    ) -> Result<Vec<u32>, Error> {
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function(STRFTIME_NOW_FUNCTION_NAME, strftime_now);
        environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
        environment.add_filter(TOJSON_FILTER_NAME, to_json);
        environment
            .add_template(TEMPLATE_NAME, config.prompt_template.as_str())
            .map_err(|_| Error::UnableToEncodeText)?;

        let messages: Vec<HashMap<String, String>> = input
            .iter()
            .map(|message| {
                let role = match message.role {
                    ClassificationRole::Assistant => config.assistant_role_name.as_str(),
                    ClassificationRole::User => config.user_role_name.as_str(),
                };

                let mut map: HashMap<String, String> = HashMap::new();
                map.insert("role".to_string(), role.to_string());
                map.insert("content".to_string(), message.content.clone());
                map
            })
            .collect();

        let rendered_message = environment
            .get_template(TEMPLATE_NAME)
            .unwrap()
            .render(context!(
                messages => messages,
                add_generation_prompt => false,
                bos_token => config.bos_token.clone(),
                eos_token => config.eos_token.clone(),
                enable_thinking => false,
            ))
            .map_err(|_| Error::UnableToEncodeText)?;

        let tokens = tokenizer.encode(rendered_message, false).map_err(|_| Error::UnableToEncodeText)?;

        Ok(tokens.get_ids().to_vec())
    }

    fn encode_raw_text(
        tokenizer: &Tokenizer,
        input: &[ClassificationMessage],
    ) -> Result<Vec<u32>, Error> {
        let text = input.iter().map(|message| message.content.as_str()).collect::<String>();
        let tokens = tokenizer.encode(text, false).map_err(|_| Error::UnableToEncodeText)?;

        Ok(tokens.get_ids().to_vec())
    }
}
