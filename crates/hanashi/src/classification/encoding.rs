use std::{collections::HashMap, fs::File, path::PathBuf, string::ToString};

use minijinja::{Environment, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde_json::Value;
use shoji::types::session::classification::{ClassificationMessage, ClassificationRole};
use tokenizers::Tokenizer;

use crate::{
    chat::{
        TokenizerLocation,
        hanashi::{
            Error,
            renderer::{TEMPLATE_NAME, raise_exception, to_json},
        },
        strftime_now,
    },
    classification::config::{ChatTokenCodecConfig, TokenCodecConfig},
    util::tokenizer::load_tokenizer,
};

const CONFIG_FILE_NAME: &str = "config.json";

pub struct ClassificationEncoding {
    tokenizer: Tokenizer,
    config: TokenCodecConfig,
}

impl ClassificationEncoding {
    pub fn new(model_dir: &str) -> Result<Self, Error> {
        let model_path = PathBuf::from(model_dir);
        let config_file = File::open(model_path.join(CONFIG_FILE_NAME))
            .map_err(|_| Error::ConfigNotFound(CONFIG_FILE_NAME.to_string()))?;
        let config = serde_json::from_reader::<File, Value>(config_file)
            .map_err(|_| Error::InvalidConfig(CONFIG_FILE_NAME.to_string()))?;
        let codec_config =
            config.get("token_codec_config").ok_or_else(|| Error::InvalidConfig(CONFIG_FILE_NAME.to_string()))?;
        let config = TokenCodecConfig::from_value(codec_config.clone())
            .map_err(|_| Error::InvalidConfig(CONFIG_FILE_NAME.to_string()))?;

        let tokenizer_location = TokenizerLocation::Directory {
            path: model_dir.to_string(),
            name: None,
        };
        let tokenizer = load_tokenizer(&tokenizer_location).map_err(|_| Error::UnableToLoadTokenizer)?;

        Ok(Self {
            tokenizer,
            config,
        })
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
        environment.add_function("strftime_now", strftime_now);
        environment.add_function("raise_exception", raise_exception);
        environment.add_filter("tojson", to_json);
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
