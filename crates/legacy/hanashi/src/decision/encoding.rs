use std::{collections::HashSet, sync::Arc};

use minijinja::{Environment, Error, ErrorKind, Value, context, syntax::SyntaxConfig};
use minijinja_contrib::pycompat::unknown_method_callback;
use shoji::types::session::{
    classification::{ChatTokenCodecConfig, TokenCodecConfig},
    decision::DecisionQuestion,
};
use tokenizers::Tokenizer;

use crate::{
    chat::{
        hanashi::renderer::{
            RAISE_EXCEPTION_FUNCTION_NAME, STRFTIME_NOW_FUNCTION_NAME, TEMPLATE_NAME, TOJSON_FILTER_NAME,
            raise_exception, to_json,
        },
        strftime_now,
    },
    decision::error::DecisionEncodingError,
};

pub struct DecisionEncoding {
    tokenizer: Arc<Tokenizer>,
    config: ChatTokenCodecConfig,
}

impl DecisionEncoding {
    pub fn new(
        config: TokenCodecConfig,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self, DecisionEncodingError> {
        let TokenCodecConfig::Chat(config) = config else {
            return Err(DecisionEncodingError::UnsupportedTokenCodec);
        };
        Ok(Self {
            tokenizer,
            config,
        })
    }

    pub fn encode(
        &self,
        state: &str,
        question: &DecisionQuestion,
    ) -> Result<Vec<u32>, DecisionEncodingError> {
        let request = context!(state => state, question => Self::question(question)?);

        // Whitespace handling of transformers' `apply_chat_template`, which the d3 prompts were rendered with.
        let mut environment = Environment::new();
        environment.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|error| DecisionEncodingError::UnableToRenderPrompt(error.to_string()))?,
        );
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function(STRFTIME_NOW_FUNCTION_NAME, strftime_now);
        environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
        environment.add_filter(TOJSON_FILTER_NAME, to_json);
        environment
            .add_template(TEMPLATE_NAME, self.config.prompt_template.as_str())
            .map_err(|error| DecisionEncodingError::UnableToRenderPrompt(error.to_string()))?;

        let messages = [context!(role => &self.config.user_role_name, content => request)];
        let rendered_prompt = environment
            .get_template(TEMPLATE_NAME)
            .unwrap()
            .render(context!(messages => messages))
            .map_err(Self::render_error)?;

        let tokens =
            self.tokenizer.encode(rendered_prompt, false).map_err(|_| DecisionEncodingError::UnableToEncodeText)?;

        Ok(tokens.get_ids().to_vec())
    }

    fn render_error(error: Error) -> DecisionEncodingError {
        match (error.kind(), error.detail()) {
            (ErrorKind::InvalidOperation, Some(message)) => DecisionEncodingError::InvalidQuestion(message.to_string()),
            _ => DecisionEncodingError::UnableToRenderPrompt(error.to_string()),
        }
    }

    fn question(question: &DecisionQuestion) -> Result<Value, DecisionEncodingError> {
        Ok(match question {
            DecisionQuestion::Choice {
                instructions,
                options,
            } => {
                // Choice criteria are a mapping, so a repeated key would silently drop an option.
                let mut keys = HashSet::new();
                if let Some(option) = options.iter().find(|option| !keys.insert(option.key.as_str())) {
                    return Err(DecisionEncodingError::DuplicateOption(option.key.clone()));
                }
                let criteria = Value::from_pairs(
                    options.iter().map(|option| (option.key.as_str(), Value::from(&option.description))),
                );
                context!(type => "choice", instructions => instructions, criteria => criteria)
            },
            DecisionQuestion::Score {
                instructions,
                levels,
            } => context!(type => "score", instructions => instructions, criteria => levels),
            DecisionQuestion::YesNo {
                instructions,
                no_description,
                yes_description,
            } => {
                let criteria =
                    Value::from_pairs([("false", Value::from(no_description)), ("true", Value::from(yes_description))]);
                context!(type => "noul", instructions => instructions, criteria => criteria)
            },
        })
    }
}
