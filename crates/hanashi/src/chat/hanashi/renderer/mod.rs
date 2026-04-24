mod config;
mod error;
mod functions;

pub use config::{JinjaFunction, RendererConfig};
pub use error::Error;
use functions::{raise_exception, strftime_now, to_json};
use indexmap::IndexMap;
use minijinja::Environment;
use minijinja_contrib::pycompat::unknown_method_callback;
use shoji::types::{basic::Token, session::chat::ChatMessage};

use crate::chat::hanashi::messages::rendered::Message as RenderedMessage;

static TEMPLATE_NAME: &str = "chat_template";

pub struct Renderer {
    config: RendererConfig,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
        }
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        should_add_preamble: bool,
        bos_token: Option<Token>,
        eos_token: Option<Token>,
        additional_context: Option<&IndexMap<String, serde_json::Value>>,
    ) -> Result<String, Error> {
        let mut environment = Environment::new();
        environment.set_unknown_method_callback(unknown_method_callback);
        for function in &self.config.jinja.required_functions {
            environment.add_function(
                function.to_string(),
                match function {
                    JinjaFunction::StrftimeNow => strftime_now,
                },
            );
        }
        environment.add_function("raise_exception", raise_exception);
        environment.add_filter("tojson", to_json);
        environment
            .add_template(TEMPLATE_NAME, self.config.jinja.template.as_str())
            .map_err(|_| Error::InvalidTemplate)?;

        let mut jinja_context = IndexMap::<String, minijinja::Value>::new();
        let mut jinja_messages = Vec::new();
        for message in messages {
            let rendered = RenderedMessage::from_message(message, &self.config.canonization, &self.config.rendering)?;
            for (key, value) in &rendered.context {
                insert_into_context(&mut jinja_context, key.clone(), minijinja::Value::from_serialize(value))?;
            }
            if !rendered.message.is_empty() {
                jinja_messages.push(minijinja::Value::from_serialize(&rendered.message));
            }
        }
        insert_into_context(&mut jinja_context, "messages".to_string(), minijinja::Value::from(jinja_messages))?;
        insert_into_context(
            &mut jinja_context,
            self.config.jinja.preamble_control_key.clone(),
            minijinja::Value::from(should_add_preamble),
        )?;
        if let Some(bos_token_key) = &self.config.jinja.bos_token_key {
            let bos_token = bos_token.ok_or(Error::BosTokenRequired)?;
            insert_into_context(
                &mut jinja_context,
                bos_token_key.clone(),
                minijinja::Value::from(bos_token.value.clone()),
            )?;
        }
        if let Some(eos_token_key) = &self.config.jinja.eos_token_key {
            let eos_token = eos_token.ok_or(Error::EosTokenRequired)?;
            insert_into_context(
                &mut jinja_context,
                eos_token_key.clone(),
                minijinja::Value::from(eos_token.value.clone()),
            )?;
        }
        if let Some(additional_context) = additional_context {
            for (key, value) in additional_context {
                insert_into_context(&mut jinja_context, key.clone(), minijinja::Value::from_serialize(value))?;
            }
        }

        let template = environment.get_template(TEMPLATE_NAME).map_err(|_| Error::InvalidTemplate)?;
        let result = template
            .render(minijinja::Value::from_serialize(&jinja_context))
            .map_err(|error| Error::RenderFailed {
                reason: error.to_string(),
            })?
            .trim_start()
            .to_string();
        Ok(result)
    }
}

fn insert_into_context(
    context: &mut IndexMap<String, minijinja::Value>,
    key: String,
    value: minijinja::Value,
) -> Result<(), Error> {
    if context.contains_key(&key) {
        return Err(Error::DuplicateContextKey {
            key,
        });
    }
    context.insert(key, value);
    Ok(())
}
