mod config;
mod error;
mod functions;

pub use config::{JinjaFunction, RendererConfig};
pub use error::Error;
pub use functions::{raise_exception, strftime_now, to_json};
use indexmap::IndexMap;
use minijinja::Environment;
use minijinja_contrib::pycompat::unknown_method_callback;
use shoji::types::{
    basic::{ReasoningEffort, Token},
    session::chat::ChatMessage,
};

use crate::chat::hanashi::messages::rendered::{FieldConfig, Message as RenderedMessage};

pub static TEMPLATE_NAME: &str = "chat_template";
pub static STRFTIME_NOW_FUNCTION_NAME: &str = "strftime_now";
pub static RAISE_EXCEPTION_FUNCTION_NAME: &str = "raise_exception";
pub static TOJSON_FILTER_NAME: &str = "tojson";

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
        environment.add_function(RAISE_EXCEPTION_FUNCTION_NAME, raise_exception);
        environment.add_filter(TOJSON_FILTER_NAME, to_json);
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

        // Context fields driven by an optional control block (e.g. `reasoning_effort`)
        // fall back to the model's declared default when no message carries the block:
        // absence must behave exactly like `ReasoningEffort::Default`, never like a
        // hidden extra state. A `"default"` mapping of `null` means the model's default
        // is to not pass anything, so nothing is injected in that case.
        let default_key = ReasoningEffort::Default.to_string();
        for role_config in self.config.rendering.values() {
            for (field_name, field) in &role_config.context {
                if jinja_context.contains_key(field_name) {
                    continue;
                }
                let FieldConfig::Unique {
                    mapping: Some(mapping),
                    ..
                } = &field.config
                else {
                    continue;
                };
                if let Some(Some(default_value)) = mapping.get(&default_key) {
                    insert_into_context(
                        &mut jinja_context,
                        field_name.clone(),
                        minijinja::Value::from_serialize(default_value),
                    )?;
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::hanashi::config::HanashiConfig;

    fn renderer(config: HanashiConfig) -> Renderer {
        Renderer::new(config.resolve().unwrap().rendering)
    }

    fn render(
        renderer: &Renderer,
        messages: Vec<ChatMessage>,
    ) -> String {
        renderer.render(&messages, true, None, None, None).unwrap()
    }

    fn user_message() -> ChatMessage {
        ChatMessage::user().with_text("Hi".to_string())
    }

    fn system_message_with_effort(effort: ReasoningEffort) -> ChatMessage {
        ChatMessage::system().with_reasoning_effort(effort)
    }

    #[test]
    fn absent_reasoning_effort_renders_like_default() {
        let renderer = renderer(HanashiConfig::Qwen35);

        let without_block = render(&renderer, vec![user_message()]);
        let with_default_block =
            render(&renderer, vec![system_message_with_effort(ReasoningEffort::Default), user_message()]);

        assert_eq!(without_block, with_default_block);
        assert!(
            without_block.ends_with("<|im_start|>assistant\n<think>\n"),
            "expected a thinking-enabled generation prompt, got: {without_block:?}"
        );
    }

    #[test]
    fn disabled_reasoning_effort_still_disables_thinking() {
        let renderer = renderer(HanashiConfig::Qwen35);

        let prompt = render(&renderer, vec![system_message_with_effort(ReasoningEffort::Disabled), user_message()]);

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "expected a thinking-disabled generation prompt, got: {prompt:?}"
        );
    }

    #[test]
    fn absent_reasoning_effort_stays_unset_when_default_is_not_mapped() {
        // gpt-oss maps `"default"` to `null`: its default is to pass nothing, so
        // absence and an explicit default block must both leave the variable unset.
        let renderer = renderer(HanashiConfig::GptOss);

        let without_block = render(&renderer, vec![user_message()]);
        let with_default_block =
            render(&renderer, vec![system_message_with_effort(ReasoningEffort::Default), user_message()]);

        assert_eq!(without_block, with_default_block);
    }

    #[test]
    fn qwen38_preserves_historical_thinking_by_default() {
        let renderer = renderer(HanashiConfig::Qwen38);

        let history = vec![
            user_message(),
            ChatMessage::assistant().with_reasoning("deep thought".to_string()).with_text("the answer".to_string()),
            user_message(),
        ];
        let prompt = render(&renderer, history);

        assert!(prompt.contains("<think>\ndeep thought\n</think>"), "historical thinking dropped: {prompt:?}");
    }

    #[test]
    fn qwen38_every_supported_effort_renders() {
        let renderer = renderer(HanashiConfig::Qwen38);

        for effort in [
            ReasoningEffort::Disabled,
            ReasoningEffort::Default,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::XHigh,
        ] {
            render(&renderer, vec![system_message_with_effort(effort), user_message()]);
        }
    }

    #[test]
    fn qwen38_reasoning_effort_injects_instructions() {
        let renderer = renderer(HanashiConfig::Qwen38);

        let low = render(&renderer, vec![system_message_with_effort(ReasoningEffort::Low), user_message()]);
        let medium = render(&renderer, vec![system_message_with_effort(ReasoningEffort::Medium), user_message()]);
        let xhigh = render(&renderer, vec![system_message_with_effort(ReasoningEffort::XHigh), user_message()]);
        let default = render(&renderer, vec![system_message_with_effort(ReasoningEffort::Default), user_message()]);

        assert!(low.contains("Reasoning effort is set to low."), "got: {low:?}");
        assert!(!medium.contains("Reasoning effort is set to"), "got: {medium:?}");
        assert!(xhigh.contains("Reasoning effort is set to xhigh."), "got: {xhigh:?}");
        assert_eq!(default, xhigh);
    }

    #[test]
    fn qwen38_disabled_still_disables_thinking() {
        let renderer = renderer(HanashiConfig::Qwen38);

        let prompt = render(&renderer, vec![system_message_with_effort(ReasoningEffort::Disabled), user_message()]);

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "expected a thinking-disabled generation prompt, got: {prompt:?}"
        );
    }

    #[test]
    fn qwen38_capabilities_report_levels() {
        let capabilities = HanashiConfig::Qwen38.capabilities().unwrap();

        assert!(capabilities.supports_reasoning);
        assert!(capabilities.supports_disable_reasoning);
        for effort in [
            ReasoningEffort::Disabled,
            ReasoningEffort::Default,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::XHigh,
        ] {
            assert!(capabilities.reasoning_efforts.contains(&effort), "missing {effort}");
        }
        assert!(!capabilities.reasoning_efforts.contains(&ReasoningEffort::High));
    }
}
