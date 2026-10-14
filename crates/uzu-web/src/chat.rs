use minijinja::{Environment, Error, ErrorKind};
use serde::Deserialize;
use serde_json::{Value, json};
use uzu_engine::engine::language_model::stream::SamplingMethod;

#[derive(Deserialize)]
pub(crate) struct ModelConfig {
    pub token_codec_config: CodecConfig,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub(crate) enum CodecConfig {
    ChatCodecConfig {
        prompt_template: String,
        output_parser_regex: Option<String>,
        system_role_name: String,
        user_role_name: String,
        assistant_role_name: String,
        bos_token: Option<String>,
        eos_token: Option<String>,
        end_of_thinking_tag: Option<String>,
        default_system_prompt: Option<String>,
        reasoning_config: Option<ReasoningConfig>,
    },
    RawTextCodecConfig,
}

#[derive(Deserialize)]
pub(crate) struct ReasoningConfig {
    default_reasoning_effort: String,
    field_name: String,
    reasoning_effort_to_field_value: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunParams {
    pub grammar: Option<GrammarRequest>,
    #[serde(default)]
    pub tools: Vec<shoji::types::basic::ToolDescription>,
    pub messages: Vec<Message>,
    pub sampling_policy: Option<SamplingPolicy>,
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub date_time_tool_enabled: bool,
    #[serde(default)]
    pub chart_tool_enabled: bool,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub(crate) enum GrammarRequest {
    JsonSchema {
        schema: Value,
    },
    Regex {
        pattern: String,
    },
    BuiltinJson,
}

#[cfg(target_arch = "wasm32")]
impl GrammarRequest {
    pub fn config(&self) -> uzu_engine::engine::language_model::grammar::GrammarConfig {
        use uzu_engine::engine::language_model::grammar::GrammarConfig;
        match self {
            Self::JsonSchema {
                schema,
            } => GrammarConfig::json_schema_simple(
                schema.as_str().map(str::to_owned).unwrap_or_else(|| schema.to_string()),
            ),
            Self::Regex {
                pattern,
            } => GrammarConfig::regex(pattern.clone(), false),
            Self::BuiltinJson => GrammarConfig::builtin_json(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub(crate) enum SamplingPolicy {
    Default,
    #[serde(alias = "Argmax")]
    Greedy,
    #[serde(rename_all = "camelCase")]
    Stochastic {
        temperature: Option<f32>,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
        repetition_penalty: Option<f32>,
        suffix_repetition_length: Option<u32>,
    },
}

pub(crate) fn sampling_method(
    policy: Option<&SamplingPolicy>,
    default: SamplingMethod,
) -> Result<SamplingMethod, String> {
    let method = match policy {
        None | Some(SamplingPolicy::Default) => default,
        Some(SamplingPolicy::Greedy) => SamplingMethod::Greedy,
        Some(SamplingPolicy::Stochastic {
            temperature: Some(t),
            ..
        }) if *t <= 0.0 => SamplingMethod::Greedy,
        Some(SamplingPolicy::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            suffix_repetition_length,
        }) => SamplingMethod::Stochastic {
            temperature: *temperature,
            top_k: *top_k,
            top_p: *top_p,
            min_p: *min_p,
            repetition_penalty: *repetition_penalty,
            suffix_repetition_length: repetition_penalty.map(|_| suffix_repetition_length.unwrap_or(32)),
        },
    };
    if let SamplingMethod::Stochastic {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        suffix_repetition_length,
    } = &method
    {
        for (name, value) in [("temperature", temperature), ("repetitionPenalty", repetition_penalty)] {
            if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return Err(format!("{name} must be a finite positive number"));
            }
        }
        for (name, value) in [("topP", top_p), ("minP", min_p)] {
            if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
                return Err(format!("{name} must be between 0 and 1"));
            }
        }
        if *top_k == Some(0) {
            return Err("topK must be positive".into());
        }
        if suffix_repetition_length.is_some_and(|n| n == 0 || n > 65536) {
            return Err("suffixRepetitionLength must be between 1 and 65536".into());
        }
        if repetition_penalty.is_some() && suffix_repetition_length.is_none() {
            return Err("repetitionPenalty requires suffixRepetitionLength".into());
        }
    }
    Ok(method)
}

impl CodecConfig {
    pub fn render(
        &self,
        params: &RunParams,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<String, String> {
        if params.messages.is_empty() {
            return Err("Add a message before generating a reply".into());
        }
        if params.date_time_tool_enabled || params.chart_tool_enabled || !params.tools.is_empty() {
            return Err("Tool calls require a registry chat encoding".into());
        }
        if params.max_output_tokens.is_some_and(|limit| limit == 0 || limit > 65536) {
            return Err("maxOutputTokens must be between 1 and 65536".into());
        }
        if params.messages.iter().any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant")) {
            return Err("Unsupported chat role".into());
        }
        let Self::ChatCodecConfig {
            prompt_template,
            system_role_name,
            user_role_name,
            assistant_role_name,
            bos_token,
            eos_token,
            default_system_prompt,
            reasoning_config,
            ..
        } = self
        else {
            return Ok(params.messages.iter().map(|m| m.content.as_str()).collect());
        };
        let mut messages = Vec::new();
        if !params.messages.iter().any(|m| m.role == "system")
            && let Some(default) = default_system_prompt
        {
            messages.push(json!({"role":system_role_name,"content":default}));
        }
        for message in &params.messages {
            let role = match message.role.as_str() {
                "system" => system_role_name,
                "assistant" => assistant_role_name,
                _ => user_role_name,
            };
            let mut value = json!({"role":role, "content":message.content});
            if let Some(reasoning) = &message.reasoning_content {
                value["reasoning_content"] = json!(reasoning);
            }
            messages.push(value);
        }
        let mut context = json!({
            "messages":messages,"add_generation_prompt":true,
            "bos_token":bos_token.as_deref().unwrap_or(""),"eos_token":eos_token.as_deref().unwrap_or(""),
            "tools":[],
        });
        if let Some(config) = reasoning_config {
            let requested = params.reasoning_effort.as_deref().unwrap_or("default");
            let effort = if requested == "default" {
                &config.default_reasoning_effort
            } else {
                requested
            };
            let value = config
                .reasoning_effort_to_field_value
                .get(effort)
                // The GUI names the off setting `disabled`; converted model
                // configs may retain the engine's `no_reasoning` name.
                .or_else(|| {
                    (effort == "disabled").then(|| config.reasoning_effort_to_field_value.get("no_reasoning")).flatten()
                })
                .ok_or_else(|| format!("This model does not support reasoning effort {effort}"))?;
            context[&config.field_name] = value.clone();
        } else if let Some(effort) = &params.reasoning_effort
            && !matches!(effort.as_str(), "default" | "disabled")
        {
            return Err("This model has no configurable reasoning effort".into());
        }
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        });
        env.add_function("strftime_now", move |format: String| -> Result<String, Error> {
            use std::fmt::Write;
            let mut rendered = String::new();
            write!(&mut rendered, "{}", now.format(&format))
                .map_err(|_| Error::new(ErrorKind::InvalidOperation, "Invalid strftime format"))?;
            Ok(rendered)
        });
        env.add_template("chat", prompt_template).map_err(|e| format!("Invalid model chat template: {e}"))?;
        env.get_template("chat").unwrap().render(context).map_err(|e| format!("Chat template: {e}"))
    }

    pub fn thinking_tag(&self) -> Option<&str> {
        match self {
            Self::ChatCodecConfig {
                end_of_thinking_tag,
                ..
            } => end_of_thinking_tag.as_deref().map(str::trim).filter(|s| !s.is_empty()),
            Self::RawTextCodecConfig => None,
        }
    }

    /// XML-style delimiters are described by the model's closing tag. A model
    /// may support reasoning without using it in every answer.
    pub fn thinking_opening_tag(&self) -> Option<String> {
        let tag = self.thinking_tag()?;
        let name = tag.strip_prefix("</")?.strip_suffix('>')?;
        (!name.is_empty()).then(|| format!("<{name}>"))
    }

    pub fn output_regex(&self) -> Option<&str> {
        match self {
            Self::ChatCodecConfig {
                output_parser_regex,
                ..
            } => output_parser_regex.as_deref(),
            Self::RawTextCodecConfig => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(value: Value) -> RunParams {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn output_limit_is_optional_and_explicit_limits_are_validated() {
        let run = params(json!({"messages":[{"role":"user","content":"hello"}]}));
        assert_eq!(run.max_output_tokens, None);
        assert!(CodecConfig::RawTextCodecConfig.render(&run, chrono::DateTime::UNIX_EPOCH).is_ok());
        for limit in [0, 65537] {
            let run = params(json!({"messages":[{"role":"user","content":"hello"}],"maxOutputTokens":limit}));
            assert!(CodecConfig::RawTextCodecConfig.render(&run, chrono::DateTime::UNIX_EPOCH).is_err());
        }
        let run = params(json!({"messages":[{"role":"user","content":"hello"}],"maxOutputTokens":4096}));
        assert_eq!(run.max_output_tokens, Some(4096));
    }

    #[test]
    fn renders_model_roles_reasoning_and_default_system_prompt() {
        let codec: CodecConfig = serde_json::from_value(json!({
            "type":"ChatCodecConfig", "prompt_template":"{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}{{ enable_thinking }}:{{ add_generation_prompt }}",
            "system_role_name":"S", "user_role_name":"U", "assistant_role_name":"A", "bos_token":"BOS", "default_system_prompt":"be helpful",
            "reasoning_config":{"default_reasoning_effort":"high","field_name":"enable_thinking","reasoning_effort_to_field_value":{"high":true,"disabled":false}}
        })).unwrap();
        let run = params(json!({"messages":[{"role":"user","content":"hello"}],"reasoningEffort":"disabled"}));
        assert_eq!(codec.render(&run, chrono::DateTime::UNIX_EPOCH).unwrap(), "BOSS:be helpful;U:hello;false:true");
    }

    #[test]
    fn rejects_tools_instead_of_ignoring_them() {
        let run = params(json!({"messages":[{"role":"user","content":"hello"}],"chartToolEnabled":true}));
        assert!(CodecConfig::RawTextCodecConfig.render(&run, chrono::DateTime::UNIX_EPOCH).is_err());
    }

    #[test]
    fn accepts_gui_disabled_effort_for_no_reasoning_model_configs() {
        let codec: CodecConfig = serde_json::from_value(json!({
            "type":"ChatCodecConfig", "prompt_template":"{{ enable_thinking }}",
            "system_role_name":"system", "user_role_name":"user", "assistant_role_name":"assistant",
            "reasoning_config":{"default_reasoning_effort":"medium","field_name":"enable_thinking","reasoning_effort_to_field_value":{"medium":true,"no_reasoning":false}}
        })).unwrap();
        let run = params(json!({"messages":[{"role":"user","content":"hello"}],"reasoningEffort":"disabled"}));
        assert_eq!(codec.render(&run, chrono::DateTime::UNIX_EPOCH).unwrap(), "false");
    }

    #[test]
    fn validates_sampling_and_preserves_greedy_temperature_zero() {
        let policy = |v| serde_json::from_value::<SamplingPolicy>(v).unwrap();
        assert_eq!(
            sampling_method(Some(&policy(json!({"type":"Stochastic","temperature":0}))), SamplingMethod::Greedy)
                .unwrap(),
            SamplingMethod::Greedy
        );
        assert!(sampling_method(Some(&policy(json!({"type":"Stochastic","topK":0}))), SamplingMethod::Greedy).is_err());
        assert!(
            sampling_method(Some(&policy(json!({"type":"Stochastic","topP":1.2}))), SamplingMethod::Greedy).is_err()
        );
        assert!(matches!(
            sampling_method(
                Some(&policy(json!({"type":"Stochastic","repetitionPenalty":1.1}))),
                SamplingMethod::Greedy
            )
            .unwrap(),
            SamplingMethod::Stochastic {
                suffix_repetition_length: Some(32),
                ..
            }
        ));
    }
}
