use schemars::JsonSchema;
use serde_json::{self, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredOutput {
    Schema(String),
    AnyJson,
}

#[derive(Debug, Clone)]
pub enum GrammarConfig {
    JsonSchema {
        schema: String,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    },
    Regex {
        pattern: String,
        print_converted_ebnf: bool,
    },
    BuiltinJson,
}

impl GrammarConfig {
    pub fn json_schema(
        schema: String,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    ) -> Self {
        Self::JsonSchema {
            schema,
            any_whitespace,
            indent,
            separators,
            strict_mode,
        }
    }

    pub fn json_schema_simple(schema: String) -> Self {
        Self::JsonSchema {
            schema,
            any_whitespace: true,
            indent: Some(2),
            separators: Some((",".to_string(), ":".to_string())),
            strict_mode: true,
        }
    }

    pub fn structured_output_from_schema(raw: &str) -> StructuredOutput {
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(raw) else {
            return StructuredOutput::Schema(raw.to_string());
        };

        match obj.get("type").and_then(Value::as_str) {
            Some("json_object") => return StructuredOutput::AnyJson,
            Some("json_schema") => {
                if let Some(inner) = obj.get("json_schema") {
                    return StructuredOutput::Schema(unwrap_inner_schema(inner));
                }
                if let Some(schema) = obj.get("schema") {
                    return StructuredOutput::Schema(schema.to_string());
                }
            },
            _ => {},
        }

        if let Some(schema) = obj.get("schema") {
            let looks_like_envelope = obj.get("name").is_some_and(Value::is_string)
                && !obj.contains_key("type")
                && !obj.contains_key("properties")
                && !obj.contains_key("$ref");
            if looks_like_envelope {
                return StructuredOutput::Schema(schema.to_string());
            }
        }

        StructuredOutput::Schema(raw.to_string())
    }

    pub fn regex(
        pattern: String,
        print_converted_ebnf: bool,
    ) -> Self {
        Self::Regex {
            pattern,
            print_converted_ebnf,
        }
    }

    pub fn builtin_json() -> Self {
        Self::BuiltinJson
    }

    pub fn from_json_schema_type<T: JsonSchema>() -> Result<Self, String> {
        let schema_root = schemars::schema_for!(T);
        let schema_str =
            serde_json::to_string(&schema_root).map_err(|e| format!("Failed to serialize schema: {}", e))?;

        Ok(Self::JsonSchema {
            schema: schema_str,
            any_whitespace: true,
            indent: Some(2),
            separators: Some((",".to_string(), ":".to_string())),
            strict_mode: true,
        })
    }

    pub fn from_json_schema_type_with_config<T: JsonSchema>(
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    ) -> Result<Self, String> {
        let schema_root = schemars::schema_for!(T);
        let schema_str =
            serde_json::to_string(&schema_root).map_err(|e| format!("Failed to serialize schema: {}", e))?;

        Ok(Self::JsonSchema {
            schema: schema_str,
            any_whitespace,
            indent,
            separators,
            strict_mode,
        })
    }
}

fn unwrap_inner_schema(inner: &Value) -> String {
    if let Some(schema) = inner.as_object().and_then(|object| object.get("schema")) {
        return schema.to_string();
    }
    inner.to_string()
}

#[cfg(test)]
#[path = "../../../unit/session/config/grammar_config.rs"]
mod tests;
