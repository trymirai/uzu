use schemars::JsonSchema;
use serde_json;

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

    pub fn from_json_schema_type<T>() -> Result<Self, String>
    where
        T: JsonSchema,
    {
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

    pub fn from_json_schema_type_with_config<T>(
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(String, String)>,
        strict_mode: bool,
    ) -> Result<Self, String>
    where
        T: JsonSchema,
    {
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
