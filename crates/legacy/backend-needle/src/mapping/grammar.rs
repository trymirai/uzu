use shoji::types::{basic::Grammar, session::chat::ChatMessage};

use crate::{error::Error, mapping::tools::collect_tools};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTools {
    pub json: String,
    pub fingerprint: String,
    pub grammar_fingerprint: Option<String>,
    pub is_grammar: bool,
}

pub fn bind_tools(
    messages: &[ChatMessage],
    grammar: Option<&Grammar>,
) -> Result<BoundTools, Error> {
    match grammar {
        None => {
            let json = crate::mapping::tools::tools_json(messages)?;
            Ok(BoundTools {
                fingerprint: json.clone(),
                json,
                grammar_fingerprint: None,
                is_grammar: false,
            })
        },
        Some(Grammar::Regex {
            ..
        }) => Err(Error::UnsupportedGrammar),
        Some(Grammar::JsonAny {}) => {
            let tool = serde_json::json!([{
                "name": "extract",
                "parameters": { "type": "object" }
            }]);
            let json = serde_json::to_string(&tool)?;
            Ok(BoundTools {
                fingerprint: json.clone(),
                json,
                grammar_fingerprint: Some("json_any".to_string()),
                is_grammar: true,
            })
        },
        Some(Grammar::JsonSchema {
            schema,
        }) => {
            if !collect_tools(messages)?.is_empty() {
                tracing::warn!("Needle grammar overrides session tools for this turn");
            }
            let parsed: serde_json::Value = serde_json::from_str(schema)?;
            let (name, parameters) = schema_tool(&parsed);
            let tool = serde_json::json!([{
                "name": name,
                "parameters": parameters,
            }]);
            let json = serde_json::to_string(&tool)?;
            Ok(BoundTools {
                fingerprint: json.clone(),
                json,
                grammar_fingerprint: Some(schema.clone()),
                is_grammar: true,
            })
        },
    }
}

fn schema_tool(schema: &serde_json::Value) -> (String, serde_json::Value) {
    let name = schema
        .get("title")
        .and_then(serde_json::Value::as_str)
        .filter(|title| !title.is_empty())
        .unwrap_or("extract")
        .to_string();
    if schema.get("type").and_then(serde_json::Value::as_str) == Some("object") && schema.get("properties").is_some() {
        return (name, schema.clone());
    }
    let mut parameters = serde_json::Map::new();
    parameters.insert("type".to_string(), serde_json::Value::String("object".to_string()));
    if let Some(properties) = schema.get("properties") {
        parameters.insert("properties".to_string(), properties.clone());
    } else {
        parameters.insert("properties".to_string(), serde_json::json!({}));
    }
    for key in ["required", "$defs", "definitions"] {
        if let Some(value) = schema.get(key) {
            parameters.insert(key.to_string(), value.clone());
        }
    }
    (name, serde_json::Value::Object(parameters))
}
