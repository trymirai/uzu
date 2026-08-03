use serde_json::Value;

use crate::{
    TransformError,
    regex::{Regex, RegexEngine},
};

#[tracing::instrument(skip(input))]
pub fn execute_format(
    template: &str,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::String(text) = input else {
        return Ok(Value::Null);
    };
    Ok(Value::String(template.replace("{}", &text)))
}

#[tracing::instrument(skip_all)]
pub fn execute_regex_replace(
    pattern: &str,
    template: &str,
    regex_engine: &RegexEngine,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::String(text) = input else {
        return Ok(Value::Null);
    };
    let regex = Regex::new(pattern, regex_engine)?;
    let replaced = regex.replace_all(&text, template);
    Ok(Value::String(replaced))
}

#[tracing::instrument(skip_all)]
pub fn execute_regex_find_all(
    pattern: &str,
    regex_engine: &RegexEngine,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::String(text) = input else {
        return Ok(Value::Null);
    };
    let regex = Regex::new(pattern, regex_engine)?;
    let matches: Vec<Value> = regex
        .captures_iter(&text)
        .iter()
        .map(|captures| {
            let matched_text = captures
                .get(1)
                .or_else(|| captures.get(0))
                .map(|regex_match| regex_match.text.clone())
                .unwrap_or_default();
            Value::String(matched_text)
        })
        .collect();
    Ok(Value::Array(matches))
}

#[tracing::instrument(skip(input))]
pub fn execute_split_top_level(
    separator: char,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::String(text) = input else {
        return Ok(Value::Null);
    };

    let mut values = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (index, character) in text.char_indices() {
        if let Some(quote_character) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == quote_character {
                quote = None;
            }
            continue;
        }

        match character {
            '"' | '\'' => quote = Some(character),
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth = depth.saturating_sub(1),
            _ if character == separator && depth == 0 => {
                if start < index {
                    values.push(Value::String(text[start..index].to_string()));
                }
                start = index + character.len_utf8();
            },
            _ => {},
        }
    }

    if start < text.len() {
        values.push(Value::String(text[start..].to_string()));
    }

    Ok(Value::Array(values))
}

#[tracing::instrument(skip(input))]
pub fn execute_parse_json(
    repair: bool,
    input: Value,
) -> Result<Value, TransformError> {
    let Value::String(text) = input else {
        return Ok(Value::Null);
    };
    if repair {
        let options = llm_json::RepairOptions {
            ensure_ascii: false,
            ..Default::default()
        };
        let repaired = llm_json::repair_json(&text, &options).unwrap_or_else(|_| text.clone());
        let parsed = serde_json::from_str::<Value>(&repaired).unwrap_or(Value::String(text));
        Ok(parsed)
    } else {
        let parsed = serde_json::from_str::<Value>(&text).map_err(|error| TransformError::InvalidJson {
            message: error.to_string(),
        })?;
        Ok(parsed)
    }
}
