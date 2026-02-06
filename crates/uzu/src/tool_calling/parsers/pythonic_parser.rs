use indexmap::IndexMap;
use regex::Regex;

use super::ToolCallParser;
use crate::tool_calling::{ToolCall, Value};

pub struct ToolCallPythonicParser {
    function_regex: Regex,
    argument_separator: char,
    string_token: Option<String>,
}

impl ToolCallPythonicParser {
    pub fn new(
        function_regex: String,
        argument_separator: String,
        string_token: Option<String>,
    ) -> Option<Self> {
        Some(Self {
            function_regex: Regex::new(&function_regex).ok()?,
            argument_separator: argument_separator.chars().next()?,
            string_token,
        })
    }
}

impl ToolCallParser for ToolCallPythonicParser {
    fn parse(&self, content: String) -> Option<Vec<ToolCall>> {
        let mut tool_calls = Vec::new();
        for function_captures in
            self.function_regex.captures_iter(content.trim())
        {
            let function_name = function_captures
                .name("function_name")?
                .as_str()
                .to_string();
            let arguments_text =
                function_captures.name("arguments")?.as_str();

            let mut arguments = IndexMap::new();
            for pair in split_respecting_nesting(
                arguments_text,
                ',',
                self.string_token.clone(),
            ) {
                let pair = pair.trim();
                if pair.is_empty() {
                    continue;
                }
                let separator_position =
                    pair.find(self.argument_separator)?;
                let argument_name =
                    pair[..separator_position].trim().to_string();
                let raw_value = pair[separator_position + 1..].trim();
                arguments.insert(
                    argument_name,
                    parse_value(
                        raw_value.to_string(),
                        self.string_token.clone(),
                    ),
                );
            }

            tool_calls.push(ToolCall::new(
                function_name,
                Value::Object(arguments),
            ));
        }

        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        }
    }
}

enum NestingContext {
    Parenthesis,
    Bracket,
    Brace,
    SingleQuote,
    DoubleQuote,
    StringToken,
}

fn split_respecting_nesting(
    text: &str,
    delimiter: char,
    string_token: Option<String>,
) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut stack: Vec<NestingContext> = Vec::new();

    for (index, character) in text.char_indices() {
        let in_string = matches!(
            stack.last(),
            Some(
                NestingContext::SingleQuote
                    | NestingContext::DoubleQuote
                    | NestingContext::StringToken
            )
        );

        if let Some(token) = &string_token {
            if text[index..].starts_with(token.as_str()) {
                match stack.last() {
                    Some(NestingContext::StringToken) => {
                        stack.pop();
                    },
                    _ if !in_string => {
                        stack.push(NestingContext::StringToken);
                    },
                    _ => {},
                }
                continue;
            }
        }

        match character {
            '\'' if !matches!(
                stack.last(),
                Some(
                    NestingContext::DoubleQuote
                        | NestingContext::StringToken
                )
            ) =>
            {
                match stack.last() {
                    Some(NestingContext::SingleQuote) => {
                        stack.pop();
                    },
                    _ => stack.push(NestingContext::SingleQuote),
                }
            },
            '"' if !matches!(
                stack.last(),
                Some(
                    NestingContext::SingleQuote
                        | NestingContext::StringToken
                )
            ) =>
            {
                match stack.last() {
                    Some(NestingContext::DoubleQuote) => {
                        stack.pop();
                    },
                    _ => stack.push(NestingContext::DoubleQuote),
                }
            },
            '(' if !in_string => {
                stack.push(NestingContext::Parenthesis);
            },
            '[' if !in_string => {
                stack.push(NestingContext::Bracket);
            },
            '{' if !in_string => {
                stack.push(NestingContext::Brace);
            },
            ')' if matches!(
                stack.last(),
                Some(NestingContext::Parenthesis)
            ) =>
            {
                stack.pop();
            },
            ']' if matches!(
                stack.last(),
                Some(NestingContext::Bracket)
            ) =>
            {
                stack.pop();
            },
            '}' if matches!(
                stack.last(),
                Some(NestingContext::Brace)
            ) =>
            {
                stack.pop();
            },
            current_character
                if current_character == delimiter && stack.is_empty() =>
            {
                parts.push(text[start..index].to_string());
                start = index + current_character.len_utf8();
            },
            _ => {},
        }
    }
    if start <= text.len() {
        parts.push(text[start..].to_string());
    }
    parts
}

fn parse_value(raw_value: String, string_token: Option<String>) -> Value {
    let normalized = normalize_value(raw_value.clone(), string_token);
    serde_json::from_str::<Value>(&normalized)
        .unwrap_or_else(|_| Value::String(raw_value))
}

fn normalize_value(
    raw_value: String,
    string_token: Option<String>,
) -> String {
    match raw_value.as_str() {
        "True" => "true".to_string(),
        "False" => "false".to_string(),
        "None" => "null".to_string(),
        _ => {
            if let Some(token) = string_token {
                if let Some(content) = raw_value
                    .strip_prefix(token.as_str())
                    .and_then(|value| value.strip_suffix(token.as_str()))
                {
                    return format!("\"{}\"", content);
                }
            }
            if let Some(content) = raw_value
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
            {
                return format!("\"{}\"", content);
            }
            raw_value
        },
    }
}
