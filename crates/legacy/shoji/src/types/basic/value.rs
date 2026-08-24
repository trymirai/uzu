use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[bindings::export(Structure(Class))]
#[derive(Clone, PartialEq)]
pub struct Value {
    pub json: String,
}

impl fmt::Debug for Value {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let serde_value: serde_json::Value = serde_json::from_str(&self.json).map_err(|_| fmt::Error)?;
        let pretty_json = serde_json::to_string_pretty(&serde_value).map_err(|_| fmt::Error)?;
        write!(formatter, "{pretty_json}")
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let serde_value: serde_json::Value = serde_json::from_str(&self.json).map_err(serde::ser::Error::custom)?;
        serde_value.serialize(serializer)
    }
}

impl<'d> Deserialize<'d> for Value {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_value = serde_json::Value::deserialize(deserializer)?;
        Ok(Value {
            json: serde_value.to_string(),
        })
    }
}

impl From<serde_json::Value> for Value {
    fn from(value: serde_json::Value) -> Self {
        Self {
            json: value.to_string(),
        }
    }
}

impl TryFrom<Value> for serde_json::Value {
    type Error = serde_json::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        serde_json::from_str(&value.json)
    }
}

/// Parses JSON text produced by a model: strict first, then one repair pass that
/// escapes literal control characters and bare quotes inside string values.
pub fn parse_lenient_json(text: &str) -> Option<serde_json::Value> {
    serde_json::from_str(text).ok().or_else(|| serde_json::from_str(&repair_json_text(text)).ok())
}

/// Repairs model-generated JSON text inside string values: escapes literal
/// control characters, doubles backslashes that do not form a valid JSON
/// escape (the model meant a literal backslash), and escapes quotes that cannot
/// terminate a string (a closing quote must be followed by a JSON delimiter).
/// Text outside strings passes through untouched.
pub fn repair_json_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut in_string = false;
    while let Some(c) = chars.next() {
        if in_string {
            match c {
                '\n' => {
                    out.push_str("\\n");
                    continue;
                },
                '\r' => {
                    out.push_str("\\r");
                    continue;
                },
                '\t' => {
                    out.push_str("\\t");
                    continue;
                },
                c if (c as u32) < 0x20 => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                    continue;
                },
                '\\' => {
                    if matches!(chars.peek(), Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')) {
                        out.push('\\');
                        out.push(chars.next().unwrap());
                    } else {
                        out.push_str("\\\\");
                    }
                    continue;
                },
                '"' => {
                    let mut lookahead = chars.clone();
                    let terminates = loop {
                        match lookahead.next() {
                            Some(c) if c.is_whitespace() => {},
                            Some(',' | '}' | ']' | ':') | None => break true,
                            Some(_) => break false,
                        }
                    };
                    if !terminates {
                        out.push('\\');
                    } else {
                        in_string = false;
                    }
                    out.push('"');
                    continue;
                },
                _ => {},
            }
        } else if c == '"' {
            in_string = true;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_json_text_only_touches_string_values() {
        assert_eq!(repair_json_text("\"esc\\ned\\n\""), "\"esc\\ned\\n\"");
        assert_eq!(repair_json_text("\"a\nb\""), "\"a\\nb\"");
        assert_eq!(repair_json_text("[1,\n2]"), "[1,\n2]");
        assert_eq!(
            repair_json_text(r#"{"a": "called "The Square Mile" — here"}"#),
            r#"{"a": "called \"The Square Mile\" — here"}"#
        );
        assert_eq!(repair_json_text(r#"{"a": "x", "b": "y"}"#), r#"{"a": "x", "b": "y"}"#);
        assert_eq!(repair_json_text(r#"{"cmd": "grep 'a\|b'}""#), r#"{"cmd": "grep 'a\\|b'}""#);
        assert_eq!(repair_json_text(r#"{"a": "x\ny\"z"}"#), r#"{"a": "x\ny\"z"}"#);
        assert_eq!(repair_json_text(r#"{"a": "C:\\temp"}"#), r#"{"a": "C:\\temp"}"#);
    }

    #[test]
    fn parse_lenient_json_repairs_control_characters_and_bare_quotes() {
        assert_eq!(parse_lenient_json("{ \"content\": \"a\nb\" }"), Some(serde_json::json!({ "content": "a\nb" })));
        assert_eq!(
            parse_lenient_json(r#"{ "content": "called "The Square Mile" home" }"#),
            Some(serde_json::json!({ "content": "called \"The Square Mile\" home" }))
        );
        assert_eq!(parse_lenient_json("{oops"), None);
    }
}
