use serde_json::Value;

use crate::ParserState;

#[derive(Debug, Clone)]
pub struct ExtractionParserState {
    pub value: Value,
}

impl ExtractionParserState {
    pub fn new() -> Self {
        Self {
            value: Value::Null,
        }
    }
}

impl ParserState for ExtractionParserState {
    fn is_substate_of(
        &self,
        other: &Self,
    ) -> bool {
        value_is_substate(&self.value, &other.value)
    }

    fn tokens(&self) -> Vec<&crate::types::Token> {
        Vec::new()
    }
}

fn value_is_substate(
    current: &Value,
    other: &Value,
) -> bool {
    let is_empty = match current {
        Value::Null => true,
        Value::String(string_value) => string_value.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        _ => false,
    };
    if is_empty {
        return true;
    }

    match (current, other) {
        (Value::String(current_string), Value::String(other_string)) => {
            other_string.starts_with(current_string.as_str())
        },
        (Value::Array(current_items), Value::Array(other_items)) => {
            let check_count = current_items.len();
            for index in 0..check_count {
                if index < check_count - 1 {
                    if current_items[index] != other_items[index] {
                        return false;
                    }
                } else if !value_is_substate(&current_items[index], &other_items[index]) {
                    return false;
                }
            }
            true
        },
        (Value::Object(current_map), Value::Object(other_map)) => {
            for (key, current_value) in current_map {
                match other_map.get(key) {
                    Some(other_value) => {
                        if !value_is_substate(current_value, other_value) {
                            return false;
                        }
                    },
                    None => return false,
                }
            }
            true
        },
        (Value::String(_), _) => true,
        _ => current == other,
    }
}
