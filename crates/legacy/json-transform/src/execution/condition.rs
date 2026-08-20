use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A condition for filtering or branching.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Condition {
    Equals {
        value: Value,
    },
    StartsWith {
        value: String,
    },
    Contains {
        value: String,
    },
    IsNull,
    Field {
        key: String,
        condition: Box<Condition>,
    },
    Not {
        condition: Box<Condition>,
    },
    And {
        conditions: Vec<Condition>,
    },
    Or {
        conditions: Vec<Condition>,
    },
}

impl fmt::Display for Condition {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Equals {
                value,
            } => write!(formatter, "equals({value})"),
            Self::StartsWith {
                value,
            } => {
                write!(formatter, "starts_with({value})")
            },
            Self::Contains {
                value,
            } => {
                write!(formatter, "contains({value})")
            },
            Self::IsNull => write!(formatter, "is_null"),
            Self::Field {
                key,
                condition,
            } => {
                write!(formatter, "field({key}, {condition})")
            },
            Self::Not {
                condition,
            } => {
                write!(formatter, "not({condition})")
            },
            Self::And {
                conditions,
            } => {
                write!(formatter, "and({})", conditions.len())
            },
            Self::Or {
                conditions,
            } => {
                write!(formatter, "or({})", conditions.len())
            },
        }
    }
}

impl Condition {
    pub fn evaluate(
        &self,
        input: &Value,
    ) -> bool {
        match self {
            Condition::Equals {
                value,
            } => input == value,
            Condition::StartsWith {
                value,
            } => input.as_str().map(|text| text.starts_with(value.as_str())).unwrap_or(false),
            Condition::Contains {
                value,
            } => input.as_str().map(|text| text.contains(value.as_str())).unwrap_or(false),
            Condition::IsNull => input.is_null(),
            Condition::Field {
                key,
                condition,
            } => {
                let field_value = match input {
                    Value::Object(map) => map.get(key).unwrap_or(&Value::Null),
                    _ => &Value::Null,
                };
                condition.evaluate(field_value)
            },
            Condition::Not {
                condition,
            } => !condition.evaluate(input),
            Condition::And {
                conditions,
            } => conditions.iter().all(|condition| condition.evaluate(input)),
            Condition::Or {
                conditions,
            } => conditions.iter().any(|condition| condition.evaluate(input)),
        }
    }
}
