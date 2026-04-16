use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    TransformError, TransformSchema,
    execution::{access, collection, condition::Condition, construction, control_flow, string},
    regex::RegexEngine,
};

/// A sequence of operations applied left to right.
/// Empty pipeline passes input through unchanged (identity).
pub type Pipeline = Vec<Operation>;

/// A single segment in a path-based Get.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum PathSegment {
    Index(usize),
    Key(String),
}

/// Target for the Get operation — either a single key or a multi-step path.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum GetTarget {
    Path {
        path: Vec<PathSegment>,
    },
    Key {
        key: String,
    },
}

/// Target for the Call operation — static name or dynamic key lookup.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum CallTarget {
    Dynamic {
        key: String,
    },
    Static {
        name: String,
    },
}

/// A single case in a Switch operation.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SwitchCase {
    pub when: Condition,
    pub then: Pipeline,
}

/// A single transformation step.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    // Access
    /// Access a value by key or path. Returns Null if missing.
    /// Key: Object → Value | Path: Object|Array → Value
    Get {
        #[serde(flatten)]
        target: GetTarget,
    },
    /// Take the first element of an Array. Returns Null if empty or not an Array.
    /// Array → Value
    First,

    // Construction
    /// Build an Object from named fields. Each field pipeline receives the same input.
    /// Value → Object
    Object {
        fields: IndexMap<String, Pipeline>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        required: Vec<String>,
    },
    /// Produce a constant value, ignoring input.
    /// _ → Value
    Literal {
        value: Value,
    },
    /// Wrap input in a single-element Array.
    /// Value → Array
    ToArray,
    /// If input is Null, return the fallback value. Otherwise pass through.
    /// Value → Value
    Default {
        value: Value,
    },
    /// Replace a field's value through a mapping table. Returns input with field replaced.
    /// Object → Object
    Resolve {
        key: String,
        map: IndexMap<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Value>,
    },

    // Collection
    /// Apply a pipeline to each element of an Array.
    /// Array → Array
    Each {
        apply: Pipeline,
    },
    /// Apply a pipeline to each element of an Array, flattening Array results into the output.
    /// Non-array inputs pass through unchanged.
    /// Array → Array
    FlatMap {
        apply: Pipeline,
    },
    /// Keep only Array elements matching a condition.
    /// Array → Array
    Filter {
        condition: Condition,
    },
    /// Concatenate an Array of Strings with a separator.
    /// Array<String> → String
    Join {
        separator: String,
    },
    /// Group consecutive Array elements by a key, then merge each group.
    /// Array → Array
    Reduce {
        key: Pipeline,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        r#if: Option<Condition>,
        then: Pipeline,
    },

    // String
    /// Insert input string into a template at `{}` placeholder.
    /// String → String
    Format {
        template: String,
    },
    /// Regex replace_all on a string. Template can reference capture groups via `$1`, `$2`.
    /// String → String
    RegexReplace {
        pattern: String,
        template: String,
        #[serde(default)]
        regex_engine: RegexEngine,
    },
    /// Find all regex matches in a string. Extracts capture group 1 if present, otherwise group 0.
    /// String → Array<String>
    RegexFindAll {
        pattern: String,
        #[serde(default)]
        regex_engine: RegexEngine,
    },
    /// Parse a JSON string into a Value. With `repair: true`, attempts to fix malformed JSON.
    /// String → Value
    ParseJson {
        #[serde(default)]
        repair: bool,
    },

    // Control Flow
    /// Branch on a value produced by the key pipeline.
    /// Value → Value
    Switch {
        key: Pipeline,
        cases: Vec<SwitchCase>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<Pipeline>,
    },
    /// Call a named pipeline. Name is static or read from input field.
    /// Value → Value
    Call {
        #[serde(flatten)]
        target: CallTarget,
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        arguments: IndexMap<String, Value>,
    },
    /// Run pipeline only when field equals true, optionally extracting a working value.
    /// If field is not true, returns the working value (or full input) unchanged.
    /// Value → Value
    On {
        field: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        with: Option<String>,
        r#do: Pipeline,
    },
}

impl fmt::Display for Operation {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Get {
                target,
            } => write!(formatter, "get({target})"),
            Self::First => write!(formatter, "first"),
            Self::Object {
                fields,
                ..
            } => {
                write!(formatter, "object({})", fields.len())
            },
            Self::Literal {
                value,
            } => write!(formatter, "literal({value})"),
            Self::ToArray => write!(formatter, "to_array"),
            Self::Resolve {
                key,
                ..
            } => write!(formatter, "resolve({key})"),
            Self::Default {
                ..
            } => write!(formatter, "default"),
            Self::Each {
                ..
            } => write!(formatter, "each"),
            Self::FlatMap {
                ..
            } => write!(formatter, "flat_map"),
            Self::Filter {
                ..
            } => write!(formatter, "filter"),
            Self::Join {
                separator,
            } => {
                write!(formatter, "join({separator})")
            },
            Self::Reduce {
                ..
            } => write!(formatter, "reduce"),
            Self::Format {
                ..
            } => write!(formatter, "format"),
            Self::RegexReplace {
                ..
            } => write!(formatter, "regex_replace"),
            Self::RegexFindAll {
                ..
            } => write!(formatter, "regex_find_all"),
            Self::ParseJson {
                repair,
            } => {
                write!(formatter, "parse_json(repair={repair})")
            },
            Self::Switch {
                cases,
                ..
            } => {
                write!(formatter, "switch({})", cases.len())
            },
            Self::Call {
                target,
                ..
            } => {
                write!(formatter, "call({target})")
            },
            Self::On {
                field,
                with,
                ..
            } => match with {
                Some(key) => write!(formatter, "on({field}, {key})"),
                None => write!(formatter, "on({field})"),
            },
        }
    }
}

impl fmt::Display for GetTarget {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Key {
                key,
            } => write!(formatter, "{key}"),
            Self::Path {
                path,
            } => {
                let segments: Vec<String> = path
                    .iter()
                    .map(|segment| match segment {
                        PathSegment::Key(key) => key.clone(),
                        PathSegment::Index(index) => index.to_string(),
                    })
                    .collect();
                write!(formatter, "{}", segments.join("."))
            },
        }
    }
}

impl fmt::Display for CallTarget {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Static {
                name,
            } => write!(formatter, "{name}"),
            Self::Dynamic {
                key,
            } => {
                write!(formatter, "dynamic({key})")
            },
        }
    }
}

impl Operation {
    pub fn execute(
        &self,
        input: Value,
        schema: &TransformSchema,
    ) -> Result<Value, TransformError> {
        match self {
            // Access
            Self::Get {
                target,
            } => access::execute_get(target, input),
            Self::First => access::execute_first(input),

            // Construction
            Self::Object {
                fields,
                required,
            } => construction::execute_object(fields, required, input, schema),
            Self::Literal {
                value,
            } => construction::execute_literal(value),
            Self::ToArray => construction::execute_to_array(input),
            Self::Default {
                value,
            } => construction::execute_default(value, input),
            Self::Resolve {
                key,
                map,
                default,
            } => construction::execute_resolve(key, map, default.as_ref(), input),

            // Collection
            Self::Each {
                apply,
            } => collection::execute_each(apply, input, schema),
            Self::FlatMap {
                apply,
            } => collection::execute_flat_map(apply, input, schema),
            Self::Filter {
                condition,
            } => collection::execute_filter(condition, input),
            Self::Join {
                separator,
            } => collection::execute_join(separator, input),
            Self::Reduce {
                key,
                r#if,
                then,
            } => collection::execute_reduce(key, r#if, then, input, schema),

            // String
            Self::Format {
                template,
            } => string::execute_format(template, input),
            Self::RegexReplace {
                pattern,
                template,
                regex_engine,
            } => string::execute_regex_replace(pattern, template, regex_engine, input),
            Self::RegexFindAll {
                pattern,
                regex_engine,
            } => string::execute_regex_find_all(pattern, regex_engine, input),
            Self::ParseJson {
                repair,
            } => string::execute_parse_json(*repair, input),

            // Control Flow
            Self::Switch {
                key,
                cases,
                default,
            } => control_flow::execute_switch(key, cases, default, input, schema),
            Self::Call {
                target,
                arguments,
            } => control_flow::execute_call(target, arguments, input, schema),
            Self::On {
                field,
                with,
                r#do,
            } => control_flow::execute_on(field, with.as_deref(), r#do, input, schema),
        }
    }
}
