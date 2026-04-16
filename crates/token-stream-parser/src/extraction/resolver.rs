use std::collections::{HashMap, HashSet};

use json_transform::TransformSchema;
use serde_json::Value;

use crate::{
    framing::FramingParserSection,
    reduction::{ReductionParserSection, ReductionParserState},
};

const NODE_NAME_TEXT: &str = "$text";
const SECTION_KEY_TYPE: &str = "type";
const SECTION_KEY_VALUE: &str = "value";
const PIPELINE_NAME_ROOT: &str = "root";
const PIPELINE_INPUT_NAME: &str = "$name";
const PIPELINE_INPUT_FINISHED: &str = "$finished";
const PIPELINE_INPUT_VALUE: &str = "$value";

type ExtractionParserGroupPath = Vec<usize>;

/// Computes extraction output from reduction state using transformation pipelines
pub(crate) struct ExtractionParserResolver {
    cache: HashMap<ExtractionParserGroupPath, Value>,
    sections_compose_groups: HashSet<String>,
    schema: Option<TransformSchema>,
}

impl ExtractionParserResolver {
    pub fn new(
        sections_compose_groups: HashSet<String>,
        schema: Option<TransformSchema>,
    ) -> Self {
        Self {
            cache: HashMap::new(),
            sections_compose_groups,
            schema,
        }
    }

    pub fn reset(&mut self) {
        self.cache.clear();
    }

    /// Compute output value from reduction state
    #[tracing::instrument(skip_all)]
    pub fn compute_output(
        &mut self,
        reduction_state: &ReductionParserState,
    ) -> Value {
        let child_values = self.compute_children(&reduction_state.sections, &vec![]);

        let root_values: Vec<Value> = child_values
            .into_iter()
            .filter(|(name, _)| name != NODE_NAME_TEXT)
            .map(|(_, value)| value)
            .filter(|value| !is_empty(value))
            .collect();

        let composed = Value::Array(root_values);
        execute_named_pipeline(PIPELINE_NAME_ROOT, false, composed, &self.schema)
    }
}

impl ExtractionParserResolver {
    /// Compute values for a list of reduction sections
    fn compute_children(
        &mut self,
        sections: &[ReductionParserSection],
        parent_path: &ExtractionParserGroupPath,
    ) -> Vec<(String, Value)> {
        let mut results = Vec::new();

        for (section_index, section) in sections.iter().enumerate() {
            match section {
                ReductionParserSection::Frame(frame) => {
                    if let Some(text) = extract_text(frame) {
                        merge_or_push_text(&mut results, text);
                    }
                },
                ReductionParserSection::Group {
                    name,
                    finished,
                    sections: group_sections,
                    ..
                } => {
                    let mut group_path = parent_path.clone();
                    group_path.push(section_index);

                    let value = self.compute_group(name, group_sections, *finished, &group_path);
                    results.push((name.clone(), value));
                },
            }
        }

        results
    }

    /// Compute a single group's value, cached if finalized, computed otherwise
    #[tracing::instrument(skip_all, fields(name = name, finished = finished))]
    fn compute_group(
        &mut self,
        name: &str,
        sections: &[ReductionParserSection],
        finished: bool,
        path: &ExtractionParserGroupPath,
    ) -> Value {
        if let Some(cached) = self.cache.get(path) {
            tracing::trace!("cached");
            return cached.clone();
        }

        let child_values = self.compute_children(sections, path);

        let sections_compose = self.sections_compose_groups.contains(name);
        let composed = auto_compose(child_values, sections_compose);

        let value = execute_named_pipeline(name, finished, composed, &self.schema);

        if finished {
            tracing::trace!(group = name, "finalized");
            self.cache.insert(path.clone(), value.clone());
        }

        value
    }
}

fn auto_compose(
    child_values: Vec<(String, Value)>,
    sections_compose: bool,
) -> Value {
    if sections_compose {
        return compose_typed_sections(child_values);
    }

    let has_text = child_values.iter().any(|(name, _)| name == NODE_NAME_TEXT);
    let has_groups = child_values.iter().any(|(name, _)| name != NODE_NAME_TEXT);

    match (has_text, has_groups) {
        // Text-only → joined string
        (_, false) => {
            let text: String = child_values.iter().filter_map(|(_, value)| value.as_str()).collect();
            Value::String(text)
        },
        // Mixed (text + groups) → array of {type, value}
        (true, true) => compose_typed_sections(child_values),
        // Groups only with unique names → object {name: value, ...}
        (false, true) => {
            let mut seen = HashSet::new();
            let all_unique = child_values.iter().all(|(name, _)| seen.insert(name.as_str()));

            if all_unique {
                let mut map = serde_json::Map::new();
                for (name, value) in child_values {
                    map.insert(name, value);
                }
                Value::Object(map)
            } else {
                compose_typed_sections(child_values)
            }
        },
    }
}

fn compose_typed_sections(child_values: Vec<(String, Value)>) -> Value {
    let mut sections: Vec<Value> = Vec::new();
    for (name, value) in child_values {
        // Spread arrays for non-text children into separate sections
        if name != NODE_NAME_TEXT {
            if let Value::Array(items) = value {
                for item in items {
                    let mut map = serde_json::Map::new();
                    map.insert(SECTION_KEY_TYPE.to_string(), Value::String(name.clone()));
                    map.insert(SECTION_KEY_VALUE.to_string(), item);
                    sections.push(Value::Object(map));
                }
                continue;
            }
        }
        let mut map = serde_json::Map::new();
        map.insert(SECTION_KEY_TYPE.to_string(), Value::String(name));
        map.insert(SECTION_KEY_VALUE.to_string(), value);
        sections.push(Value::Object(map));
    }
    Value::Array(sections)
}

fn execute_named_pipeline(
    name: &str,
    finished: bool,
    composed_value: Value,
    schema: &Option<TransformSchema>,
) -> Value {
    let Some(schema) = schema else {
        return composed_value;
    };
    if !schema.pipelines.contains_key(name) {
        return composed_value;
    }
    let _span = tracing::info_span!("execute_named_pipeline", name = name, finished = finished,).entered();
    let fallback = composed_value.clone();

    let mut pipeline_input_map = serde_json::Map::new();
    pipeline_input_map.insert(PIPELINE_INPUT_NAME.to_string(), Value::String(name.to_string()));
    pipeline_input_map.insert(PIPELINE_INPUT_FINISHED.to_string(), Value::Bool(finished));
    pipeline_input_map.insert(PIPELINE_INPUT_VALUE.to_string(), composed_value);
    let pipeline_input = Value::Object(pipeline_input_map);
    schema.execute(name, pipeline_input).unwrap_or(fallback)
}

fn extract_text(frame: &FramingParserSection) -> Option<String> {
    let text: String = match frame {
        FramingParserSection::Text(tokens) => tokens.iter().map(|token| token.value.as_str()).collect(),
        FramingParserSection::Marker(token) => token.value.clone(),
    };
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn merge_or_push_text(
    results: &mut Vec<(String, Value)>,
    text: String,
) {
    if let Some((name, Value::String(existing))) = results.last_mut() {
        if name == NODE_NAME_TEXT {
            existing.push_str(&text);
            return;
        }
    }
    results.push((NODE_NAME_TEXT.to_string(), Value::String(text)));
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        _ => false,
    }
}
