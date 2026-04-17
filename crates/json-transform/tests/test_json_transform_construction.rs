mod helpers;

use helpers::{execute_root, get_key};
use indexmap::IndexMap;
use json_transform::execution::Operation;
use serde_json::json;

#[test]
fn test_object_extracts_fields() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), vec![get_key("name")]);
    fields.insert("age".to_string(), vec![get_key("age")]);
    let result = execute_root(
        vec![Operation::Object {
            fields,
            required: vec![],
        }],
        json!({"name": "Alice", "age": 30}),
    )
    .unwrap();
    assert_eq!(result, json!({"name": "Alice", "age": 30}));
}

#[test]
fn test_object_omits_empty_non_required() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), vec![get_key("name")]);
    fields.insert("missing".to_string(), vec![get_key("missing")]);
    let result = execute_root(
        vec![Operation::Object {
            fields,
            required: vec![],
        }],
        json!({"name": "Alice"}),
    )
    .unwrap();
    assert_eq!(result, json!({"name": "Alice"}));
}

#[test]
fn test_object_keeps_empty_required() {
    let mut fields = IndexMap::new();
    fields.insert("name".to_string(), vec![get_key("name")]);
    fields.insert("missing".to_string(), vec![get_key("missing")]);
    let result = execute_root(
        vec![Operation::Object {
            fields,
            required: vec!["missing".to_string()],
        }],
        json!({"name": "Alice"}),
    )
    .unwrap();
    assert_eq!(result, json!({"name": "Alice", "missing": null}));
}

#[test]
fn test_literal_ignores_input() {
    let result = execute_root(
        vec![Operation::Literal {
            value: json!(42),
        }],
        json!("Anything"),
    )
    .unwrap();
    assert_eq!(result, json!(42));
}

#[test]
fn test_to_array() {
    let result = execute_root(vec![Operation::ToArray], json!("Hello")).unwrap();
    assert_eq!(result, json!(["Hello"]));
}

#[test]
fn test_default_passes_through_non_null() {
    let result = execute_root(
        vec![Operation::Default {
            value: json!("fallback"),
        }],
        json!("original"),
    )
    .unwrap();
    assert_eq!(result, json!("original"));
}

#[test]
fn test_default_substitutes_null() {
    let result = execute_root(
        vec![Operation::Default {
            value: json!("fallback"),
        }],
        json!(null),
    )
    .unwrap();
    assert_eq!(result, json!("fallback"));
}

#[test]
fn test_empty_pipeline() {
    let input = json!({"name": "Alice"});
    let result = execute_root(vec![], input.clone()).unwrap();
    assert_eq!(result, input);
}
