mod helpers;

use helpers::{execute_root, get_key, get_path};
use json_transform::execution::{Operation, PathSegment};
use serde_json::json;

#[test]
fn test_get_from_object() {
    let result = execute_root(vec![get_key("name")], json!({"name": "Alice", "age": 30})).unwrap();
    assert_eq!(result, json!("Alice"));
}

#[test]
fn test_get_missing_key_returns_null() {
    let result = execute_root(vec![get_key("missing")], json!({"name": "Alice"})).unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_get_from_non_object_returns_null() {
    let result = execute_root(vec![get_key("name")], json!("Hello")).unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_get_path_traverses_nested() {
    let result = execute_root(
        vec![get_path(vec![PathSegment::Key("user".to_string()), PathSegment::Key("name".to_string())])],
        json!({"user": {"name": "Alice"}}),
    )
    .unwrap();
    assert_eq!(result, json!("Alice"));
}

#[test]
fn test_get_path_with_index() {
    let result = execute_root(
        vec![get_path(vec![
            PathSegment::Key("role".to_string()),
            PathSegment::Index(0),
            PathSegment::Key("$text".to_string()),
        ])],
        json!({"role": [{"$text": "assistant"}]}),
    )
    .unwrap();
    assert_eq!(result, json!("assistant"));
}

#[test]
fn test_get_path_missing_key_returns_null() {
    let result = execute_root(
        vec![get_path(vec![PathSegment::Key("user".to_string()), PathSegment::Key("missing".to_string())])],
        json!({"user": {"name": "Alice"}}),
    )
    .unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_get_path_index_out_of_bounds_returns_null() {
    let result = execute_root(
        vec![get_path(vec![PathSegment::Key("items".to_string()), PathSegment::Index(5)])],
        json!({"items": [1, 2]}),
    )
    .unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_get_path_single_key() {
    let result =
        execute_root(vec![get_path(vec![PathSegment::Key("name".to_string())])], json!({"name": "Alice"})).unwrap();
    assert_eq!(result, json!("Alice"));
}

#[test]
fn test_get_path_empty_returns_input() {
    let input = json!({"name": "Alice"});
    let result = execute_root(vec![get_path(vec![])], input.clone()).unwrap();
    assert_eq!(result, input);
}

#[test]
fn test_first_from_array() {
    let result = execute_root(vec![Operation::First], json!([1, 2, 3])).unwrap();
    assert_eq!(result, json!(1));
}

#[test]
fn test_first_from_empty_array_returns_null() {
    let result = execute_root(vec![Operation::First], json!([])).unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_first_from_non_array_returns_null() {
    let result = execute_root(vec![Operation::First], json!("Hello")).unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_pipeline_chaining() {
    let result = execute_root(vec![get_key("user"), get_key("name")], json!({"user": {"name": "Alice"}})).unwrap();
    assert_eq!(result, json!("Alice"));
}
