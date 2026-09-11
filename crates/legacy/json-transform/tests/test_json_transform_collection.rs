mod helpers;

use helpers::{execute_root, get_key};
use indexmap::IndexMap;
use json_transform::execution::{Condition, Operation};
use serde_json::json;

#[test]
fn test_each_maps_array() {
    let result = execute_root(
        vec![Operation::Each {
            apply: vec![get_key("name")],
        }],
        json!([{"name": "Alice"}, {"name": "Bob"}]),
    )
    .unwrap();
    assert_eq!(result, json!(["Alice", "Bob"]));
}

#[test]
fn test_each_on_non_array_returns_null() {
    let result = execute_root(
        vec![Operation::Each {
            apply: vec![],
        }],
        json!("Hello"),
    )
    .unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_flat_map_flattens_array_results() {
    let result = execute_root(
        vec![Operation::FlatMap {
            apply: vec![get_key("items")],
        }],
        json!([{"items": [1, 2]}, {"items": [3, 4]}]),
    )
    .unwrap();
    assert_eq!(result, json!([1, 2, 3, 4]));
}

#[test]
fn test_flat_map_passes_through_non_array_input() {
    let result = execute_root(
        vec![Operation::FlatMap {
            apply: vec![],
        }],
        json!("Hello"),
    )
    .unwrap();
    assert_eq!(result, json!("Hello"));
}

#[test]
fn test_filter_keeps_matching() {
    let result = execute_root(
        vec![Operation::Filter {
            condition: Condition::Field {
                key: "active".to_string(),
                condition: Box::new(Condition::Equals {
                    value: json!(true),
                }),
            },
        }],
        json!([
            {"name": "Alice", "active": true},
            {"name": "Bob", "active": false},
            {"name": "Carol", "active": true}
        ]),
    )
    .unwrap();
    assert_eq!(
        result,
        json!([
            {"name": "Alice", "active": true},
            {"name": "Carol", "active": true}
        ])
    );
}

#[test]
fn test_filter_with_not_condition() {
    let result = execute_root(
        vec![Operation::Filter {
            condition: Condition::Not {
                condition: Box::new(Condition::IsNull),
            },
        }],
        json!([1, null, 2, null, 3]),
    )
    .unwrap();
    assert_eq!(result, json!([1, 2, 3]));
}

#[test]
fn test_join_concatenates() {
    let result = execute_root(
        vec![Operation::Join {
            separator: ", ".to_string(),
        }],
        json!(["Alice", "Bob", "Carol"]),
    )
    .unwrap();
    assert_eq!(result, json!("Alice, Bob, Carol"));
}

#[test]
fn test_join_empty_separator() {
    let result = execute_root(
        vec![Operation::Join {
            separator: "".to_string(),
        }],
        json!(["a", "b", "c"]),
    )
    .unwrap();
    assert_eq!(result, json!("abc"));
}

#[test]
fn test_reduce() {
    let mut merge_fields = IndexMap::new();
    merge_fields.insert("role".to_string(), vec![Operation::First, get_key("role")]);
    merge_fields.insert(
        "content".to_string(),
        vec![
            Operation::Each {
                apply: vec![get_key("content")],
            },
            Operation::Join {
                separator: "".to_string(),
            },
        ],
    );

    let result = execute_root(
        vec![Operation::Reduce {
            key: vec![get_key("role")],
            r#if: None,
            then: vec![Operation::Object {
                fields: merge_fields,
                required: vec!["role".to_string()],
            }],
        }],
        json!([
            {"role": "assistant", "content": "Hello "},
            {"role": "assistant", "content": "World"},
            {"role": "user", "content": "Hi"}
        ]),
    )
    .unwrap();
    assert_eq!(
        result,
        json!([
            {"role": "assistant", "content": "Hello World"},
            {"role": "user", "content": "Hi"}
        ])
    );
}

#[test]
fn test_reduce_with_if() {
    let mut merge_fields = IndexMap::new();
    merge_fields.insert("role".to_string(), vec![Operation::First, get_key("role")]);
    merge_fields.insert(
        "content".to_string(),
        vec![
            Operation::Each {
                apply: vec![get_key("content")],
            },
            Operation::Join {
                separator: "".to_string(),
            },
        ],
    );

    let result = execute_root(
        vec![Operation::Reduce {
            key: vec![get_key("role")],
            r#if: Some(Condition::Equals {
                value: json!("assistant"),
            }),
            then: vec![Operation::Object {
                fields: merge_fields,
                required: vec!["role".to_string()],
            }],
        }],
        json!([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1 "},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q2"}
        ]),
    )
    .unwrap();
    assert_eq!(
        result,
        json!([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1 a2"},
            {"role": "user", "content": "q2"}
        ])
    );
}
