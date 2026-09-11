mod helpers;

use helpers::{execute_root, get_key, init_tracing_for_tests};
use indexmap::IndexMap;
use json_transform::{
    TransformSchema,
    execution::{CallTarget, Condition, Operation, SwitchCase},
};
use serde_json::json;

// Switch

#[test]
fn test_switch_matches_first_case() {
    let result = execute_root(
        vec![Operation::Switch {
            key: vec![get_key("role")],
            cases: vec![
                SwitchCase {
                    when: Condition::Equals {
                        value: json!("admin"),
                    },
                    then: vec![Operation::Literal {
                        value: json!("is_admin"),
                    }],
                },
                SwitchCase {
                    when: Condition::Equals {
                        value: json!("user"),
                    },
                    then: vec![Operation::Literal {
                        value: json!("is_user"),
                    }],
                },
            ],
            default: None,
        }],
        json!({"role": "admin"}),
    )
    .unwrap();
    assert_eq!(result, json!("is_admin"));
}

#[test]
fn test_switch_falls_through_to_default() {
    let result = execute_root(
        vec![Operation::Switch {
            key: vec![],
            cases: vec![SwitchCase {
                when: Condition::Equals {
                    value: json!("x"),
                },
                then: vec![Operation::Literal {
                    value: json!("matched"),
                }],
            }],
            default: Some(vec![Operation::Literal {
                value: json!("default"),
            }]),
        }],
        json!("y"),
    )
    .unwrap();
    assert_eq!(result, json!("default"));
}

#[test]
fn test_switch_no_match_no_default_returns_null() {
    let result = execute_root(
        vec![Operation::Switch {
            key: vec![],
            cases: vec![],
            default: None,
        }],
        json!("anything"),
    )
    .unwrap();
    assert_eq!(result, json!(null));
}

#[test]
fn test_switch_starts_with_condition() {
    let result = execute_root(
        vec![Operation::Switch {
            key: vec![get_key("channel")],
            cases: vec![SwitchCase {
                when: Condition::StartsWith {
                    value: "commentary".to_string(),
                },
                then: vec![Operation::Literal {
                    value: json!("tool_call"),
                }],
            }],
            default: Some(vec![Operation::Literal {
                value: json!("text"),
            }]),
        }],
        json!({"channel": "commentary:func_name"}),
    )
    .unwrap();
    assert_eq!(result, json!("tool_call"));
}

// Call

#[test]
fn test_call_static_reference() {
    init_tracing_for_tests();
    let mut pipelines = IndexMap::new();
    pipelines.insert(
        "root".to_string(),
        vec![Operation::Call {
            target: CallTarget::Static {
                name: "get_name".to_string(),
            },
            arguments: IndexMap::new(),
        }],
    );
    pipelines.insert("get_name".to_string(), vec![get_key("name")]);
    let schema = TransformSchema {
        pipelines,
    };
    let result = schema.execute("root", json!({"name": "Alice"})).unwrap();
    assert_eq!(result, json!("Alice"));
}

#[test]
fn test_call_undefined_errors() {
    let result = execute_root(
        vec![Operation::Call {
            target: CallTarget::Static {
                name: "non-existent".to_string(),
            },
            arguments: IndexMap::new(),
        }],
        json!(null),
    );
    assert!(result.is_err());
}

#[test]
fn test_missing_pipeline_errors() {
    init_tracing_for_tests();
    let schema = TransformSchema {
        pipelines: IndexMap::new(),
    };
    let result = schema.execute("root", json!(null));
    assert!(result.is_err());
}
