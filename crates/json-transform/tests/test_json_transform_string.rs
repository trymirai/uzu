mod helpers;

use helpers::{execute_root, schema_with_root};
use json_transform::{execution::Operation, regex::RegexEngine};
use serde_json::json;

// Format

#[test]
fn test_format_replaces_placeholder() {
    let result = execute_root(
        vec![Operation::Format {
            template: "Hello {}".to_string(),
        }],
        json!("World"),
    )
    .unwrap();
    assert_eq!(result, json!("Hello World"));
}

#[test]
fn test_format_empty_input() {
    let result = execute_root(
        vec![Operation::Format {
            template: "Before {} After".to_string(),
        }],
        json!(""),
    )
    .unwrap();
    assert_eq!(result, json!("Before  After"));
}

#[test]
fn test_format_no_placeholder() {
    let result = execute_root(
        vec![Operation::Format {
            template: "No placeholder".to_string(),
        }],
        json!("Input"),
    )
    .unwrap();
    assert_eq!(result, json!("No placeholder"));
}

#[test]
fn test_format_multiple_placeholders() {
    let result = execute_root(
        vec![Operation::Format {
            template: "{} and {}".to_string(),
        }],
        json!("X"),
    )
    .unwrap();
    assert_eq!(result, json!("X and X"));
}

#[test]
fn test_format_preserves_special_characters() {
    let result = execute_root(
        vec![Operation::Format {
            template: "Result: {}".to_string(),
        }],
        json!("a=\"b\" c='d'"),
    )
    .unwrap();
    assert_eq!(result, json!("Result: a=\"b\" c='d'"));
}

#[test]
fn test_format_preserves_newlines() {
    let result = execute_root(
        vec![Operation::Format {
            template: "Code: {}".to_string(),
        }],
        json!("line1\nline2"),
    )
    .unwrap();
    assert_eq!(result, json!("Code: line1\nline2"));
}

#[test]
fn test_format_inserts_input() {
    let result = execute_root(
        vec![Operation::Format {
            template: r#"{"name": "code_interpreter", "code": "{}"}"#.to_string(),
        }],
        json!("print('hello')"),
    )
    .unwrap();
    assert_eq!(result, json!(r#"{"name": "code_interpreter", "code": "print('hello')"}"#));
}

// RegexReplace

#[test]
fn test_regex_replace_simple_substitution() {
    let result = execute_root(
        vec![Operation::RegexReplace {
            pattern: "<escape>".to_string(),
            template: "\"".to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!("Hello <escape>World<escape>"),
    )
    .unwrap();
    assert_eq!(result, json!("Hello \"World\""));
}

#[test]
fn test_regex_replace_with_captures() {
    let result = execute_root(
        vec![Operation::RegexReplace {
            pattern: r"(\w+)\.call\(([\s\S]*)\)".to_string(),
            template: r#"{"name": "$1", "arguments": "$2"}"#.to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!("get_weather.call(city=London)"),
    )
    .unwrap();
    assert_eq!(result, json!(r#"{"name": "get_weather", "arguments": "city=London"}"#));
}

#[test]
fn test_regex_replace_substitutes_capture_groups() {
    let result = execute_root(
        vec![Operation::RegexReplace {
            pattern: r"(\w+)\((.*)\)".to_string(),
            template: r#"{"name": "$1", "arguments": "$2"}"#.to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!("get_temperature(city=London, unit=celsius)"),
    )
    .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
    assert_eq!(parsed, json!({"name": "get_temperature", "arguments": "city=London, unit=celsius"}));
}

// RegexFindAll

#[test]
fn test_regex_find_all_splits_lines() {
    let result = execute_root(
        vec![Operation::RegexFindAll {
            pattern: r"[^\n]+".to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!("line1\nline2\nline3"),
    )
    .unwrap();
    assert_eq!(result, json!(["line1", "line2", "line3"]));
}

#[test]
fn test_regex_find_all_capture_group() {
    let result = execute_root(
        vec![Operation::RegexFindAll {
            pattern: r#""name":\s*"(\w+)""#.to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!(r#"{"name": "Alice"} {"name": "Bob"}"#),
    )
    .unwrap();
    assert_eq!(result, json!(["Alice", "Bob"]));
}

#[test]
fn test_regex_find_all_extracts_capture_group() {
    let result = execute_root(
        vec![Operation::RegexFindAll {
            pattern: r"(\w+)\(.*?\)".to_string(),
            regex_engine: RegexEngine::Standard,
        }],
        json!("get_temperature(city=London)"),
    )
    .unwrap();
    assert_eq!(result, json!(["get_temperature"]));
}

#[test]
fn test_regex_find_all_then_each_parse() {
    let result = execute_root(
        vec![
            Operation::RegexFindAll {
                pattern: r"[^\n]+".to_string(),
                regex_engine: RegexEngine::Standard,
            },
            Operation::Each {
                apply: vec![Operation::ParseJson {
                    repair: false,
                }],
            },
        ],
        json!("{\"a\":1}\n{\"b\":2}"),
    )
    .unwrap();
    assert_eq!(result, json!([{"a": 1}, {"b": 2}]));
}

#[test]
fn test_regex_find_all_with_each_parse_json() {
    let result = execute_root(
        vec![
            Operation::RegexFindAll {
                pattern: r"([^;]+)".to_string(),
                regex_engine: RegexEngine::Standard,
            },
            Operation::Each {
                apply: vec![Operation::ParseJson {
                    repair: false,
                }],
            },
        ],
        json!(r#"{"city":"London"}; {"city":"Tokyo"}"#),
    )
    .unwrap();
    assert_eq!(result, json!([{"city": "London"}, {"city": "Tokyo"}]));
}

// ParseJson

#[test]
fn test_parse_json_strict() {
    let result = execute_root(
        vec![Operation::ParseJson {
            repair: false,
        }],
        json!("{\"name\":\"Alice\"}"),
    )
    .unwrap();
    assert_eq!(result, json!({"name": "Alice"}));
}

#[test]
fn test_parse_json_strict_invalid_errors() {
    let result = execute_root(
        vec![Operation::ParseJson {
            repair: false,
        }],
        json!("{invalid}"),
    );
    assert!(result.is_err());
}

#[test]
fn test_parse_json_with_repair() {
    let result = execute_root(
        vec![Operation::ParseJson {
            repair: true,
        }],
        json!("{\"name\":\"Alice\"}"),
    )
    .unwrap();
    assert_eq!(result, json!({"name": "Alice"}));
}

#[test]
fn test_parse_json_valid_object() {
    let result = execute_root(
        vec![Operation::ParseJson {
            repair: false,
        }],
        json!(r#"{"city": "London", "temperature": 42}"#),
    )
    .unwrap();
    assert_eq!(result, json!({"city": "London", "temperature": 42}));
}

#[test]
fn test_parse_json_strict_parses_unicode_emoji() {
    let result = execute_root(
        vec![Operation::ParseJson {
            repair: false,
        }],
        json!(r#"{"reaction": "🎉"}"#),
    )
    .unwrap();
    assert_eq!(result, json!({"reaction": "🎉"}));
}

// ParseJson, repair=true

fn validate_repair(
    input: &str,
    expected_json: &str,
) {
    let schema = schema_with_root(vec![Operation::ParseJson {
        repair: true,
    }]);
    let result = schema.execute("root", json!(input)).unwrap();
    let expected: serde_json::Value = serde_json::from_str(expected_json).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_repair_json_trailing_comma() {
    validate_repair(r#"{"a": 1, "b": 2,}"#, r#"{"a": 1, "b": 2}"#);
}

#[test]
fn test_repair_json_unbalanced_brackets() {
    validate_repair(r#"{"a": 1, "b": [1, 2"#, r#"{"a": 1, "b": [1, 2]}"#);
}

#[test]
fn test_repair_json_excess_braces() {
    validate_repair(
        r#"{"name": "test", "parameters": {"location": "London"}}}}"#,
        r#"{"name": "test", "parameters": {"location": "London"}}"#,
    );
}

#[test]
fn test_repair_json_converts_single_quotes_to_double() {
    validate_repair(r#"{'name': "London"}"#, r#"{"name": "London"}"#);
    validate_repair(r#"{"name": 'London'}"#, r#"{"name": "London"}"#);
    validate_repair(r#"{'name': 'London'}"#, r#"{"name": "London"}"#);
}

#[test]
fn test_repair_json_replaces_true_false_none() {
    validate_repair(
        "{'enabled': True, 'visible': False, 'description': None}",
        r#"{"enabled": true, "visible": false, "description": null}"#,
    );
}

#[test]
fn test_repair_json_handles_nested_dict() {
    validate_repair("{'location': {'city': 'London'}}", r#"{"location": {"city": "London"}}"#);
}

#[test]
fn test_repair_json_handles_three_levels_of_nesting() {
    validate_repair("{'model': {'params': {'layers': True}}}", r#"{"model": {"params": {"layers": true}}}"#);
}

#[test]
fn test_repair_json_handles_list_with_mixed_types() {
    validate_repair("{'tags': [1, 2, 'urgent']}", r#"{"tags": [1, 2, "urgent"]}"#);
}

#[test]
fn test_repair_json_handles_list_of_dicts() {
    validate_repair("{'users': [{'id': 1}, {'id': 2}]}", r#"{"users": [{"id": 1}, {"id": 2}]}"#);
}

#[test]
fn test_repair_json_handles_unicode_emoji() {
    validate_repair("{'reaction': '🎉'}", r#"{"reaction": "🎉"}"#);
}

#[test]
fn test_repair_json_handles_empty_dict() {
    validate_repair("{}", "{}");
}

#[test]
fn test_repair_json_preserves_numeric_values() {
    validate_repair("{'temperature': 42, 'humidity': 3.14}", r#"{"temperature": 42, "humidity": 3.14}"#);
}

#[test]
fn test_repair_json_handles_nested_dict_with_list_of_keywords() {
    validate_repair(
        "{'config': {'enabled': True, 'features': [False, None, True]}}",
        r#"{"config": {"enabled": true, "features": [false, null, true]}}"#,
    );
}

#[test]
fn test_repair_json_incomplete_then_parse() {
    validate_repair(r#"{"city": "London","#, r#"{"city": "London"}"#);
}

#[test]
fn test_repair_json_python_dict_to_json() {
    validate_repair("{'city': 'London'}", r#"{"city": "London"}"#);
}
