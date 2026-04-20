mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn llama3_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("llama-3", "meta-llama_Llama-3.2-1B-Instruct")
}

#[test]
fn test_token_stream_parser_llama3_text_response() {
    let suite = llama3_suite();
    let data = TestData {
        prompt: "<|start_header_id|>user<|end_header_id|>\n\nWhat is the weather?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        completion: "I cannot check the weather for you.<|eot_id|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start_header_id|>"},
                {"text": ["user"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "What", " is", " the", " weather", "?"]},
                {"marker": "<|eot_id|>"},
                {"marker": "<|start_header_id|>"},
                {"text": ["assistant"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "I", " cannot", " check", " the", " weather", " for", " you", "."]},
                {"marker": "<|eot_id|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "What", " is", " the", " weather", "?"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "I", " cannot", " check", " the", " weather", " for", " you", "."]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {
                    "role": "user",
                    "content": [{"type": "$text", "value": "\n\nWhat is the weather?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "$text", "value": "\n\nI cannot check the weather for you."}]
                }
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_llama3_tool_call() {
    let suite = llama3_suite();
    let data = TestData {
        prompt: "<|start_header_id|>user<|end_header_id|>\n\nWhat is the temperature?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        completion: "<|python_tag|>{\"name\": \"get_current_temperature\", \"parameters\": {\"location\": {\"city\": \"London\", \"country\": \"UK\"}, \"unit\": \"celsius\"}}<|eom_id|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(r#"{"sections": [
                {"marker": "<|start_header_id|>"},
                {"text": ["user"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "What", " is", " the", " temperature", "?"]},
                {"marker": "<|eot_id|>"},
                {"marker": "<|start_header_id|>"},
                {"text": ["assistant"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n"]},
                {"marker": "<|python_tag|>"},
                {"text": ["{\"", "name", "\":", " \"", "get", "_current", "_temperature", "\",", " \"", "parameters", "\":", " {\"", "location", "\":", " {\"", "city", "\":", " \"", "London", "\",", " \"", "country", "\":", " \"", "UK", "\"},", " \"", "unit", "\":", " \"", "c", "elsius", "\"}}"]},
                {"marker": "<|eom_id|>"}
            ]}"#)),
            reduction: Some(suite.expect_reduction(r#"{"sections": [
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "What", " is", " the", " temperature", "?"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eom_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n"]}},
                        {"group": {"name": "tool_call", "open": "<|python_tag|>", "sections": [
                            {"frame": {"text": ["{\"", "name", "\":", " \"", "get", "_current", "_temperature", "\",", " \"", "parameters", "\":", " {\"", "location", "\":", " {\"", "city", "\":", " \"", "London", "\",", " \"", "country", "\":", " \"", "UK", "\"},", " \"", "unit", "\":", " \"", "c", "elsius", "\"}}"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#)),
            extraction: suite.expect_extraction(json!([
                {
                    "role": "user",
                    "content": [{"type": "$text", "value": "\n\nWhat is the temperature?"}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "$text", "value": "\n\n"},
                        {"type": "tool_call", "value": {"name": "get_current_temperature", "arguments": {"location": {"city": "London", "country": "UK"}, "unit": "celsius"}}}
                    ]
                }
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_llama3_multiple_tool_calls() {
    let suite = llama3_suite();
    let data = TestData {
        prompt: "<|start_header_id|>user<|end_header_id|>\n\nCompare temps<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        completion: "<|python_tag|>{\"name\": \"get_temp\", \"parameters\": {\"city\": \"London\"}}; {\"name\": \"get_temp\", \"parameters\": {\"city\": \"NYC\"}}<|eom_id|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(r#"{"sections": [
                {"marker": "<|start_header_id|>"},
                {"text": ["user"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "Compare", " temps"]},
                {"marker": "<|eot_id|>"},
                {"marker": "<|start_header_id|>"},
                {"text": ["assistant"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n"]},
                {"marker": "<|python_tag|>"},
                {"text": ["{\"", "name", "\":", " \"", "get", "_temp", "\",", " \"", "parameters", "\":", " {\"", "city", "\":", " \"", "London", "\"}}", ";", " {\"", "name", "\":", " \"", "get", "_temp", "\",", " \"", "parameters", "\":", " {\"", "city", "\":", " \"", "NY", "C", "\"}}"]},
                {"marker": "<|eom_id|>"}
            ]}"#)),
            reduction: Some(suite.expect_reduction(r#"{"sections": [
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "Compare", " temps"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eom_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n"]}},
                        {"group": {"name": "tool_call", "open": "<|python_tag|>", "sections": [
                            {"frame": {"text": ["{\"", "name", "\":", " \"", "get", "_temp", "\",", " \"", "parameters", "\":", " {\"", "city", "\":", " \"", "London", "\"}}", ";", " {\"", "name", "\":", " \"", "get", "_temp", "\",", " \"", "parameters", "\":", " {\"", "city", "\":", " \"", "NY", "C", "\"}}"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#)),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\n\nCompare temps"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n\n"},
                    {"type": "tool_call", "value": {"name": "get_temp", "arguments": {"city": "London"}}},
                    {"type": "tool_call", "value": {"name": "get_temp", "arguments": {"city": "NYC"}}}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_llama3_builtin_tool_call() {
    let suite = llama3_suite();
    let data = TestData {
        prompt: "<|start_header_id|>user<|end_header_id|>\n\nWeather?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        completion: "<|python_tag|>brave_search.call(query=\"weather london\")<|eom_id|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(r#"{"sections": [
                {"marker": "<|start_header_id|>"},
                {"text": ["user"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "Weather", "?"]},
                {"marker": "<|eot_id|>"},
                {"marker": "<|start_header_id|>"},
                {"text": ["assistant"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n"]},
                {"marker": "<|python_tag|>"},
                {"text": ["br", "ave", "_search", ".call", "(query", "=\"", "weather", " london", "\")"]},
                {"marker": "<|eom_id|>"}
            ]}"#)),
            reduction: Some(suite.expect_reduction(r#"{"sections": [
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "Weather", "?"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eom_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n"]}},
                        {"group": {"name": "tool_call", "open": "<|python_tag|>", "sections": [
                            {"frame": {"text": ["br", "ave", "_search", ".call", "(query", "=\"", "weather", " london", "\")"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#)),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\n\nWeather?"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n\n"},
                    {"type": "tool_call", "value": {"name": "brave_search", "arguments": {"query": "weather london"}}}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_llama3_code_interpreter() {
    let suite = llama3_suite();
    let data = TestData {
        prompt: "<|start_header_id|>user<|end_header_id|>\n\nCalculate<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        completion: "<|python_tag|>print(2 + 2)<|eom_id|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start_header_id|>"},
                {"text": ["user"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n", "Calculate"]},
                {"marker": "<|eot_id|>"},
                {"marker": "<|start_header_id|>"},
                {"text": ["assistant"]},
                {"marker": "<|end_header_id|>"},
                {"text": ["\n\n"]},
                {"marker": "<|python_tag|>"},
                {"text": ["print", "(", "2", " +", " ", "2", ")"]},
                {"marker": "<|eom_id|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eot_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n", "Calculate"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start_header_id|>", "close": "<|eom_id|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|end_header_id|>", "sections": [
                        {"frame": {"text": ["\n\n"]}},
                        {"group": {"name": "tool_call", "open": "<|python_tag|>", "sections": [
                            {"frame": {"text": ["print", "(", "2", " +", " ", "2", ")"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\n\nCalculate"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n\n"},
                    {"type": "tool_call", "value": {"name": "code_interpreter", "arguments": {"code": "print(2 + 2)"}}}
                ]}
            ])),
        },
    );
}
