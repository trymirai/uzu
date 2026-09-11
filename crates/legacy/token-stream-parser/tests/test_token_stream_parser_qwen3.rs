mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn qwen3_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("qwen3", "Qwen_Qwen3-0.6B")
}

#[test]
fn test_token_stream_parser_qwen3_text_response() {
    let suite = qwen3_suite();
    let data = TestData {
        prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n".into(),
        completion: "Hi!<|im_end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|im_start|>"},
                {"text": ["user", "\n", "Hello"]},
                {"marker": "<|im_end|>"},
                {"text": ["\n"]},
                {"marker": "<|im_start|>"},
                {"text": ["assistant", "\n", "Hi", "!"]},
                {"marker": "<|im_end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n", "Hello"]}}
                    ]}}
                ]}},
                {"frame": {"text": ["\n"]}},
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n", "Hi", "!"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\nHello"}]},
                {"role": "assistant", "content": [{"type": "$text", "value": "\nHi!"}]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_qwen3_thinking_response() {
    let suite = qwen3_suite();
    let data = TestData {
        prompt: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n".into(),
        completion: "<think>\nLet me think.\n</think>\n\nHello!<|im_end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|im_start|>"},
                {"text": ["user", "\n", "Hi"]},
                {"marker": "<|im_end|>"},
                {"text": ["\n"]},
                {"marker": "<|im_start|>"},
                {"text": ["assistant", "\n"]},
                {"marker": "<think>"},
                {"text": ["\n", "Let", " me", " think", ".\n"]},
                {"marker": "</think>"},
                {"text": ["\n\n", "Hello", "!"]},
                {"marker": "<|im_end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n", "Hi"]}}
                    ]}}
                ]}},
                {"frame": {"text": ["\n"]}},
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n"]}},
                        {"group": {"name": "reasoning", "open": "<think>", "close": "</think>", "sections": [
                            {"frame": {"text": ["\n", "Let", " me", " think", ".\n"]}}
                        ]}},
                        {"frame": {"text": ["\n\n", "Hello", "!"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\nHi"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n"},
                    {"type": "reasoning", "value": "\nLet me think.\n"},
                    {"type": "$text", "value": "\n\nHello!"}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_qwen3_tool_call() {
    let suite = qwen3_suite();
    let data = TestData {
        prompt: "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n".into(),
        completion: "<tool_call>\n{\"name\": \"get_temp\"}\n</tool_call><|im_end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|im_start|>"},
                {"text": ["user", "\n", "Weather", "?"]},
                {"marker": "<|im_end|>"},
                {"text": ["\n"]},
                {"marker": "<|im_start|>"},
                {"text": ["assistant", "\n"]},
                {"marker": "<tool_call>"},
                {"text": ["\n", "{\"", "name", "\":", " \"", "get", "_temp", "\"}\n"]},
                {"marker": "</tool_call>"},
                {"marker": "<|im_end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n", "Weather", "?"]}}
                    ]}}
                ]}},
                {"frame": {"text": ["\n"]}},
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n"]}},
                        {"group": {"name": "tool_call", "open": "<tool_call>", "close": "</tool_call>", "sections": [
                            {"frame": {"text": ["\n", "{\"", "name", "\":", " \"", "get", "_temp", "\"}\n"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\nWeather?"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n"},
                    {"type": "tool_call", "value": {"name": "get_temp"}}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_qwen3_thinking_tool_call() {
    let suite = qwen3_suite();
    let data = TestData {
        prompt: "<|im_start|>user\nTemp?<|im_end|>\n<|im_start|>assistant\n".into(),
        completion: "<think>\nNeed tool.\n</think>\n\n<tool_call>\n{\"name\": \"get_temp\"}\n</tool_call><|im_end|>"
            .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|im_start|>"},
                {"text": ["user", "\n", "Temp", "?"]},
                {"marker": "<|im_end|>"},
                {"text": ["\n"]},
                {"marker": "<|im_start|>"},
                {"text": ["assistant", "\n"]},
                {"marker": "<think>"},
                {"text": ["\n", "Need", " tool", ".\n"]},
                {"marker": "</think>"},
                {"text": ["\n\n"]},
                {"marker": "<tool_call>"},
                {"text": ["\n", "{\"", "name", "\":", " \"", "get", "_temp", "\"}\n"]},
                {"marker": "</tool_call>"},
                {"marker": "<|im_end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n", "Temp", "?"]}}
                    ]}}
                ]}},
                {"frame": {"text": ["\n"]}},
                {"group": {"name": "message", "open": "<|im_start|>", "close": "<|im_end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "content", "sections": [
                        {"frame": {"text": ["\n"]}},
                        {"group": {"name": "reasoning", "open": "<think>", "close": "</think>", "sections": [
                            {"frame": {"text": ["\n", "Need", " tool", ".\n"]}}
                        ]}},
                        {"frame": {"text": ["\n\n"]}},
                        {"group": {"name": "tool_call", "open": "<tool_call>", "close": "</tool_call>", "sections": [
                            {"frame": {"text": ["\n", "{\"", "name", "\":", " \"", "get", "_temp", "\"}\n"]}}
                        ]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\nTemp?"}]},
                {"role": "assistant", "content": [
                        {"type": "$text", "value": "\n"},
                        {"type": "reasoning", "value": "\nNeed tool.\n"},
                        {"type": "$text", "value": "\n\n"},
                        {"type": "tool_call", "value": {"name": "get_temp"}}
                    ]}
            ])),
        },
    );
}
