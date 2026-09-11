mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn gpt_oss_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("gpt-oss", "openai_gpt-oss-20b")
}

#[test]
fn test_token_stream_parser_gpt_oss_text_response() {
    let suite = gpt_oss_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Hi<|end|><|start|>assistant".into(),
        completion: "<|channel|>final<|message|>Hello!<|end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start|>"},
                {"text": ["user"]},
                {"marker": "<|message|>"},
                {"text": ["Hi"]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["final"]},
                {"marker": "<|message|>"},
                {"text": ["Hello", "!"]},
                {"marker": "<|end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Hi"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["final"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Hello", "!"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [{"type": "$text", "value": "Hello!"}]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_gpt_oss_analysis_response() {
    let suite = gpt_oss_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Hi<|end|><|start|>assistant".into(),
        completion: "<|channel|>analysis<|message|>Let me think.<|end|><|start|>assistant<|channel|>final<|message|>Done!<|end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start|>"},
                {"text": ["user"]},
                {"marker": "<|message|>"},
                {"text": ["Hi"]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["analysis"]},
                {"marker": "<|message|>"},
                {"text": ["Let", " me", " think", "."]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["final"]},
                {"marker": "<|message|>"},
                {"text": ["Done", "!"]},
                {"marker": "<|end|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Hi"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["analysis"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Let", " me", " think", "."]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["final"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Done", "!"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "reasoning", "value": "Let me think."},
                    {"type": "$text", "value": "Done!"}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_gpt_oss_tool_call() {
    let suite = gpt_oss_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Hi<|end|><|start|>assistant".into(),
        completion:
            "<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"city\":\"NYC\"}<|call|>"
                .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start|>"},
                {"text": ["user"]},
                {"marker": "<|message|>"},
                {"text": ["Hi"]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["comment", "ary", " to", "=", "functions", ".get", "_weather"]},
                {"marker": "<|constrain|>"},
                {"text": ["json"]},
                {"marker": "<|message|>"},
                {"text": ["{\"", "city", "\":\"", "NY", "C", "\"}"]},
                {"marker": "<|call|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Hi"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|call|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["comment", "ary", " to", "=", "functions", ".get", "_weather"]}}
                    ]}},
                    {"group": {"name": "content_type", "open": "<|constrain|>", "sections": [
                        {"frame": {"text": ["json"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["{\"", "city", "\":\"", "NY", "C", "\"}"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "value": {"name": "get_weather", "arguments": {"city": "NYC"}}}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_gpt_oss_preamble_tool_call() {
    let suite = gpt_oss_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Do it<|end|><|start|>assistant".into(),
        completion: "<|channel|>commentary<|message|>Let me plan.<|end|><|start|>assistant<|channel|>commentary to=functions.do_step<|constrain|>json<|message|>{\"step\":1}<|call|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: Some(suite.expect_framing(
                r#"{"sections": [
                {"marker": "<|start|>"},
                {"text": ["user"]},
                {"marker": "<|message|>"},
                {"text": ["Do", " it"]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["comment", "ary"]},
                {"marker": "<|message|>"},
                {"text": ["Let", " me", " plan", "."]},
                {"marker": "<|end|>"},
                {"marker": "<|start|>"},
                {"text": ["assistant"]},
                {"marker": "<|channel|>"},
                {"text": ["comment", "ary", " to", "=", "functions", ".do", "_step"]},
                {"marker": "<|constrain|>"},
                {"text": ["json"]},
                {"marker": "<|message|>"},
                {"text": ["{\"", "step", "\":", "1", "}"]},
                {"marker": "<|call|>"}
            ]}"#,
            )),
            reduction: Some(suite.expect_reduction(
                r#"{"sections": [
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["user"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Do", " it"]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|end|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["comment", "ary"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["Let", " me", " plan", "."]}}
                    ]}}
                ]}},
                {"group": {"name": "message", "open": "<|start|>", "close": "<|call|>", "sections": [
                    {"group": {"name": "role", "sections": [
                        {"frame": {"text": ["assistant"]}}
                    ]}},
                    {"group": {"name": "channel", "open": "<|channel|>", "sections": [
                        {"frame": {"text": ["comment", "ary", " to", "=", "functions", ".do", "_step"]}}
                    ]}},
                    {"group": {"name": "content_type", "open": "<|constrain|>", "sections": [
                        {"frame": {"text": ["json"]}}
                    ]}},
                    {"group": {"name": "content", "open": "<|message|>", "sections": [
                        {"frame": {"text": ["{\"", "step", "\":", "1", "}"]}}
                    ]}}
                ]}}
            ]}"#,
            )),
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": "Do it"},
                {"role": "assistant", "content": [
                    {"type": "reasoning", "value": "Let me plan."},
                    {"type": "tool_call", "value": {"name": "do_step", "arguments": {"step": 1}}}
                ]}
            ])),
        },
    );
}
