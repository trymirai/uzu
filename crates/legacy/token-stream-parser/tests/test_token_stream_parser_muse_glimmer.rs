mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn muse_glimmer_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("muse-glimmer", "meta-models_Muse-Glimmer-30B")
}

#[test]
fn test_token_stream_parser_muse_glimmer_reasoning_and_reply() {
    let suite = muse_glimmer_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Hello<|eot|><|start|>assistant".into(),
        completion: " to=self<|message|>Thinking about it.<|eom|><|start|>assistant to=user<|message|>Hi there!<|eot|>"
            .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "Hello"}]},
                {"role": "assistant", "content": [
                    {"type": "reasoning", "value": "Thinking about it."},
                    {"type": "$text", "value": "Hi there!"}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_muse_glimmer_tool_call() {
    let suite = muse_glimmer_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Temp?<|eot|><|start|>assistant".into(),
        completion: concat!(
            " to=get_current_temperature<|message|>",
            "<atem:function_calls>\n<atem:invoke name=\"get_current_temperature\">\n",
            "<atem:parameter name=\"latitude\">51.5074</atem:parameter>\n",
            "<atem:parameter name=\"city\">Paris</atem:parameter>\n",
            "<atem:parameter name=\"days\">3</atem:parameter>\n",
            "<atem:parameter name=\"metric\">true</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        )
        .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "Temp?"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "value": {
                        "name": "get_current_temperature",
                        "arguments": {
                            "latitude": "51.5074",
                            "city": "Paris",
                            "days": "3",
                            "metric": "true"
                        }
                    }}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_muse_glimmer_preserves_ambiguous_string_arguments() {
    let suite = muse_glimmer_suite();
    let data = TestData {
        prompt: "<|start|>user<|message|>Look up IDs<|eot|><|start|>assistant".into(),
        completion: concat!(
            " to=lookup<|message|>",
            "<atem:function_calls>\n<atem:invoke name=\"lookup\">\n",
            "<atem:parameter name=\"padded_id\">00123</atem:parameter>\n",
            "<atem:parameter name=\"nullable_id\">null</atem:parameter>\n",
            "<atem:parameter name=\"metric\">true</atem:parameter>\n",
            "</atem:invoke>\n</atem:function_calls><|eom|>"
        )
        .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "Look up IDs"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "value": {
                        "name": "lookup",
                        "arguments": {
                            "padded_id": "00123",
                            "nullable_id": "null",
                            "metric": "true"
                        }
                    }}
                ]}
            ])),
        },
    );
}

#[test]
fn test_token_stream_parser_muse_glimmer_tool_output_and_reply() {
    let suite = muse_glimmer_suite();
    let data = TestData {
        prompt: concat!(
            "<|start|>user<|message|>Temp?<|eot|>",
            "<|start|>assistant to=get_current_temperature<|message|>",
            "<atem:function_calls>\n<atem:invoke name=\"get_current_temperature\">\n</atem:invoke>\n</atem:function_calls><|eom|>",
            "<|start|>tool get_current_temperature<|message|>",
            "<tool_output name=\"get_current_temperature\">\n{\"temp\": 17}\n</tool_output><|eot|>",
            "<|start|>assistant"
        )
        .into(),
        completion: " to=user<|message|>It is 17 degrees.<|eot|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "Temp?"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "value": {
                        "name": "get_current_temperature",
                        "arguments": {}
                    }}
                ]},
                {"role": "tool", "content": [
                    {"type": "tool_call_result", "value": {
                        "name": "get_current_temperature",
                        "value": {"temp": 17}
                    }}
                ]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "It is 17 degrees."}
                ]}
            ])),
        },
    );
}
