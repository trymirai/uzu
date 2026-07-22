mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn qwen35_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("qwen3.5", "Qwen_Qwen3.5-0.8B")
}

#[test]
fn test_token_stream_parser_qwen35_tool_call_typed_arguments() {
    let suite = qwen35_suite();
    let data = TestData {
        prompt: "<|im_start|>user\nTemp?<|im_end|>\n<|im_start|>assistant\n".into(),
        completion: "<tool_call>\n<function=get_current_temperature>\n<parameter=latitude>\n51.5074\n</parameter>\n<parameter=longitude>\n-0.1278\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call><|im_end|>".into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "user", "content": [{"type": "$text", "value": "\nTemp?"}]},
                {"role": "assistant", "content": [
                    {"type": "$text", "value": "\n"},
                    {"type": "tool_call", "value": {
                        "name": "get_current_temperature",
                        "arguments": {
                            "latitude": 51.5074,
                            "longitude": -0.1278,
                            "days": 3,
                            "metric": true,
                            "unit": "celsius"
                        }
                    }}
                ]}
            ])),
        },
    );
}
