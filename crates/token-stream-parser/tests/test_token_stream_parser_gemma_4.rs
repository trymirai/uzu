mod helpers;

use helpers::{TestData, TestExpectations, TestSuite, init_tracing_for_tests, run_parser_test};
use serde_json::json;

fn gemma_4_suite() -> TestSuite {
    init_tracing_for_tests();
    TestSuite::load("gemma-4", "google_gemma-4-E4B-it")
}

#[test]
fn test_token_stream_parser_gemma_4_thinking_tool_call_tool_response() {
    let suite = gemma_4_suite();
    let data = TestData {
        prompt: "<bos><|turn>system\n<|think|>You are a helpful assistant<|tool>declaration:get_current_temperature{description:<|\"|>Get the current temperature at a location.<|\"|>,parameters:{properties:{location:{description:<|\"|>The location to get the temperature for<|\"|>,properties:{city:{description:<|\"|>The city name<|\"|>,type:<|\"|>STRING<|\"|>},country:{description:<|\"|>The country name<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>,<|\"|>country<|\"|>],type:<|\"|>OBJECT<|\"|>},unit:{description:<|\"|>The unit to return the temperature in.<|\"|>,enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>location<|\"|>,<|\"|>unit<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nWhat is the current temperature in London, UK in celsius?<turn|>\n<|turn>model\n"
            .into(),
        completion:
            "<|channel>thought\nThinking Process:\n\nThe user is asking for the current temperature in London.<channel|><|tool_call>call:get_current_temperature{location:{city:<|\"|>London<|\"|>,country:<|\"|>United Kingdom<|\"|>},unit:<|\"|>celsius<|\"|>}<tool_call|><|tool_response>response:get_current_temperature{temperature:15}<tool_response|>The current weather in London is 15 celsius.<turn|>\n"
                .into(),
    };

    run_parser_test(
        &suite,
        &data,
        &TestExpectations {
            framing: None,
            reduction: None,
            extraction: suite.expect_extraction(json!([
                {"role": "system", "content": [
                    {"type": "$text", "value": "\n<|think|>You are a helpful assistant<|tool>declaration:get_current_temperature{description:<|\"|>Get the current temperature at a location.<|\"|>,parameters:{properties:{location:{description:<|\"|>The location to get the temperature for<|\"|>,properties:{city:{description:<|\"|>The city name<|\"|>,type:<|\"|>STRING<|\"|>},country:{description:<|\"|>The country name<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>,<|\"|>country<|\"|>],type:<|\"|>OBJECT<|\"|>},unit:{description:<|\"|>The unit to return the temperature in.<|\"|>,enum:[<|\"|>celsius<|\"|>,<|\"|>fahrenheit<|\"|>],type:<|\"|>STRING<|\"|>}},required:[<|\"|>location<|\"|>,<|\"|>unit<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"}
                ]},
                {"role": "user", "content": [{"type": "$text", "value": "\nWhat is the current temperature in London, UK in celsius?"}]},
                {"role": "model", "content": [
                    {"type": "$text", "value": "\n"},
                    {"type": "reasoning", "value": "\nThinking Process:\n\nThe user is asking for the current temperature in London."},
                    {"type": "tool_call", "value": {
                        "name": "get_current_temperature",
                        "arguments": {
                            "location": {
                                "city": "London",
                                "country": "United Kingdom"
                            },
                            "unit": "celsius"
                        }
                    }},
                    {"type": "tool_call_result", "value": {
                        "name": "get_current_temperature",
                        "value": {
                            "temperature": 15
                        }
                    }},
                    {"type": "$text", "value": "The current weather in London is 15 celsius."}
                ]},
            ])),
        },
    );
}
