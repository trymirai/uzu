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

mod hermetic {
    use std::path::PathBuf;

    use token_stream_parser::{
        Parser,
        token_stream::{TokenStreamParser, TokenStreamParserConfig},
        types::Token,
    };

    // Fabricated tokens: the parser matches framing tokens by value, so no tokenizer is needed.
    fn feed(
        parser: &mut TokenStreamParser,
        text: &str,
        next_id: &mut u32,
    ) {
        const SPECIALS: [&str; 6] =
            ["<|begin_of_text|>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"];
        let mut rest = text;
        while !rest.is_empty() {
            let next = SPECIALS.iter().filter_map(|special| rest.find(special).map(|i| (i, special))).min();
            match next {
                Some((0, special)) => {
                    parser
                        .push(&Token {
                            id: *next_id,
                            value: special.to_string(),
                            is_special: true,
                        })
                        .unwrap();
                    *next_id += 1;
                    rest = &rest[special.len()..];
                },
                Some((index, _)) => {
                    push_text(parser, &rest[..index], next_id);
                    rest = &rest[index..];
                },
                None => {
                    push_text(parser, rest, next_id);
                    rest = "";
                },
            }
        }
    }

    fn push_text(
        parser: &mut TokenStreamParser,
        text: &str,
        next_id: &mut u32,
    ) {
        if text.is_empty() {
            return;
        }
        parser
            .push(&Token {
                id: *next_id,
                value: text.to_string(),
                is_special: false,
            })
            .unwrap();
        *next_id += 1;
    }

    fn parser() -> TokenStreamParser {
        let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("hanashi")
            .join("configs")
            .join("parsing")
            .join("muse-glimmer.json");
        let config: TokenStreamParserConfig =
            serde_json::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
        TokenStreamParser::new(config).unwrap()
    }

    #[test]
    fn unfinished_tool_call_does_not_parse_as_finished() {
        let mut parser = parser();
        let mut next_id = 0;
        feed(&mut parser, "<|start|>user<|message|>hi<|eot|><|start|>assistant", &mut next_id);
        // stream stops mid-call, right after the markup's first char
        feed(&mut parser, " to=bash<|message|><", &mut next_id);

        let assistant = &parser.state().value[1];
        let section = &assistant["content"][0];
        assert_eq!(section["type"], "tool_call");
        assert!(section["value"].get("arguments").is_none(), "an unfinished call must not expose arguments: {section}");
    }

    #[test]
    fn closed_tool_call_parses_arguments() {
        let mut parser = parser();
        let mut next_id = 0;
        feed(&mut parser, "<|start|>user<|message|>hi<|eot|><|start|>assistant", &mut next_id);
        feed(
            &mut parser,
            concat!(
                " to=bash<|message|>",
                "<atem:function_calls>\n<atem:invoke name=\"bash\">\n",
                "<atem:parameter name=\"command\">printenv</atem:parameter>\n",
                "</atem:invoke>\n</atem:function_calls><|eot|>"
            ),
            &mut next_id,
        );

        let assistant = &parser.state().value[1];
        let section = &assistant["content"][0];
        assert_eq!(section["value"]["name"], "bash");
        assert_eq!(section["value"]["arguments"]["command"], "printenv");
    }
}
