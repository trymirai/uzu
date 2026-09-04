#![cfg(all(target_os = "macos", feature = "backend-metal"))]

use std::{path::PathBuf, sync::Arc, time::Instant};

use hanashi::{
    Encoding as _,
    chat::{
        EncodingConfig,
        hanashi::{HanashiEncodingImpl, config::HanashiConfig, renderer::Renderer},
    },
};
use shoji::{
    traits::backend::chat_token::TokenStreamMetrics,
    types::{
        basic::{ReasoningEffort, ToolDescription, ToolFunction, ToolNamespace, Value},
        session::chat::{ChatContentBlock, ChatMessage},
    },
};
use tokenizers::Tokenizer;
use uzu_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState, stream::SamplingMethod},
    },
};

fn messages(padding_lines: usize) -> Vec<ChatMessage> {
    let mut user_text =
        "Use write_file to create config.json with exactly this one-line content, with no added spaces: \
        {\"compilerOptions\":{\"strict\":true,\"target\":\"ES2022\"}}"
            .to_string();
    if padding_lines != 0 {
        user_text = format!("Repository context:{}\n\n{user_text}", "\nlet item = 1;".repeat(padding_lines));
    }
    vec![
        ChatMessage::system()
            .with_text("You are a coding assistant. Use the tool as instructed. Keep responses brief.".to_string())
            .with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
            name: "functions".to_string(),
            description: None,
            tools: vec![ToolDescription::Function {
                tool_function: ToolFunction {
                    name: "write_file".to_string(),
                    description: "Write the exact string content to a file.".to_string(),
                    parameters: Some(Value {
                        json: serde_json::json!({
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"],
                            "additionalProperties": false,
                        })
                        .to_string(),
                    }),
                    return_definition: None,
                },
            }],
        }]),
        ChatMessage::user().with_text(user_text),
    ]
}

fn token_ids(encoding: &HanashiEncodingImpl) -> Vec<u64> {
    encoding.state().tokens.iter().map(|token| u64::from(token.id)).collect()
}

fn generate_tool_call(
    model: &LanguageModel<Metal>,
    state: &mut LanguageModelState<Metal>,
    encoding: &mut HanashiEncodingImpl,
    end_id: u64,
) {
    let input = token_ids(encoding);
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let started = Instant::now();
    let mut stream = model.stream(&input, state, options).unwrap();
    let mut completed = false;
    let mut generated = 0;
    for output in stream.by_ref().take(512) {
        let token = output.unwrap();
        encoding.decode(vec![u32::try_from(token).unwrap()]).unwrap();
        generated += 1;
        if token == end_id {
            completed = true;
            break;
        }
    }
    let metrics = stream.metrics().clone();
    drop(stream);
    assert!(completed, "model must complete the tool call within 512 output tokens");
    assert_eq!(state.tokens(), token_ids(encoding), "backend teardown must retain exactly the emitted transcript");
    println!(
        "initial_input={} generated={generated} elapsed_ms={} metrics={metrics:?}",
        input.len(),
        started.elapsed().as_millis(),
    );
}

fn generate_next(
    model: &LanguageModel<Metal>,
    state: &mut LanguageModelState<Metal>,
    input: &[u64],
    label: &str,
) -> (Vec<u64>, TokenStreamMetrics) {
    let mut expected_tokens = state.tokens().to_vec();
    expected_tokens.extend_from_slice(input);
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    let started = Instant::now();
    let mut stream = model.stream(input, state, options).unwrap();
    let mut output = Vec::new();
    let mut first_token_millis = None;
    for next in stream.by_ref().take(32) {
        let token = next.unwrap();
        output.push(token);
        first_token_millis.get_or_insert_with(|| started.elapsed().as_millis());
        if model.generation_config().stop_token_ids.iter().any(|stop| u64::from(*stop) == token) {
            break;
        }
    }
    let metrics = stream.metrics().clone();
    drop(stream);
    assert!(!output.is_empty());
    expected_tokens.extend_from_slice(&output);
    assert_eq!(state.tokens(), expected_tokens, "{label}: backend teardown changed the consumed token ledger");
    println!(
        "{label}: input={} output={} ttft_ms={first_token_millis:?} elapsed_ms={} metrics={metrics:?}",
        input.len(),
        output.len(),
        started.elapsed().as_millis(),
    );
    (output, metrics)
}

#[test]
#[ignore = "requires a local Qwen3.5-encoded model in UZU_CACHE_TEST_MODEL and an idle Metal GPU"]
fn qwen_continuation_matches_fresh_actual_transcript() {
    let model_path =
        PathBuf::from(std::env::var_os("UZU_CACHE_TEST_MODEL").expect(
            "set UZU_CACHE_TEST_MODEL to the native model directory before explicitly running this ignored test",
        ));
    let encodings: Vec<EncodingConfig> =
        serde_json::from_slice(&std::fs::read(model_path.join("encoding.json")).unwrap()).unwrap();
    assert!(
        encodings.iter().any(|encoding| matches!(
            encoding,
            EncodingConfig::Hanashi {
                config: HanashiConfig::Qwen35
            }
        )),
        "this test requires the bundled Qwen3.5 encoding"
    );
    let tokenizer = Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap());
    let end_id = u64::from(tokenizer.token_to_id("<|im_end|>").expect("model tokenizer must declare im_end"));
    let engine = Engine::<Metal>::new().unwrap();
    let model = engine.load_language_model(&model_path).unwrap();

    for padding_lines in [0, 800] {
        let mut encoding = HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer.clone()).unwrap();
        encoding.encode(messages(padding_lines)).unwrap();
        if padding_lines != 0 {
            assert!(encoding.state().tokens.len() >= 5000, "long case must exercise a repository-sized context");
        }
        let mut cached_state = model.create_empty_state(Some(16384), 42).unwrap();
        generate_tool_call(&model, &mut cached_state, &mut encoding, end_id);
        let actual_tokens = token_ids(&encoding);
        let actual_text = encoding.state().text();
        let mut history = encoding.state().messages.clone();
        let calls = history.last().unwrap().tool_calls();
        assert_eq!(calls.len(), 1, "fixture must generate one completed write_file call");
        assert_eq!(calls[0].name, "write_file");
        assert!(encoding.record_completion(history.clone()).unwrap());
        history.push(ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
            identifier: calls[0].identifier.clone(),
            name: Some(calls[0].name.clone()),
            value: Value {
                json: "\"Successfully wrote config.json.\"".to_string(),
            },
        }));
        let renderer = Renderer::new(HanashiConfig::Qwen35.resolve().unwrap().rendering);
        let canonical = renderer.render(&history, true, None, None, None).unwrap();
        assert!(!canonical.starts_with(&actual_text), "fixture must reproduce a parser/renderer prefix mismatch");

        let suffix: Vec<u64> = encoding
            .try_append(&history)
            .unwrap()
            .expect("completed tool turn should append")
            .into_iter()
            .map(u64::from)
            .collect();
        assert!(!suffix.is_empty());
        assert_eq!(
            &encoding.state().tokens[..actual_tokens.len()].iter().map(|token| u64::from(token.id)).collect::<Vec<_>>(),
            &actual_tokens
        );
        let mut exact_input = actual_tokens.clone();
        exact_input.extend_from_slice(&suffix);
        assert_eq!(token_ids(&encoding), exact_input);
        println!("padding_lines={padding_lines} cached_tokens={} suffix_tokens={}", actual_tokens.len(), suffix.len());

        let (cached_output, cached_metrics) = generate_next(&model, &mut cached_state, &suffix, "cached");
        assert_eq!(cached_metrics.num_tokens_prefilled, suffix.len());
        let mut fresh_state = model.create_empty_state(Some(16384), 42).unwrap();
        // The oracle is the exact sampled transcript plus suffix, never its lossy canonical rerendering.
        let (fresh_output, fresh_metrics) = generate_next(&model, &mut fresh_state, &exact_input, "fresh");
        assert_eq!(fresh_metrics.num_tokens_prefilled, exact_input.len());
        assert_eq!(cached_output, fresh_output, "cached continuation must match fresh inference on identical IDs");
    }
    println!("peak_memory_bytes={:?}", engine.peak_memory_usage());
}
