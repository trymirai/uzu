#![cfg(metal_backend)]

use std::sync::Arc;

use backend_uzu::{
    prelude::{FixedTokensSpeculator, PromptLookupSpeculator, SpeculatorConfig},
    session::{
        Session,
        config::{DecodingConfig, RunConfig},
        parameter::SamplingSeed,
        types::{Input, Message, Output},
    },
};
use shoji::types::basic::Feature;
use tokenizers::Tokenizer;

use crate::common::path::get_test_model_path;

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42))
}

fn build_default_text() -> String {
    String::from("Tell about London")
}

#[test]
fn test_text_session_base() {
    run(build_default_text(), build_decoding_config(), 128);
}

#[test]
fn test_text_session_with_prompt_lookup_speculator() {
    let number_of_speculated_tokens = 16 - 1;
    let speculator = PromptLookupSpeculator::new_with_params(3);
    let speculator_config = SpeculatorConfig {
        number_of_speculated_tokens,
        speculator: Arc::new(speculator),
    };
    let decoding_config = build_decoding_config().with_speculator_config(speculator_config);

    let text_to_summarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    let text = format!("Text is: \"{}\". Write only summary itself.", text_to_summarize);

    run(text, decoding_config, 256);
}

#[test]
fn test_text_session_with_fixed_speculator() {
    let tokenizer = Tokenizer::from_file(get_test_model_path().join("tokenizer.json")).unwrap();

    let feature = Feature {
        name: String::from("sentiment"),
        values: vec!["Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted"]
            .into_iter()
            .map(String::from)
            .collect(),
    };
    let proposals: Vec<Vec<u64>> = feature
        .values
        .iter()
        .map(|value| {
            tokenizer.encode(value.clone().as_str(), false).unwrap().get_ids().iter().map(|&id| id as u64).collect()
        })
        .collect();
    let speculator = FixedTokensSpeculator::new(proposals);
    let speculator_config = SpeculatorConfig {
        number_of_speculated_tokens: speculator.max_trie_nodes(),
        speculator: Arc::new(speculator),
    };
    let decoding_config = build_decoding_config().with_speculator_config(speculator_config);

    let text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    let text = format!(
        "Text is: \"{}\". Choose {} from the list: {}. Answer with one word. Dont't add dot at the end.",
        text_to_detect_feature,
        feature.name,
        feature.values.join(", ")
    );

    run(text, decoding_config, 32);
}

#[ignore]
#[test]
fn test_text_session_ngram_speculator_chat() {
    todo!("Implement test_text_session_ngram_speculator_chat")
}

#[test]
fn test_text_session_scenario() {
    let system_prompt = String::from("You are a helpful assistant.");
    let user_prompts = vec![String::from("Tell about London"), String::from("Compare with New York")];
    run_scenario(Some(system_prompt), user_prompts);
}

#[test]
fn test_text_session_stability() {
    let mut session = Session::new(get_test_model_path(), build_decoding_config()).unwrap();
    println!("Index | TTFT, s | Prompt t/s | Generate t/s");
    for index in 0..10 {
        let input = Input::Text(build_default_text());
        let output = session
            .run(
                input,
                RunConfig::default().tokens_limit(128),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        println!(
            "{:.5} | {:.5} | {:.5} | {:.5}",
            index,
            output.stats.prefill_stats.duration,
            output.stats.prefill_stats.processed_tokens_per_second,
            output.stats.generate_stats.unwrap().tokens_per_second
        );
    }
}

fn run(
    text: String,
    decoding_config: DecodingConfig,
    tokens_limit: u64,
) {
    let mut session = Session::new(get_test_model_path(), decoding_config).unwrap();
    let input = Input::Text(text);
    let output = session
        .run(
            input,
            RunConfig::default().tokens_limit(tokens_limit),
            Some(|_: Output| {
                return true;
            }),
        )
        .unwrap();

    let empty_response = String::from("None");

    println!("-------------------------");
    println!("{}", output.text.parsed.chain_of_thought.unwrap_or(empty_response.clone()));
    println!("-------------------------");
    println!("{}", output.text.parsed.response.unwrap_or(empty_response.clone()));
    println!("-------------------------");
    println!("{:#?}", output.stats);
    println!("-------------------------");
    println!("Finish reason: {:?}", output.finish_reason);
    println!("-------------------------");
}

fn run_scenario(
    system_prompt: Option<String>,
    user_prompts: Vec<String>,
) {
    let mut session = Session::new(get_test_model_path(), build_decoding_config()).unwrap();

    let mut messages: Vec<Message> = vec![];
    if let Some(system_prompt) = system_prompt {
        messages.push(Message::system(system_prompt.clone()));
        println!("System > {}", system_prompt.clone());
    }

    for user_prompt in user_prompts {
        messages.push(Message::user(user_prompt.clone()));
        println!("User > {}", user_prompt.clone());

        let input = Input::Messages(messages.clone());
        let output = session
            .run(
                input,
                RunConfig::default(),
                Some(|_: Output| {
                    return true;
                }),
            )
            .unwrap();
        messages.push(Message::assistant(
            output.text.parsed.response.clone().unwrap_or(String::new()),
            output.text.parsed.chain_of_thought.clone(),
        ));
        println!("Assistant > {}", output.text.original.clone());
    }
}
