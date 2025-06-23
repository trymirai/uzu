mod common;
use std::path::PathBuf;

use uzu::{
    generator::config::{ContextLength, SamplingSeed},
    session::{
        session::Session,
        session_classification_feature::SessionClassificationFeature,
        session_config::{SessionConfig, SessionPreset, SessionRunConfig},
        session_input::SessionInput,
        session_output::SessionOutput,
    },
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

#[test]
#[ignore]
fn test_generation_base() {
    let text = String::from("Tell about London");
    let config = SessionConfig::new(
        SessionPreset::General,
        SamplingSeed::Default,
        ContextLength::Default,
    );
    run(text, config, 128);
}

#[test]
#[ignore]
fn test_generation_with_fixed_speculator() {
    // classifcation feature
    let feature = SessionClassificationFeature {
        name: String::from("sentiment"),
        values: vec![
            "Happy",
            "Sad",
            "Angry",
            "Fearful",
            "Surprised",
            "Disgusted",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    };

    let text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    let text = format!(
        "Text is: \"{}\". Choose {} from the list: {}. Answer with one word. Dont't add dot at the end.",
        text_to_detect_feature,
        feature.name,
        feature.values.join(", ")
    );

    let config = SessionConfig::new(
        SessionPreset::Classification(feature),
        SamplingSeed::Default,
        ContextLength::Default,
    );
    run(text, config, 32);
}

#[test]
#[ignore]
fn test_generation_with_prompt_lookup_speculator() {
    // summarization feature
    let text_to_summarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    let text = format!(
        "Text is: \"{}\". Write only summary itself.",
        text_to_summarize
    );
    let config = SessionConfig::new(
        SessionPreset::Summarization,
        SamplingSeed::Default,
        ContextLength::Default,
    );
    run(text, config, 256);
}

#[test]
#[ignore]
fn test_generation_with_ngram_general_speculator() {
    let text = String::from("Tell about London");
    let config = SessionConfig::new(
        SessionPreset::General,
        SamplingSeed::Default,
        ContextLength::Default,
    );
    run(text, config, 128);
}

fn run(
    text: String,
    config: SessionConfig,
    tokens_limit: u64,
) {
    let mut session = Session::new(build_model_path()).unwrap();
    session.load(config).unwrap();

    let input = SessionInput::Text(text);
    let output = session.run(
        input,
        SessionRunConfig::new(tokens_limit),
        Some(|_: SessionOutput| {
            return true;
        }),
    );

    println!("-------------------------");
    println!("{}", output.text);
    println!("-------------------------");
    println!("{:#?}", output.stats);
    println!("-------------------------");
    println!("Finish reason: {:?}", output.finish_reason);
    println!("-------------------------");
}
