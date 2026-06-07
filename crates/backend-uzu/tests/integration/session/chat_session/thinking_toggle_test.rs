#![cfg(metal_backend)]

use std::path::{Path, PathBuf};

use backend_uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig},
    parameter::SamplingSeed,
    types::{Input, Message, Output},
};
use test_tag::tag;

fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("THINKING_TEST_MODEL").ok()?);
    path.join("config.json").exists().then_some(path)
}

fn ask(
    path: &Path,
    prompt: &str,
    enable_thinking: bool,
) -> Output {
    let mut session =
        Session::new(path.to_path_buf(), DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42)))
            .expect("load model");
    session
        .run(
            Input::Messages(vec![Message::user(prompt.to_string())]),
            RunConfig::default().tokens_limit(256).enable_thinking(enable_thinking),
            Some(|_: Output| true),
        )
        .expect("run")
}

fn dump(
    label: &str,
    output: &Output,
) {
    println!("\n========== {label} ==========");
    println!("[raw generated]\n{}", output.text.original);
    println!("[chain_of_thought] {:?}", output.text.parsed.chain_of_thought);
    println!("[response] {:?}", output.text.parsed.response);
}

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded thinking model"]
#[tag(heavy)]
#[test]
fn thinking_toggle_end_to_end() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded thinking-model directory");
    };
    let prompt = "What is 17 + 25? Reply with just the number.";

    let on = ask(&path, prompt, true);
    dump("thinking ENABLED", &on);

    let off = ask(&path, prompt, false);
    dump("thinking DISABLED", &off);

    assert!(
        off.text.parsed.response.as_deref().is_some_and(|response| !response.trim().is_empty()),
        "disabled-thinking output must carry a non-empty response, got {:?}",
        off.text.parsed,
    );
    assert!(
        off.text.parsed.chain_of_thought.is_none(),
        "disabled-thinking output must not render a reasoning chain, got {:?}",
        off.text.parsed.chain_of_thought,
    );
}

#[ignore = "requires THINKING_TEST_MODEL pointing at a downloaded thinking model"]
#[tag(heavy)]
#[test]
fn thinking_toggle_within_one_session() {
    let Some(path) = model_path() else {
        panic!("set THINKING_TEST_MODEL to a downloaded thinking-model directory");
    };
    let mut session =
        Session::new(path, DecodingConfig::default().with_sampling_seed(SamplingSeed::Custom(42))).expect("load model");

    let turns = [
        ("What is 17 + 25? Reply with just the number.", true),
        ("And what is 8 + 9? Reply with just the number.", false),
        ("And what is 3 + 4? Reply with just the number.", true),
    ];

    let mut messages: Vec<Message> = Vec::new();
    for (prompt, enable_thinking) in turns {
        messages.push(Message::user(prompt.to_string()));
        let output = session
            .run(
                Input::Messages(messages.clone()),
                RunConfig::default().tokens_limit(256).enable_thinking(enable_thinking),
                Some(|_: Output| true),
            )
            .expect("run");
        dump(&format!("same-session turn (thinking={enable_thinking})"), &output);

        if !enable_thinking {
            assert!(
                output.text.parsed.chain_of_thought.is_none(),
                "disabled turn must not render reasoning, got {:?}",
                output.text.parsed.chain_of_thought,
            );
            assert!(
                output.text.parsed.response.as_deref().is_some_and(|response| !response.trim().is_empty()),
                "disabled turn must carry a non-empty response, got {:?}",
                output.text.parsed,
            );
        }

        messages.push(Message::assistant(
            output.text.parsed.response.clone().unwrap_or_default(),
            output.text.parsed.chain_of_thought.clone(),
        ));
    }
}
