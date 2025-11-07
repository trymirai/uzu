mod common;
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{ContextMode, DecodingConfig, RunConfig, SpeculatorConfig},
    parameter::{
        ContextLength, PrefillStepSize, SamplingMethod, SamplingPolicy,
        SamplingSeed,
    },
    types::{Input, Output},
};

fn model_path() -> PathBuf {
    common::get_test_model_path()
}

fn base_config() -> DecodingConfig {
    DecodingConfig::new(
        PrefillStepSize::default(),
        ContextLength::default(),
        SpeculatorConfig::default(),
        SamplingSeed::Custom(42),
        true,
        ContextMode::None,
    )
}

#[test]
fn context_mode_none_isolated() {
    let mut cfg = base_config();
    cfg.context_mode = ContextMode::None;
    let mut session = Session::new(model_path(), cfg).unwrap();

    let run_q = |s: &mut Session| {
        s.run(
            Input::Text("What is Alice's occupation?".to_string()),
            RunConfig::new(
                48,
                false,
                SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                },
            ),
            None::<fn(Output) -> bool>,
        )
        .unwrap()
        .text
        .original
    };

    let a1 = run_q(&mut session);
    let a2 = run_q(&mut session);
    println!("a1: {}", a1);
    println!("a2: {}", a2);
    assert_eq!(a1, a2);
}

#[test]
fn context_mode_static_shared_across_runs() {
    let mut cfg = base_config();
    cfg.context_mode = ContextMode::Static {
        input: Input::Text(
            "Name: Alice. Age: 30. Occupation: Engineer. City: London."
                .to_string(),
        ),
    };
    let mut session = Session::new(model_path(), cfg).unwrap();

    let ask = |s: &mut Session, q: &str| {
        s.run(
            Input::Text(q.to_string()),
            RunConfig::new(
                48,
                false,
                SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                },
            ),
            None::<fn(Output) -> bool>,
        )
        .unwrap()
        .text
        .original
    };

    let a1 = ask(&mut session, "What is Alice's occupation?");
    let a2 = ask(&mut session, "What is Alice's age?");
    let a3 = ask(&mut session, "What is Alice's city?");
    println!("a1: {}", a1);
    println!("a2: {}", a2);
    println!("a3: {}", a3);
    assert!(a1.to_lowercase().contains("engineer"));
    assert!(a2.to_lowercase().contains("30"));
    assert!(a3.to_lowercase().contains("london"));
}

#[test]
fn context_mode_dynamic_accumulates() {
    let mut cfg = base_config();
    cfg.context_mode = ContextMode::Dynamic;
    let mut session = Session::new(model_path(), cfg).unwrap();

    let run_text = |s: &mut Session, text: &str| {
        s.run(
            Input::Text(text.to_string()),
            RunConfig::new(
                0,
                false,
                SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                },
            ),
            None::<fn(Output) -> bool>,
        )
        .unwrap();
    };

    let ask = |s: &mut Session, q: &str| {
        s.run(
            Input::Text(q.to_string()),
            RunConfig::new(
                48,
                false,
                SamplingPolicy::Custom {
                    value: SamplingMethod::Greedy,
                },
            ),
            None::<fn(Output) -> bool>,
        )
        .unwrap()
        .text
        .original
    };

    run_text(&mut session, "Name: Dave. Occupation: Nurse. Age: 30.");
    let a1 = ask(&mut session, "What is Dave's occupation?");
    println!("a1: {}", a1);
    assert!(a1.to_lowercase().contains("nurse"));

    run_text(&mut session, "Update: Name: Dave. Occupation: Doctor. Age: 30.");
    let a2 = ask(&mut session, "What is Dave's occupation?");
    println!("a2: {}", a2);
    assert!(a2.to_lowercase().contains("doctor"));

    run_text(&mut session, "Update: Name: Dave. Occupation: Doctor. Age: 31.");
    let a3 = ask(&mut session, "How old is Dave?");
    println!("a3: {}", a3);
    assert!(a3.to_lowercase().contains("31"));

    run_text(
        &mut session,
        "Important update, Dave got promoted! Update: Name: Dave. Occupation: Director. Age: 32.",
    );
    let a4 = ask(&mut session, "What is Dave's current occupation?");
    println!("a4: {}", a4);
    assert!(a4.to_lowercase().contains("director"));

    let a5 = ask(&mut session, "At what age did Dave get promoted?");
    println!("a5: {}", a5);
    assert!(a5.to_lowercase().contains("32"));
}
