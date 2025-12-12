mod common;
use std::path::PathBuf;

use uzu::session::{
    Session,
    config::{DecodingConfig, RunConfig, SpeculatorConfig},
    parameter::{
        AsyncBatchSize, ContextLength, ContextMode, PrefillStepSize,
        SamplingMethod, SamplingPolicy, SamplingSeed,
    },
    types::{Input, Message, Output},
};

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

fn build_decoding_config() -> DecodingConfig {
    DecodingConfig::new(
        ContextMode::default(),
        ContextLength::default(),
        PrefillStepSize::default(),
        SpeculatorConfig::default(),
        SamplingSeed::Custom(42),
        AsyncBatchSize::default(),
        true,
    )
}

fn request(
    session: &mut Session,
    input_text: String,
    tokens_limit: u64,
) -> String {
    session
        .run(
            Input::Text(input_text.clone()),
            RunConfig::new(
                tokens_limit,
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
        .trim()
        .to_string()
}

#[test]
fn test_context_mode_none() {
    let decoding_config =
        build_decoding_config().with_context_mode(ContextMode::None);
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();

    let tokens_limit = 48;
    let input_text = "What is Alice's occupation?".to_string();
    let response_1 = request(&mut session, input_text.clone(), tokens_limit);
    let response_2 = request(&mut session, input_text.clone(), tokens_limit);

    println!("response_1: {}", response_1);
    println!("response_2: {}", response_2);

    assert_eq!(response_1, response_2);
}

#[test]
fn test_context_mode_static() {
    let parameter_name = "Alice".to_string();
    let parameter_age = "30".to_string();
    let parameter_occupation = "Engineer".to_string();
    let parameter_city = "London".to_string();

    let decoding_config =
        build_decoding_config().with_context_mode(ContextMode::Static {
            input: Input::Messages(vec![
                Message::system(
                    "Always answer with one word or number, no punctuation"
                        .to_string(),
                ),
                Message::user(
                    format!(
                        "Name: {}\nAge: {}\nOccupation: {}\nCity: {}",
                        parameter_name,
                        parameter_age,
                        parameter_occupation,
                        parameter_city
                    )
                    .to_string(),
                ),
            ]),
        });
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();

    let tokens_limit = 48;
    let response_occupation = request(
        &mut session,
        format!("What is {}'s occupation?", parameter_name),
        tokens_limit,
    );
    let response_age = request(
        &mut session,
        format!("What is {}'s age?", parameter_name),
        tokens_limit,
    );
    let response_city = request(
        &mut session,
        format!("What is {}'s city?", parameter_name),
        tokens_limit,
    );

    println!("Occupation: {}", response_occupation);
    println!("Age: {}", response_age);
    println!("City: {}", response_city);

    assert!(
        response_occupation
            .to_lowercase()
            .contains(parameter_occupation.to_lowercase().as_str())
    );
    assert!(
        response_age
            .to_lowercase()
            .contains(parameter_age.to_lowercase().as_str())
    );
    assert!(
        response_city
            .to_lowercase()
            .contains(parameter_city.to_lowercase().as_str())
    );
}

#[test]
#[ignore = "Flaky test - depends on LLM output which varies even with fixed seed"]
fn test_context_mode_dynamic() {
    let decoding_config =
        build_decoding_config().with_context_mode(ContextMode::Dynamic);
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();

    let update = |session: &mut Session, input_text: String| {
        request(session, input_text, 0)
    };
    let ask = |session: &mut Session, input_text: String| {
        request(session, input_text, 48)
    };

    update(&mut session, "Name: Dave. Occupation: Nurse. Age: 30.".to_string());
    let answer_1 = ask(&mut session, "What is Dave's occupation?".to_string());
    println!("Answer 1: {}", answer_1);
    assert!(answer_1.to_lowercase().contains("nurse"));

    update(
        &mut session,
        "Update: Name: Dave. Occupation: Doctor. Age: 30.".to_string(),
    );
    let answer_2 = ask(&mut session, "What is Dave's occupation?".to_string());
    println!("Answer 2: {}", answer_2);
    assert!(answer_2.to_lowercase().contains("doctor"));

    update(
        &mut session,
        "Update: Name: Dave. Occupation: Doctor. Age: 31.".to_string(),
    );
    let answer_3 = ask(&mut session, "How old is Dave?".to_string());
    println!("Answer 3: {}", answer_3);
    assert!(answer_3.to_lowercase().contains("31"));

    update(
        &mut session,
        "Important update, Dave got promoted! Update: Name: Dave. Occupation: Director. Age: 32.".to_string(),
    );
    let answer_4 =
        ask(&mut session, "What is Dave's current occupation?".to_string());
    println!("Answer 4: {}", answer_4);
    assert!(answer_4.to_lowercase().contains("director"));

    let answer_5 = ask(
        &mut session,
        "At what age did the last promotion occur?".to_string(),
    );
    println!("Answer 5: {}", answer_5);
    assert!(answer_5.to_lowercase().contains("32"));
}

#[test]
fn test_context_mode_dynamic_scenario() {
    let decoding_config =
        build_decoding_config().with_context_mode(ContextMode::Dynamic);
    let mut session =
        Session::new(build_model_path(), decoding_config).unwrap();

    let user_prompts = vec![
        String::from("Tell about London"),
        String::from("Compare with New York"),
    ];

    println!("-------------------------");
    for user_prompt in user_prompts {
        let answer = request(&mut session, user_prompt.clone(), 1024);
        println!("{} -> {}", user_prompt, answer);
        println!("-------------------------");
    }
}
