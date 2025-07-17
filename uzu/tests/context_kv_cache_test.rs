use std::path::PathBuf;

use uzu::{
    backends::metal::sampling_config::SamplingConfig,
    context_registry::{ContextHandle, ContextRegistry},
    generator::config::{ContextLength, SamplingSeed, SpeculatorConfig},
    session::{
        session::Session,
        session_config::SessionConfig,
        session_input::SessionInput,
        session_output::SessionOutput,
        session_run_config::{RunMode, SessionRunConfig},
    },
};

mod common;

fn model_path() -> PathBuf {
    common::get_test_model_path()
}

#[test]
fn test_context_reuse() {
    let mut session = create_session();
    let registry = ContextRegistry::new();

    let system_prompt = "Name: Alice. Age: 30. Occupation: Engineer.";
    let handle = build_context(&mut session, &registry, system_prompt);
    let ctx_id = handle;

    let answer = ask_with_context(
        &mut session,
        &registry,
        &ctx_id,
        "What is Alice's occupation?",
    );

    println!("Answer: {}", answer);
    assert!(answer.to_lowercase().contains("engineer"));
}

#[test]
fn test_multiple_contexts() {
    let mut session = create_session();
    let registry = ContextRegistry::new();

    let ctx1 = build_context(
        &mut session,
        &registry,
        "Name: Alice. Occupation: Engineer.",
    );
    let id1 = ctx1;

    let ctx2 = build_context(
        &mut session,
        &registry,
        "Name: Alice. Occupation: Artist.",
    );
    let id2 = ctx2;

    let answer1 = ask_with_context(
        &mut session,
        &registry,
        &id1,
        "What is Alice's occupation?",
    );
    println!("Answer 1: {}", answer1);
    assert!(answer1.to_lowercase().contains("engineer"));

    let answer2 = ask_with_context(
        &mut session,
        &registry,
        &id2,
        "What is Alice's occupation?",
    );
    println!("Answer 2: {}", answer2);
    assert!(answer2.to_lowercase().contains("artist"));
}

#[test]
fn test_context_extension() {
    let mut session = create_session();
    let registry = ContextRegistry::new();
    let handle = build_context(
        &mut session,
        &registry,
        "Name: Dave. Occupation: Nurse.",
    );
    let id = handle;

    let answer_initial = ask_with_context(
        &mut session,
        &registry,
        &id,
        "What is Dave's occupation?",
    );
    assert!(answer_initial.to_lowercase().contains("nurse"));
    println!("Answer initial: {}", answer_initial);

    let cloned_gen =
        registry.get(&id).unwrap().read().unwrap().clone_generator();
    session.attach_generator(cloned_gen);
    let _ = session.run(
        SessionInput::Text("Update:Name: Dave. Occupation: Doctor.".into()),
        SessionRunConfig::new_with_sampling_config(
            1,
            Some(SamplingConfig::Argmax),
        )
        .with_run_mode(RunMode::WithPrefix),
        None::<fn(SessionOutput) -> bool>,
    );
    let new_gen = session.take_generator().unwrap();
    registry.remove(&id);
    let new_handle = ContextHandle::new(new_gen.tokens.clone(), new_gen);
    let new_id = registry.insert(new_handle);

    let answer_updated = ask_with_context(
        &mut session,
        &registry,
        &new_id,
        "What is Dave's occupation?",
    );
    println!("Answer updated: {}", answer_updated);
    assert!(answer_updated.to_lowercase().contains("doctor"));
}

#[test]
fn test_deep_copy_isolated() {
    let mut session = create_session();
    let registry = ContextRegistry::new();

    let handle = build_context(
        &mut session,
        &registry,
        "Name: Alice. Occupation: Singer.",
    );
    let id = handle;

    let answer_no_ctx1 =
        ask_without_context(&mut session, "What is Alice's occupation?");
    println!("Answer without context #1: {}", answer_no_ctx1);

    let answer_with_ctx = ask_with_context(
        &mut session,
        &registry,
        &id,
        "What is Alice's occupation?",
    );
    println!("Answer with context: {}", answer_with_ctx);

    let answer_no_ctx2 =
        ask_without_context(&mut session, "What is Alice's occupation?");
    println!("Answer without context #2: {}", answer_no_ctx2);

    assert_eq!(answer_no_ctx1, answer_no_ctx2);
    assert_ne!(answer_with_ctx, answer_no_ctx1);
    assert!(answer_with_ctx.to_lowercase().contains("singer"));
}

#[test]
fn test_performance_cached_vs_plain() {
    let mut session = create_session();
    let system_prompt = "
    Eve Smith is a talented 25-year-old professional dancer who lives in New York City. 
    She was born on March 15th, 1999, in a small town called Middlebury in Vermont to parents Robert and Sarah Smith. 
    Eve has beautiful emerald green eyes and long auburn hair that she often wears in a bun for performances. 
    She stands 5 feet 6 inches tall and weighs 125 pounds, with an athletic dancer's build.
    She graduated from Juilliard School with a Bachelor of Fine Arts in Dance in 2021, graduating summa cum laude. 
    Eve specializes in contemporary and ballet dance styles, though she also enjoys jazz, hip-hop, and modern dance. 
    She currently works with the Manhattan Dance Company as a principal dancer, earning $85,000 per year.
    Her favorite ballet is Swan Lake, and her favorite contemporary piece is Martha Graham's Appalachian Spring.
    In her free time, Eve loves to paint watercolor landscapes and has sold several pieces at local art galleries for $300-500 each. 
    She lives in a cozy 600-square-foot studio apartment in Brooklyn's Park Slope neighborhood, paying $2,800 per month in rent.
    She shares her apartment with her rescue cat named Luna, a 3-year-old gray tabby she adopted from the ASPCA.
    Eve is also passionate about teaching dance to underprivileged children and volunteers at the Downtown Community Center every Saturday morning from 9 AM to 12 PM.
    Her biggest dream is to choreograph her own full-length ballet production called 'Seasons of Change' someday.
    Eve drives a blue 2019 Honda Civic and her favorite coffee shop is Blue Bottle Coffee on 5th Avenue.
    She has a best friend named Maya Rodriguez who works as a photographer in Manhattan.
    Eve's favorite food is Thai green curry, and she's allergic to shellfish and peanuts.
    She practices yoga every Tuesday and Thursday at 7 PM at Dharma Yoga Center.
    Eve's phone number is 555-123-4567 and her email is eve.smith.dancer@gmail.com.
    She has performed in 12 professional productions and won the Young Artist Award in 2022.
    Eve's Instagram handle is @evesmithdancer with 15,000 followers.
    She speaks fluent English and conversational French, having studied abroad in Paris for one semester.
    Her favorite season is autumn, and her lucky number is 7.
    Eve wears size 7.5 shoes and prefers Bloch pointe shoes for ballet performances.
    She has a small scar on her left knee from a childhood bicycle accident.
    Every Sunday, she calls her grandmother Margaret who lives in a nursing home in Vermont. ";

    let questions = [
        "What is Eve's occupation?",
        "How old is Eve?",
        "What is Eve's surname?",
        "Where does Eve live?",
        "What color are Eve's eyes?",
        "What school did Eve graduate from?",
        "What dance styles does Eve specialize in?",
        "What company does she work for?",
        "What is her hobby besides dancing?",
        "What is her cat's name?",
        "What is Eve's height?",
        "How much does she earn per year?",
        "What is her favorite ballet?",
        "Where does she volunteer?",
        "What car does she drive?",
    ];

    use std::time::Instant;

    let start_cached = Instant::now();
    let registry = ContextRegistry::new();
    let handle = build_context(&mut session, &registry, system_prompt);
    let id = handle;

    let cached_answers: Vec<String> = questions
        .iter()
        .map(|q| ask_with_context(&mut session, &registry, &id, q))
        .collect();
    let duration_cached = start_cached.elapsed().as_secs_f64();

    let start_plain = Instant::now();
    let plain_answers: Vec<String> = questions
        .iter()
        .map(|q| {
            let prompt = format!("{} {}", system_prompt, q);
            ask_without_context(&mut session, &prompt)
        })
        .collect();
    let duration_plain = start_plain.elapsed().as_secs_f64();

    println!(
        "Plain answers: {:?}",
        plain_answers
            .iter()
            .map(|a| a
                .trim()
                .replace("<think>", "")
                .replace("</think>", "")
                .replace("\n", " "))
            .collect::<Vec<String>>()
    );
    println!(
        "Cached answers: {:?}",
        cached_answers
            .iter()
            .map(|a| a
                .trim()
                .replace("<think>", "")
                .replace("</think>", "")
                .replace("\n", " "))
            .collect::<Vec<String>>()
    );
    println!("Plain duration ({} q):  {:.3}s", questions.len(), duration_plain);
    println!(
        "Cached duration ({} q): {:.3}s",
        questions.len(),
        duration_cached
    );

    assert!(duration_cached <= duration_plain);
}

fn create_session() -> Session {
    let config = SessionConfig::new(
        8,
        SpeculatorConfig::default(),
        true,
        SamplingSeed::Custom(42),
        ContextLength::Default,
    );
    let mut session = Session::new(model_path()).unwrap();
    session.load_with_session_config(config).unwrap();
    session
}

fn build_context(
    session: &mut Session,
    registry: &ContextRegistry,
    prompt: &str,
) -> u64 {
    // Run the prompt once to build the prefix.
    let _ = session.run(
        SessionInput::Text(prompt.to_string()),
        SessionRunConfig::new_with_sampling_config(
            1,
            Some(SamplingConfig::Argmax),
        ),
        None::<fn(SessionOutput) -> bool>,
    );

    // Capture the context and get its ID.
    session.capture_context(registry)
}

fn ask_with_context(
    session: &mut Session,
    registry: &ContextRegistry,
    id: &u64,
    question: &str,
) -> String {
    let output = session.run_with_context(
        registry,
        id,
        SessionInput::Text(format!("{} /no_think", question)),
        SessionRunConfig::new_with_sampling_config(
            96,
            Some(SamplingConfig::Argmax),
        ),
        None::<fn(SessionOutput) -> bool>,
    );
    output.text
}

fn ask_without_context(
    session: &mut Session,
    question: &str,
) -> String {
    let output = session.run(
        SessionInput::Text(format!("{} /no_think", question)),
        SessionRunConfig::new_with_sampling_config(
            96,
            Some(SamplingConfig::Argmax),
        ),
        None::<fn(SessionOutput) -> bool>,
    );
    output.text
}
