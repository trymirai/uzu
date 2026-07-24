#![cfg(not(target_family = "wasm"))]

use serde::Serialize;
use uzu::{
    engine::{Engine, EngineConfig},
    session::tool::{func_def::ToolFunctionDefinition, schema::UzuToolSchema, uzu_tool_function},
    types::{
        basic::{SamplingMethod, SamplingPolicy},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatRole},
    },
};

/// A geographic coordinate.
#[derive(Serialize, UzuToolSchema)]
struct Coordinate {
    /// Latitude in decimal degrees.
    latitude: f64,
    /// Longitude in decimal degrees.
    longitude: f64,
}

/// The current temperature at the requested location.
#[derive(Serialize, UzuToolSchema)]
struct Temperature {
    /// Current temperature in degrees Celsius.
    temperature: f64,
}

/// The current time.
#[derive(Serialize, UzuToolSchema)]
struct Time {
    /// Current time.
    time: String,
}

/// Get the user's current geographic location as latitude and longitude coordinates.
#[uzu_tool_function]
fn get_current_location() -> Coordinate {
    Coordinate {
        latitude: 59.938784,
        longitude: 30.314997,
    }
}

/// Get the current temperature at the given geographic coordinates.
#[uzu_tool_function]
fn get_current_temperature(
    /// Latitude in decimal degrees.
    latitude: f64,
    /// Longitude in decimal degrees.
    longitude: f64,
) -> Temperature {
    let _ = (latitude, longitude);
    Temperature {
        temperature: 25.9,
    }
}

/// Get the current time in 24h format.
#[uzu_tool_function]
fn get_current_time() -> Time {
    Time {
        time: "17:03".to_string(),
    }
}

struct TestCase {
    /// User prompt sent to the model.
    prompt: &'static str,
    /// Tools registered for this case.
    tools: &'static [fn() -> ToolFunctionDefinition],
    /// Tools that must be called (and produce results) during the turn.
    expected_tools: &'static [&'static str],
    /// Fragments that must appear in the final reply text.
    expected_fragments: &'static [&'static str],
}

/// Inputs and expected outputs that every model must pass.
const TEST_CASES: &[TestCase] = &[
    TestCase {
        prompt: "What is the time now?",
        tools: &[get_current_time::definition],
        expected_tools: &["get_current_time"],
        expected_fragments: &["17:03"],
    },
    TestCase {
        prompt: "What is the temperature at my current location?",
        tools: &[get_current_location::definition, get_current_temperature::definition],
        expected_tools: &["get_current_location", "get_current_temperature"],
        // "25" instead of "25.9": some models round the tool result when phrasing the reply
        // (gpt-oss-20b answers "25 °C"), and "25" still matches replies quoting "25.9" exactly.
        expected_fragments: &["25"],
    },
    TestCase {
        prompt: "What time is it now and what is the temperature at my current location?",
        tools: &[get_current_time::definition, get_current_location::definition, get_current_temperature::definition],
        expected_tools: &["get_current_time", "get_current_location", "get_current_temperature"],
        expected_fragments: &["17:03", "25"],
    },
];

async fn run_tool_calls_test(
    model_id: &str,
    with_system_message: bool,
    single_tool_call_per_turn: bool,
    cases: &[TestCase],
) {
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model =
        engine.model(model_id.to_string()).await.unwrap().unwrap_or_else(|| panic!("Model {model_id} not found"));
    println!("Loading {model_id}...");
    let download = engine.download(&model).await.unwrap();
    while download.next().await.is_some() {}

    for case in cases {
        println!("Prompt: {}", case.prompt);

        let mut session = engine.chat(model.clone(), ChatConfig::default()).await.unwrap();
        session
            .add_tool_function_definitions(case.tools.iter().map(|definition| definition()).collect())
            .await
            .unwrap();

        let mut messages = Vec::new();
        if with_system_message {
            messages.push(ChatMessage::system().with_text("You are a helpful assistant".to_string()));
        }
        messages.push(ChatMessage::user().with_text(case.prompt.to_string()));

        // Greedy sampling keeps these tests deterministic; with the default stochastic sampling
        // the models occasionally skip a tool call, hallucinate that part of the answer, or
        // finish without any text in the final reply.
        let reply_config = ChatReplyConfig {
            sampling_policy: SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {},
            },
            ..ChatReplyConfig::default()
        };
        let replies = session.reply(messages, reply_config).await.unwrap();
        let final_reply = replies.last().expect("Expected at least one reply");
        assert_eq!(
            final_reply.finish_reason,
            Some(ChatReplyFinishReason::Stop),
            "Expected the final reply to finish with Stop for prompt {:?}",
            case.prompt
        );
        let final_text = final_reply.message.text().unwrap_or_default();
        assert!(!final_text.is_empty(), "Expected the final reply to contain text for prompt {:?}", case.prompt);
        for fragment in case.expected_fragments {
            assert!(
                final_text.contains(fragment),
                "Expected the final reply for prompt {:?} to contain {fragment:?}, got: {final_text:?}",
                case.prompt
            );
        }

        let history = session.messages().await;
        let tool_calls = history
            .iter()
            .filter(|message| matches!(message.role, ChatRole::Assistant {}))
            .flat_map(|message| message.tool_calls())
            .collect::<Vec<_>>();
        let tool_call_results = history
            .iter()
            .filter(|message| matches!(message.role, ChatRole::Tool {}))
            .flat_map(|message| message.tool_call_results())
            .collect::<Vec<_>>();

        let called_names = tool_calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>();
        assert!(!tool_calls.is_empty(), "Expected at least one tool call for prompt {:?}", case.prompt);
        assert!(!tool_call_results.is_empty(), "Expected at least one tool call result for prompt {:?}", case.prompt);
        for expected in case.expected_tools {
            assert!(
                called_names.contains(expected),
                "Expected tool {expected} to be called for prompt {:?}, called tools: {called_names:?}",
                case.prompt
            );
            assert!(
                tool_call_results.iter().any(|(_, name, _)| name.as_deref() == Some(*expected)),
                "Expected a result for tool {expected} for prompt {:?}",
                case.prompt
            );
        }

        if single_tool_call_per_turn {
            for message in history.iter().filter(|message| matches!(message.role, ChatRole::Assistant {})) {
                assert!(
                    message.tool_calls().len() <= 1,
                    "Expected at most one tool call per assistant turn for prompt {:?}, got {:?}",
                    case.prompt,
                    message.tool_calls().iter().map(|call| call.name.clone()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[ignore]
#[tokio::test]
async fn functiongemma_270m_it() {
    // FunctionGemma 270M only handles single-step tool calls out of the box:
    // for the prompts that require chaining (get_current_location -> get_current_temperature) it invents coordinates or
    // asks the user for them instead of calling get_current_location first.
    // Per the model card, multi-step use cases require task-specific fine-tuning.
    run_tool_calls_test("google/functiongemma-270m-it", false, true, &TEST_CASES[..1]).await;
}

#[ignore]
#[tokio::test]
async fn gpt_oss_20b() {
    run_tool_calls_test("openai/gpt-oss-20b", true, false, TEST_CASES).await;
}

#[ignore]
#[tokio::test]
async fn lfm2_350m() {
    run_tool_calls_test("LiquidAI/LFM2-350M", true, false, &TEST_CASES[..1]).await;
}

#[ignore]
#[tokio::test]
async fn lfm2_5_350m() {
    run_tool_calls_test("LiquidAI/LFM2.5-350M", true, false, &TEST_CASES[..1]).await;
}

#[ignore]
#[tokio::test]
async fn llama_3_2_1b_instruct() {
    // Like FunctionGemma, Llama 3.2 1B only handles single-step tool calls: for the prompts that
    // require chaining (get_current_location -> get_current_temperature) it invents coordinates
    // (e.g. latitude "37") instead of calling get_current_location first.
    run_tool_calls_test("meta-llama/Llama-3.2-1B-Instruct", true, false, &TEST_CASES[..1]).await;
}

#[ignore]
#[tokio::test]
async fn qwen3_1_7b() {
    run_tool_calls_test("Qwen/Qwen3-1.7B", true, false, TEST_CASES).await;
}

#[ignore]
#[tokio::test]
async fn qwen3_5_0_8b() {
    run_tool_calls_test("Qwen/Qwen3.5-0.8B", true, false, TEST_CASES).await;
}
