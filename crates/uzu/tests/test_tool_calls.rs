#![cfg(not(target_family = "wasm"))]

use uzu::{
    engine::{Engine, EngineConfig},
    session::tool::func_def::ToolFunctionDefinition,
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatRole},
};

fn get_tool_functions() -> Vec<ToolFunctionDefinition> {
    vec![
        ToolFunctionDefinition::new(
            "get_current_location".to_string(),
            "Get the user's current geographic location as latitude and longitude coordinates.".to_string(),
            None,
            Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude in decimal degrees."
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude in decimal degrees."
                        }
                    },
                    "required": ["latitude", "longitude"]
                })
                .into(),
            ),
            Box::new(|_args| {
                Box::pin(async {
                    Ok(serde_json::json!({
                        "latitude": 59.938784,
                        "longitude": 30.314997,
                    })
                    .into())
                })
            }),
        ),
        ToolFunctionDefinition::new(
            "get_current_temperature".to_string(),
            "Get the current temperature at the given geographic coordinates.".to_string(),
            Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude in decimal degrees."
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude in decimal degrees."
                        }
                    },
                    "required": ["latitude", "longitude"]
                })
                .into(),
            ),
            Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "temperature": {
                            "type": "number",
                            "description": "Current temperature in degrees Celsius."
                        }
                    },
                    "required": ["temperature"]
                })
                .into(),
            ),
            Box::new(|_args| {
                Box::pin(async {
                    Ok(serde_json::json!({
                        "temperature": 25.9,
                    })
                    .into())
                })
            }),
        ),
        ToolFunctionDefinition::new(
            "get_current_time".to_string(),
            "Get the current time.".to_string(),
            None,
            Some(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "Current time."
                        }
                    },
                    "required": ["time"]
                })
                .into(),
            ),
            Box::new(|_args| {
                Box::pin(async {
                    Ok(serde_json::json!({
                        "time": "17:03",
                    })
                    .into())
                })
            }),
        ),
    ]
}

async fn run_tool_calls_test(
    model_id: &str,
    with_system_message: bool,
    single_tool_call_per_turn: bool,
    prompt: &str,
    expected_tools: &[&str],
) {
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model =
        engine.model(model_id.to_string()).await.unwrap().unwrap_or_else(|| panic!("Model {model_id} not found"));
    println!("Loading {model_id}...");
    let download = engine.download(&model).await.unwrap();
    while download.next().await.is_some() {}

    let mut session = engine.chat(model, ChatConfig::default()).await.unwrap();
    session.add_tool_functions(get_tool_functions()).await.unwrap();

    let mut messages = Vec::new();
    if with_system_message {
        messages.push(ChatMessage::system().with_text("You are a helpful assistant".to_string()));
    }
    messages.push(ChatMessage::user().with_text(prompt.to_string()));

    let replies = session.reply(messages, ChatReplyConfig::default()).await.unwrap();
    let final_reply = replies.last().expect("Expected at least one reply");
    // println!("Reasoning: {}", final_reply.message.reasoning().unwrap_or_default());
    // println!("Text: {}", final_reply.message.text().unwrap_or_default());
    assert_eq!(
        final_reply.finish_reason,
        Some(ChatReplyFinishReason::Stop),
        "Expected the final reply to finish with Stop"
    );
    assert!(!final_reply.message.text().unwrap_or_default().is_empty(), "Expected the final reply to contain text");

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
    // println!("Tool calls: {called_names:?}");
    assert!(!tool_calls.is_empty(), "Expected at least one tool call in the chat history");
    assert!(!tool_call_results.is_empty(), "Expected at least one tool call result in the chat history");
    for expected in expected_tools {
        assert!(
            called_names.contains(expected),
            "Expected tool {expected} to be called, called tools: {called_names:?}"
        );
        assert!(
            tool_call_results.iter().any(|(_, name, _)| name.as_deref() == Some(*expected)),
            "Expected a result for tool {expected}"
        );
    }

    if single_tool_call_per_turn {
        for message in history.iter().filter(|message| matches!(message.role, ChatRole::Assistant {})) {
            assert!(
                message.tool_calls().len() <= 1,
                "Expected at most one tool call per assistant turn, got {:?}",
                message.tool_calls().iter().map(|call| call.name.clone()).collect::<Vec<_>>()
            );
        }
    }
}

#[ignore]
#[tokio::test]
async fn functiongemma_270m_it() {
    run_tool_calls_test("google/functiongemma-270m-it", false, true, "What is the time now?", &["get_current_time"])
        .await;
}

#[ignore]
#[tokio::test]
async fn gpt_oss_20b() {
    run_tool_calls_test(
        "openai/gpt-oss-20b",
        true,
        false,
        "What time is it now and what is the temperature at my current location?",
        &["get_current_time", "get_current_location", "get_current_temperature"],
    )
    .await;
}

#[ignore]
#[tokio::test]
async fn lfm2_350m() {
    run_tool_calls_test("LiquidAI/LFM2-350M", true, false, "What is the time now?", &["get_current_time"]).await;
}

#[ignore]
#[tokio::test]
async fn lfm2_5_350m() {
    run_tool_calls_test("LiquidAI/LFM2.5-350M", true, false, "What is the time now?", &["get_current_time"]).await;
}

#[ignore]
#[tokio::test]
async fn llama_3_2_1b_instruct() {
    run_tool_calls_test(
        "meta-llama/Llama-3.2-1B-Instruct",
        true,
        false,
        "What time is it now and what is the temperature at my current location?",
        &["get_current_time", "get_current_location", "get_current_temperature"],
    )
    .await;
}

#[ignore]
#[tokio::test]
async fn qwen3_1_7b() {
    run_tool_calls_test("Qwen/Qwen3-1.7B", true, false, "What is the time now?", &["get_current_time"]).await;
}

#[ignore]
#[tokio::test]
async fn qwen3_5_0_8b() {
    run_tool_calls_test(
        "Qwen/Qwen3.5-0.8B",
        true,
        false,
        "What time is it now and what is the temperature at my current location?",
        &["get_current_time", "get_current_location", "get_current_temperature"],
    )
    .await;
}
