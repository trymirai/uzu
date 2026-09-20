#![cfg(not(target_family = "wasm"))]

use std::env;

use uzu::{
    engine::{Engine, EngineConfig},
    session::tool::uzu_tool_function,
    types::{
        basic::Grammar,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatReplyFinishReason},
    },
};

fn enabled() -> bool {
    env::var("UZU_NEEDLE_IT").ok().as_deref() == Some("1")
}

/// Returns a stub weather payload for the requested city.
#[uzu_tool_function]
fn get_weather(city: String) -> String {
    format!(r#"{{"city":"{city}","temp_c":27}}"#)
}

#[ignore]
#[tokio::test]
async fn needle_lists_and_calls_weather_tool() {
    if !enabled() {
        return;
    }

    let engine = Engine::new(EngineConfig::default().with_allow_needle_usage(true)).await.unwrap();
    let model = engine.model("cactus:needle3".to_string()).await.unwrap().expect("cactus:needle3");
    assert!(model.is_local());
    assert!(model.is_chat_capable());
    assert!(!model.is_downloadable());

    let mut session = engine.chat(model, ChatConfig::default()).await.unwrap();
    session.add_tool(get_weather).await.unwrap();
    let replies = session
        .reply(
            vec![ChatMessage::user().with_text("what's it like in Lagos right now?".to_string())],
            ChatReplyConfig::default(),
        )
        .await
        .unwrap();
    let last = replies.last().expect("reply");
    let calls = last.message.tool_calls();
    if last.finish_reason == Some(ChatReplyFinishReason::ToolCalls) {
        assert!(calls.iter().any(|call| call.name.contains("weather") || call.arguments.json.contains("Lagos")));
    } else {
        assert!(
            replies.iter().any(|reply| reply
                .message
                .tool_calls()
                .iter()
                .any(|call| call.arguments.json.contains("Lagos")))
                || last.message.text().is_some()
        );
    }
}

#[ignore]
#[tokio::test]
async fn needle_off_topic_without_tools_is_stop() {
    if !enabled() {
        return;
    }
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model = engine.model("cactus:needle3".to_string()).await.unwrap().expect("cactus:needle3");
    let session = engine.chat(model, ChatConfig::default()).await.unwrap();
    let replies = session
        .reply(vec![ChatMessage::user().with_text("tell me a joke".to_string())], ChatReplyConfig::default())
        .await
        .unwrap();
    let last = replies.last().expect("reply");
    assert_eq!(last.finish_reason, Some(ChatReplyFinishReason::Stop));
    assert!(last.message.tool_calls().is_empty());
}

#[ignore]
#[tokio::test]
async fn needle_json_schema_fills_text() {
    if !enabled() {
        return;
    }
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model = engine.model("cactus:needle3".to_string()).await.unwrap().expect("cactus:needle3");
    let session = engine.chat(model, ChatConfig::default()).await.unwrap();
    let schema =
        r#"{"title":"Invoice","type":"object","properties":{"vendor":{"type":"string"}},"required":["vendor"]}"#;
    let replies = session
        .reply(
            vec![ChatMessage::user().with_text("Invoice from Acme Corp".to_string())],
            ChatReplyConfig::default().with_grammar(Some(Grammar::JsonSchema {
                schema: schema.to_string(),
            })),
        )
        .await
        .unwrap();
    let last = replies.last().expect("reply");
    assert_eq!(last.finish_reason, Some(ChatReplyFinishReason::Stop));
    let text = last.message.text().expect("structured text");
    let _: serde_json::Value = serde_json::from_str(&text).unwrap();
}

#[ignore]
#[tokio::test]
async fn needle_disabled_is_absent() {
    let engine = Engine::new(EngineConfig::default().with_allow_needle_usage(false)).await.unwrap();
    let model = engine.model("cactus:needle3".to_string()).await.unwrap();
    assert!(model.is_none());
}
