#![cfg(not(target_family = "wasm"))]

use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig},
    },
};

fn contains_word(
    text: &str,
    word: &str,
) -> bool {
    text.split(|character: char| !character.is_alphanumeric()).any(|candidate| candidate.eq_ignore_ascii_case(word))
}

async fn describe_image(url: &str) -> String {
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model = engine.model("trymirai/Qwen3.5-9B-M-Vision".to_string()).await.unwrap().expect("Model not found");
    println!("Loading {model_id}...");
    let download = engine.download(&model).await.unwrap();
    while download.next().await.is_some() {}
    // let path = dirs::home_dir().unwrap().join(".cache/huggingface/hub/models--trymirai--Qwen3.5-9B-M-Vision/snapshots/214087cc7236233102135f51bd923a43630cb3f2").to_string_lossy().to_string();
    // let mut model = engine.model(path).await.unwrap().unwrap();
    // model.encoding = Some(serde_json::json!({"type": "hanashi", "name": "qwen3.5"}).into());

    let session = engine.chat(model, ChatConfig::default()).await.unwrap();
    let messages = vec![
        ChatMessage::system()
            .with_text("You are a helpful assistant".to_string())
            .with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user()
            .with_block(ChatContentBlock::Image {
                url: url.to_string(),
            })
            .with_text("What's on the image?".to_string()),
    ];
    let config = ChatReplyConfig::default().with_token_limit(Some(256)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, config).await.unwrap();
    replies.last().unwrap().message.text().unwrap_or_default()
}

#[test_tag::tag(heavy)]
#[tokio::test]
async fn test_engine_chat_vision_cat() {
    let text = describe_image("https://assets.trymirai.com/images/artifacts/photo-cat.jpeg").await;
    assert!(contains_word(&text, "cat") || contains_word(&text, "cats"), "the image should be a cat: {text}");
}

#[test_tag::tag(heavy)]
#[tokio::test]
async fn test_engine_chat_vision_london() {
    let text = describe_image("https://assets.trymirai.com/images/artifacts/photo-london.jpeg").await;
    assert!(contains_word(&text, "London"), "the image should be London: {text}");
}
