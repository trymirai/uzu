use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{Feature, ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPreset},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let feature = Feature {
        name: "sentiment".to_string(),
        values: vec![
            "Happy".to_string(),
            "Sad".to_string(),
            "Angry".to_string(),
            "Fearful".to_string(),
            "Surprised".to_string(),
            "Disgusted".to_string(),
        ],
    };
    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Classification {
        feature: feature.clone(),
    }));
    let session = engine.chat(model, chat_config).await?;

    let text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    let prompt = format!(
        "Text is: \"{text_to_detect_feature}\". Choose {} from the list: {}. Answer with one word. Don't add a dot at the end.",
        feature.name,
        feature.values.join(", ")
    );
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(prompt),
    ];

    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(32)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        println!("Prediction: {}", reply.message.text().unwrap_or_default());
        println!("Generated tokens: {}", reply.stats.tokens_count_output.unwrap_or_default());
    }

    Ok(())
}
