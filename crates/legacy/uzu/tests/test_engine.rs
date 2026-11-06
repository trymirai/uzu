#![cfg(not(target_family = "wasm"))]

use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::{ChatSession, ChatSessionStreamChunk},
    types::{
        basic::{ReasoningEffort, SamplingMethod, SamplingSeed},
        session::{
            chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig},
            classification::ClassificationMessage,
        },
    },
};

#[ignore]
#[tokio::test]
async fn test_engine_chat() {
    dotenvy::dotenv().ok();

    let config = EngineConfig::default();
    let engine = Engine::new(config).await.unwrap();

    println!("-------------------------");
    let models = engine.models().await.unwrap();
    for model in models {
        let identifier = model.identifier.clone();
        let name = model.name();
        let repo_ids = model.repo_ids();
        let registry_identifier = model.registry.identifier.clone();
        let backend_identifiers = model.backends.iter().map(|backend| backend.identifier.clone()).collect::<Vec<_>>();
        let family_vendor_identifier = model.family.as_ref().map(|family| family.vendor.identifier.clone());
        let family_identifier = model.family.as_ref().map(|family| family.identifier.clone());
        let properties_identifier = model.properties.as_ref().map(|properties| properties.identifier.clone());
        let quantization_vendor_identifier =
            model.quantization.as_ref().map(|quantization| quantization.vendor.identifier.clone());
        let quantization_identifier = model.quantization.as_ref().map(|quantization| quantization.identifier.clone());
        let download_phase = engine.downloader(&model).state().await.map(|state| state.phase);
        println!("identifier: {}", identifier);
        println!("name: {}", name);
        println!("repo_ids: {:?}", repo_ids);
        println!("registry_identifier: {}", registry_identifier);
        println!("backend_identifiers: {:?}", backend_identifiers);
        println!("family_vendor_identifier: {:?}", family_vendor_identifier);
        println!("family_identifier: {:?}", family_identifier);
        println!("properties_identifier: {:?}", properties_identifier);
        println!("quantization_vendor_identifier: {:?}", quantization_vendor_identifier);
        println!("quantization_identifier: {:?}", quantization_identifier);
        println!("download_phase: {:?}", download_phase);
        println!("specializations: {:?}", model.specializations);
        println!("-------------------------");
    }

    let model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string()).await.unwrap().unwrap();
    while let Some(update) = engine.download(&model).await.unwrap().next().await {
        println!("Downloading: {}", update.progress());
    }

    let session = engine.chat(model, ChatConfig::default()).await.unwrap();
    session.reset().await.unwrap();

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant.".to_string()),
        ChatMessage::user().with_text("My name is John Doe".to_string()),
    ];
    let stream = session.reply_with_stream(messages, ChatReplyConfig::default()).await;
    while let Some(progress) = stream.next().await {
        match progress {
            ChatSessionStreamChunk::ToolResults {
                ..
            } => {},
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                let state = session.state().await;
                let messages = session.messages().await;
                let roles = messages.iter().map(|message| message.role.clone()).collect::<Vec<_>>();
                println!("State: {state:?}");
                println!("Roles: {roles:?}");
                for reply in replies {
                    let duration = reply.stats.duration;
                    let finish_reason = reply.finish_reason;
                    let text_length = reply.message.text().unwrap_or_default().len();
                    println!(
                        "\tDuration: {duration}\n\tFinish reason: {finish_reason:?}\n\tText length: {text_length}"
                    );
                }
            },
            ChatSessionStreamChunk::Error {
                error,
            } => eprintln!("Stream error: {error}"),
        }
    }

    let messages = vec![ChatMessage::user().with_text("What is my name?".to_string())];
    let _ = session.reply(messages, ChatReplyConfig::default()).await.unwrap();

    let messages = session.messages().await;
    for message in messages {
        println!("Message: {message:?}");
    }
}

async fn reply_content(
    session: &ChatSession,
    messages: Vec<ChatMessage>,
    config: ChatReplyConfig,
) -> Vec<ChatContentBlock> {
    let replies = session.reply(messages, config).await.unwrap();
    replies.last().unwrap().message.content.clone()
}

async fn fresh_reply(
    session: &ChatSession,
    config: ChatReplyConfig,
) -> Vec<ChatContentBlock> {
    session.reset().await.unwrap();
    // Without thinking, so the sample isn't spent on the model's fixed reasoning preamble.
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text("Write one sentence about a random animal.".to_string()),
    ];
    reply_content(session, messages, config).await
}

async fn follow_up_reply(
    session: &ChatSession,
    config: ChatReplyConfig,
) -> Vec<ChatContentBlock> {
    reply_content(session, vec![ChatMessage::user().with_text("Now another one.".to_string())], config).await
}

#[ignore]
#[tokio::test]
async fn test_engine_chat_reply_seed() {
    let engine = Engine::new(EngineConfig::default()).await.unwrap();
    let model = engine.model("alibaba:qwen3.5:0.8b:mirai:mirai-m:4".to_string()).await.unwrap().unwrap();
    let download = engine.download(&model).await.unwrap();
    while download.next().await.is_some() {}

    let session_config = ChatConfig::default().with_sampling_seed(SamplingSeed::Custom {
        seed: 1,
    });
    let session = engine.chat(model, session_config).await.unwrap();
    let config =
        ChatReplyConfig::default().with_token_limit(Some(32)).with_sampling_method(SamplingMethod::Stochastic {
            temperature: Some(1.0),
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        });
    let with_seed = |seed| {
        config.with_sampling_seed(SamplingSeed::Custom {
            seed,
        })
    };

    let session_seed = fresh_reply(&session, config.clone()).await;
    let other_seed = fresh_reply(&session, with_seed(2)).await;
    let same_seed_per_reply = fresh_reply(&session, with_seed(1)).await;
    assert_ne!(session_seed, other_seed, "a reply seed should change the sample");
    assert_eq!(session_seed, same_seed_per_reply, "a reply seed should match the same session seed");

    // The follow-up continues on the state the seeded reply sampled with.
    fresh_reply(&session, with_seed(2)).await;
    let follow_up_without_seed = follow_up_reply(&session, config.clone()).await;
    fresh_reply(&session, with_seed(2)).await;
    let follow_up_with_session_seed = follow_up_reply(&session, with_seed(1)).await;
    assert_eq!(follow_up_without_seed, follow_up_with_session_seed, "a reply seed should not outlive its reply");
}

#[ignore]
#[tokio::test]
async fn test_engine_classification() {
    dotenvy::dotenv().ok();

    let config = EngineConfig::default();
    let engine = Engine::new(config).await.unwrap();
    let model = engine.model("ModernBERT-Chat-Moderation".to_string()).await.unwrap().unwrap();
    while let Some(update) = engine.download(&model).await.unwrap().next().await {
        println!("Downloading: {}", update.progress());
    }

    let session = engine.classification(model).await.unwrap();
    let messages = vec![ClassificationMessage::user("Hi!".to_string())];
    let result = session.classify(messages).await.unwrap();
    println!("Output: {result:?}");
}

#[ignore]
#[tokio::test]
async fn test_engine_text_to_speech() {
    dotenvy::dotenv().ok();

    let config = EngineConfig::default();
    let engine = Engine::new(config).await.unwrap();
    let model = engine.model("s1-mini".to_string()).await.unwrap().unwrap();
    while let Some(update) = engine.download(&model).await.unwrap().next().await {
        println!("Downloading: {}", update.progress());
    }

    let session = engine.text_to_speech(model).await.unwrap();
    let result = session.synthesize("London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.".to_string()).await.unwrap();
    let path = dirs::home_dir().unwrap().join("Desktop").join("output.wav").to_string_lossy().to_string();
    result.pcm_batch.save_as_wav(path).unwrap();
}
