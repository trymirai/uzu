#![cfg(not(target_family = "wasm"))]

use uzu::{
    engine::{Engine, EngineConfig},
    session::chat::ChatSessionStreamChunk,
    types::session::{
        chat::{ChatConfig, ChatMessage, ChatReplyConfig},
        classification::ClassificationMessage,
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
        let download_phase = engine.downloader(&model).state().await.map_or(None, |state| Some(state.phase));
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

    let model = engine.model("alibaba:qwen3:0.6b".to_string()).await.unwrap().unwrap();
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
    result.save_as_wav(path).unwrap();
}
