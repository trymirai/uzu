#![cfg(not(target_family = "wasm"))]

use uzu::{
    engine::{Config, Engine},
    types::session::{
        chat::{Config as ChatConfig, Message as ChatMessage, StreamConfig as ChatStreamConfig},
        classification::Message as ClassificationMessage,
    },
};

#[ignore]
#[tokio::test]
async fn test_engine_chat() {
    dotenvy::dotenv().ok();

    let config = Config::default();
    let engine = Engine::new(config).await.unwrap();

    let models = engine.models().await.unwrap();
    for model in models {
        let identifier = model.identifier();
        let registry_identifier = model.registry_entity().map(|entity| entity.identifier);
        let download_phase = engine.downloader(&model).state().await.map_or(None, |state| Some(state.phase));
        println!("{:?}, {}, {:?}, {:?}", registry_identifier, identifier, download_phase, model.specializations);
    }

    let model = engine.model("alibaba:qwen3:0.6b".to_string()).await.unwrap().unwrap();
    if model.is_downloadable() {
        let downloader = engine.downloader(&model);
        downloader.resume().await.unwrap();
        if let Some(stream) = downloader.progress().await {
            while let Some(update) = stream.next().await {
                println!("Downloading: {}", update.progress());
            }
        }
    }

    let session = engine.chat(model, ChatConfig::default()).await.unwrap();
    session.reset().await.unwrap();

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant.".to_string()),
        ChatMessage::user().with_text("My name is John Doe".to_string()),
    ];
    let (mut stream, _) = session.reply_with_stream(messages, ChatStreamConfig::default());
    while let Some(progress) = stream.next().await {
        match progress {
            Ok(outputs) => {
                let state = session.state().await;
                let messages = session.messages().await;
                let roles = messages.iter().map(|message| message.role.clone()).collect::<Vec<_>>();
                println!("State: {state:?}");
                println!("Roles: {roles:?}");
                for output in outputs {
                    let duration = output.stats.duration;
                    let finish_reason = output.finish_reason;
                    let text_length = output.message.text().unwrap_or_default().len();
                    println!(
                        "\tDuration: {duration}\n\tFinish reason: {finish_reason:?}\n\tText length: {text_length}"
                    );
                }
            },
            Err(error) => eprintln!("Stream error: {error}"),
        }
    }

    let messages = vec![ChatMessage::user().with_text("What is my name?".to_string())];
    let _ = session.reply(messages, ChatStreamConfig::default()).await.unwrap();

    let messages = session.messages().await;
    for message in messages {
        println!("Message: {message:?}");
    }
}

#[ignore]
#[tokio::test]
async fn test_engine_classification() {
    dotenvy::dotenv().ok();

    let config = Config::default();
    let engine = Engine::new(config).await.unwrap();
    let model = engine.model("ModernBERT-Chat-Moderation".to_string()).await.unwrap().unwrap();
    let session = engine.classification(model).await.unwrap();

    let messages = vec![ClassificationMessage::user("Hi!".to_string())];
    let result = session.classify(messages).await.unwrap();
    println!("Output: {result:?}");
}

#[ignore]
#[tokio::test]
async fn test_engine_text_to_speech() {
    dotenvy::dotenv().ok();

    let config = Config::default();
    let engine = Engine::new(config).await.unwrap();
    let model = engine.model("s1-mini".to_string()).await.unwrap().unwrap();
    let session = engine.text_to_speech(model).await.unwrap();
    let result = session.synthesize("London is the capital of United Kingdom and one of the world’s most influential cities, known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace with modern architecture such as The Shard. London is also home to renowned institutions including the British Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that attracts millions of visitors each year.".to_string()).await.unwrap();

    let path = dirs::home_dir().unwrap().join("Desktop").join("output.wav").to_string_lossy().to_string();
    result.save_as_wav(path).unwrap();
}
