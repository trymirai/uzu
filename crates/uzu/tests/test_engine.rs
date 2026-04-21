#![cfg(not(target_family = "wasm"))]

use shoji::types::{
    encoding::Message,
    session::chat::{Config as ChatConfig, StreamConfig as ChatStreamConfig},
};
use uzu::engine::{Config, Engine};

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

    let model = engine.model("alibaba:qwen3:0.6b").await.unwrap().unwrap();
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
        Message::system().with_text("You are a helpful assistant.".to_string()),
        Message::user().with_text("My name is John Doe".to_string()),
    ];
    let (mut stream, _) = session.response_stream(messages, ChatStreamConfig::default());
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

    let messages = vec![Message::user().with_text("What is my name?".to_string())];
    let _ = session.response(messages, ChatStreamConfig::default()).await.unwrap();

    let messages = session.messages().await;
    for message in messages {
        println!("Message: {message:?}");
    }
}
