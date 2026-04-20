#![cfg(not(target_family = "wasm"))]

use shoji::types::{
    encoding::Message,
    session::chat::{Config as ChatConfig, StreamConfig as ChatStreamConfig},
};
use uzu::engine::{Config, Engine};

#[tokio::test]
async fn test_engine() {
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

    let model = engine.model("gpt-3.5-turbo").await.unwrap().unwrap();
    let mut session = engine.chat(model, ChatConfig::default()).await.unwrap();
    session.reset().await.unwrap();

    let messages = vec![
        Message::system().with_text("You are a helpful assistant.".to_string()),
        Message::user().with_text("Tell about London".to_string()),
    ];
    session.stream(messages, ChatStreamConfig::default()).await.unwrap();
}
