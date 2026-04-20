#![cfg(not(target_family = "wasm"))]

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
        println!("{:?}, {}, {:?}", registry_identifier, identifier, download_phase);
    }

    let model = engine.model("alibaba:qwen3.5:0.8b:mlx:8").await.unwrap().unwrap();
    let session = engine.chat(model).await.unwrap();
}
