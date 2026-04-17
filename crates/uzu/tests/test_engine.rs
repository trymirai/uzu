#[cfg(not(target_family = "wasm"))]
use uzu::engine::{Config, Engine};

#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn test_engine() {
    dotenvy::dotenv().ok();

    let config = Config::default();
    let engine = Engine::new(config).await.unwrap();

    let models = engine.models().await.unwrap();
    for model in models {
        let identifier = model.identifier();
        let download_state = engine.downloader(&model).state().await.unwrap();
        println!("{}: {:?}", identifier, download_state.phase);
    }
}
