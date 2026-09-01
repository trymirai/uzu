use std::{path::PathBuf, sync::OnceLock};

const MODEL_FILE_NAME: &str = "model.safetensors";

pub fn get_test_model_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(resolve_test_model_path).clone()
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_test_model_path() -> PathBuf {
    use uzu::{
        engine::{Engine, EngineConfig},
        storage::types::DownloadPhase,
    };

    const TEST_MODEL_REPO_ID: &str = "meta-llama/Llama-3.2-1B-Instruct";
    let repo_id = std::env::var("TEST_MODEL").unwrap_or_else(|_| TEST_MODEL_REPO_ID.to_string());
    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        let config = EngineConfig::default().with_allow_ollama_usage(false).with_allow_lmstudio_usage(false);
        let engine = Engine::new(config).await.expect("failed to create engine");
        let model = engine
            .model(repo_id.clone())
            .await
            .unwrap_or_else(|error| panic!("failed to look up {repo_id} in the registry: {error}"))
            .unwrap_or_else(|| panic!("model {repo_id} not found in the registry"));

        let stream = engine.download(&model).await.expect("failed to start model download");
        while stream.next().await.is_some() {}

        let state = engine.download_state(&model).await.expect("model has no download state");
        assert!(
            matches!(state.phase, DownloadPhase::Downloaded {}),
            "model {repo_id} download did not complete: {:?}",
            state.phase,
        );

        let path = engine.model_path(&model).await.expect("model path unavailable after download");
        PathBuf::from(path)
    })
}

#[cfg(target_arch = "wasm32")]
fn resolve_test_model_path() -> PathBuf {
    panic!("test model download is not supported on wasm")
}
