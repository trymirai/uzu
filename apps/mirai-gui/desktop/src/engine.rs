use tokio::sync::OnceCell;
use uzu::engine::{Engine, EngineConfig};

use crate::error::{AppError, AppResult};

static ENGINE: OnceCell<Engine> = OnceCell::const_new();

pub async fn engine() -> AppResult<Engine> {
    ENGINE
        .get_or_try_init(|| async {
            // Only local models are offered, so cloud provider keys from the
            // shell and the Ollama/LM Studio probes would be wasted work.
            let config = EngineConfig {
                openai_api_key: None,
                anthropic_api_key: None,
                gemini_api_key: None,
                xai_api_key: None,
                baseten_api_key: None,
                openrouter_api_key: None,
                allow_ollama_usage: false,
                allow_lmstudio_usage: false,
                ..EngineConfig::default()
            };
            Engine::new(config).await.map_err(|e| {
                let error = AppError::msg(format!("engine init failed: {e}"));
                crate::logger::error("engine:init:error", Some(serde_json::json!({ "error": error })));
                error
            })
        })
        .await
        .cloned()
}
