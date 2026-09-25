use tokio::sync::OnceCell;
use uzu::engine::{Engine, EngineConfig};

use crate::error::{AppError, AppResult};

static ENGINE: OnceCell<Engine> = OnceCell::const_new();

// A local bundle launches with cwd=/, so the repo root is found by walking up
// from the executable as well. Only a directory that holds this crate counts:
// an installed app must not pick up a stray .env from /Applications or /.
fn repo_env_file() -> Option<std::path::PathBuf> {
    let cwd = std::env::current_dir().ok();
    let exe = std::env::current_exe().ok();
    cwd.iter()
        .chain(exe.iter())
        .flat_map(|path| path.ancestors())
        .find(|dir| dir.join("src-tauri/Cargo.toml").is_file())
        .map(|dir| dir.join(".env"))
        .filter(|env_file| env_file.is_file())
}

fn mirai_api_key() -> Option<String> {
    if let Ok(key) = std::env::var("API_KEY")
        && !key.is_empty()
    {
        return Some(key);
    }
    if let Some(env_file) = repo_env_file() {
        let _ = dotenvy::from_path(env_file);
    }
    std::env::var("API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        // A packaged app has no .env next to the binary, so a compile-time
        // value is the fallback.
        .or_else(|| option_env!("API_KEY").map(str::to_string).filter(|k| !k.is_empty()))
}

pub async fn engine() -> AppResult<Engine> {
    ENGINE
        .get_or_try_init(|| async {
            // Only local models are offered, so cloud provider keys from the
            // shell and the Ollama/LM Studio probes would be wasted work.
            let mut config = EngineConfig {
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
            if let Some(key) = mirai_api_key() {
                config = config.with_mirai_api_key(key);
            }
            Engine::new(config).await.map_err(|e| {
                let error = AppError::msg(format!("engine init failed: {e}"));
                crate::logger::error("engine:init:error", Some(serde_json::json!({ "error": error })));
                error
            })
        })
        .await
        .cloned()
}
