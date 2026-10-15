use std::path::Path;

use hanashi::chat::EncodingConfig;
use shoji::types::{basic::ReasoningEffort, model::Model, session::chat::ChatModelCapabilities};

use super::{Engine, EngineError};

/// Chat metadata available without loading model weights or allocating a session.
#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq)]
pub struct ChatModelInfo {
    /// Architectural limit; absent before download or for models without a finite limit.
    pub context_length: Option<u32>,
    pub capabilities: ChatModelCapabilities,
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

#[bindings::export(Implementation)]
impl Engine {
    /// Returns metadata for a Uzu chat model, including models not yet downloaded.
    /// Encoding capabilities come from the catalog; the context limit needs local configuration.
    /// Remote and non-chat models return `None`.
    #[bindings::export(Method)]
    pub async fn chat_model_info(
        &self,
        model: &Model,
    ) -> Result<Option<ChatModelInfo>, EngineError> {
        if !model.is_on_device()
            || !model.is_chat_capable()
            || !model.backends.iter().any(|backend| backend.identifier == "uzu")
        {
            return Ok(None);
        }
        let path = self.model_path(model).await;
        let encoding = model.encoding.clone();
        tokio::task::spawn_blocking(move || {
            let encoding = encoding.ok_or_else(|| EngineError::ModelConfig {
                message: "Chat model has no supported encoding".into(),
            })?;
            let config: EncodingConfig = serde_json::from_str(&encoding.json).map_err(config_error)?;
            Ok(Some(ChatModelInfo {
                context_length: match path {
                    Some(path) => {
                        uzu_engine::engine::resolve_model_context_length(Path::new(&path)).map_err(config_error)?
                    },
                    None => None,
                },
                capabilities: config.capabilities().map_err(config_error)?,
                default_reasoning_effort: config.default_reasoning_effort().map_err(config_error)?,
            }))
        })
        .await
        .map_err(|error| EngineError::TokioError {
            message: error.to_string(),
        })?
    }
}

fn config_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::ModelConfig {
        message: error.to_string(),
    }
}
