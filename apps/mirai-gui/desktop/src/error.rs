use crate::cli_installer::AuthError;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error(transparent)]
    Engine(#[from] uzu::engine::EngineError),
    #[error(transparent)]
    ChatSession(#[from] uzu::session::chat::ChatSessionError),
    #[error(transparent)]
    Authorization(#[from] AuthError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Message(String),
}

impl AppError {
    pub fn msg(message: impl std::fmt::Display) -> Self {
        Self::Message(message.to_string())
    }
}

// Commands reject with the plain message, the same shape as a String error.
impl serde::Serialize for AppError {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

pub type AppResult<T> = Result<T, AppError>;
