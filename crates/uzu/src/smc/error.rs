use thiserror::Error;

use crate::session::types::Error as SessionError;

#[derive(Debug, Error)]
pub enum SmcError {
    #[error("target model failed to load: {0}")]
    TargetLoad(SessionError),

    #[error("draft model failed to load: {0}")]
    DraftLoad(SessionError),

    #[error(
        "draft and target must share vocabulary; target vocab={target}, draft vocab={draft}"
    )]
    VocabMismatch {
        target: usize,
        draft: usize,
    },

    #[error("tokenizer load failed: {0}")]
    Tokenizer(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("target generation error: {0}")]
    TargetGen(SessionError),

    #[error("draft generation error: {0}")]
    DraftGen(SessionError),

    #[error("SMC phase-0 has not yet implemented: {0}")]
    Unimplemented(&'static str),
}
