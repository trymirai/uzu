use crate::chat::{hanashi::Error as HanashiError, harmony::Error as HarmonyError};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hanashi(#[from] HanashiError),
    #[error(transparent)]
    Harmony(#[from] HarmonyError),
}
