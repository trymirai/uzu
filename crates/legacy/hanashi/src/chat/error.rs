use crate::chat::hanashi::Error as HanashiError;
#[cfg(not(target_family = "wasm"))]
use crate::chat::harmony::Error as HarmonyError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hanashi(#[from] HanashiError),
    #[error(transparent)]
    #[cfg(not(target_family = "wasm"))]
    Harmony(#[from] HarmonyError),
}
