use crate::chat::hanashi::Error as HanashiError;
#[cfg(not(target_arch = "wasm32"))]
use crate::chat::harmony::Error as HarmonyError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hanashi(#[from] HanashiError),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    Harmony(#[from] HarmonyError),
}
