pub mod backend;
pub mod config;
pub mod engine;
pub mod error;
pub mod ffi;
pub mod instance;
pub mod mapping;

pub use backend::Backend;
pub use config::{
    BACKEND_IDENTIFIER, BACKEND_NAME, Config, DEFAULT_MODEL_IDENTIFIER, DiscoveryHints, KEY_NEEDLE_MODELS_DIR,
    KEY_NEEDLE_WEIGHTS_PATH, KEY_NEEDLE3_LIB_PATH, NEEDLE3_ENGINE_VERSION, NEEDLE3_TAG, model_identifier,
};
pub use engine::{EngineHandle, NeedleState};
pub use error::Error;
