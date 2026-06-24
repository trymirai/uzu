//! Top-level screens routed by the app shell.

pub mod chat;
pub mod chats;
pub mod cloud_models;
pub mod local_models;
pub mod routers;
pub mod settings;
pub mod tts;
pub mod welcome;

pub use chat::ChatView;
pub use chats::{ChatsEvent, ChatsView};
pub use cloud_models::{CloudEvent, CloudModelsView};
pub use local_models::{LocalModelsEvent, LocalModelsView};
pub use routers::RoutersView;
pub use settings::SettingsView;
pub use tts::TtsView;
pub use welcome::{WelcomeEvent, WelcomeView};
