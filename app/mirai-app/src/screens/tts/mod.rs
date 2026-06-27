//! Text-to-speech screen: editor + model picker + audio playback/history.
//! [`vm`] holds the model-row view-model; [`view`] holds `TtsView`.

mod view;
mod vm;

pub use view::TtsView;
