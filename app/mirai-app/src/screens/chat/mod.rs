//! Chat screen: streaming local inference with a reasoning panel, perf stats,
//! and a composer.

mod conversation;
mod overlays;
mod params;
mod sampling;
mod state;
mod stream;
mod view;

pub use view::{ChatEvent, ChatView};
