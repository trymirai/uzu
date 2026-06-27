//! Chat screen: streaming local inference with a reasoning panel, perf stats,
//! and a composer. [`conversation`] holds the message model + its tests,
//! [`sampling`] the sampling-mode mapping, and [`view`] the `ChatView` itself.

mod conversation;
mod params;
mod sampling;
mod stream;
mod view;

pub use view::{ChatEvent, ChatView};
