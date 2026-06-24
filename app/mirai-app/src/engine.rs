//! uzu `Engine` access as a GPUI global.
//!
//! The engine is initialized synchronously at startup (on the Tokio runtime,
//! before the window opens) and stored here, so every view can reach it via
//! `engine::try_engine(cx)`. The `Engine` handle is `Clone` (Arc-backed), so
//! cloning it to move into a `gpui_tokio::Tokio::spawn` future is cheap.

use gpui::{App, Global};
use uzu::engine::Engine;

struct GlobalEngine(Engine);

impl Global for GlobalEngine {}

/// Installs the ready engine. Call once at startup if `Engine::new` succeeded.
pub fn init(cx: &mut App, engine: Engine) {
    cx.set_global(GlobalEngine(engine));
}

/// Returns a clone of the engine if it initialized successfully, else `None`
/// (engine init failed — views should show an error/empty state).
pub fn try_engine(cx: &App) -> Option<Engine> {
    cx.try_global::<GlobalEngine>().map(|g| g.0.clone())
}
