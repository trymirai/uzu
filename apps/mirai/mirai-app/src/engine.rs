use gpui::{App, Global};
use uzu::engine::Engine;

struct GlobalEngine(Engine);

impl Global for GlobalEngine {}

pub fn init(
    cx: &mut App,
    engine: Engine,
) {
    cx.set_global(GlobalEngine(engine));
}

pub fn try_engine(cx: &App) -> Option<Engine> {
    cx.try_global::<GlobalEngine>().map(|g| g.0.clone())
}
