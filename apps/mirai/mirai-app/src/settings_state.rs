use gpui::{App, Context, Global, Subscription};

use crate::persistence::{self, AppSettings};

struct GlobalSettings(AppSettings);

impl Global for GlobalSettings {}

pub fn init(cx: &mut App) {
    cx.set_global(GlobalSettings(persistence::load_settings()));
}

pub fn current(cx: &App) -> AppSettings {
    cx.try_global::<GlobalSettings>().map(|g| g.0.clone()).unwrap_or_default()
}

pub fn set(
    cx: &mut App,
    settings: AppSettings,
) {
    persistence::save_settings(&settings);
    cx.set_global(GlobalSettings(settings));
}

pub fn observe<V: 'static>(
    cx: &mut Context<V>,
    mut on_change: impl FnMut(&mut V, &mut Context<V>) + 'static,
) -> Subscription {
    cx.observe_global::<GlobalSettings>(move |this, cx| on_change(this, cx))
}
