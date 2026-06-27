//! App settings as a GPUI global, so screens (Settings, Chat) share one source
//! of truth and react to changes. Backed by `persistence::AppSettings` on disk.

use gpui::{App, Context, Global, Subscription};

use crate::persistence::{self, AppSettings};

struct GlobalSettings(AppSettings);

impl Global for GlobalSettings {}

/// Loads persisted settings into the global. Call once at startup.
pub fn init(cx: &mut App) {
    cx.set_global(GlobalSettings(persistence::load_settings()));
}

/// Current settings (cheap clone).
pub fn current(cx: &App) -> AppSettings {
    cx.try_global::<GlobalSettings>().map(|g| g.0.clone()).unwrap_or_default()
}

/// Persists + updates the global, notifying observers.
pub fn set(
    cx: &mut App,
    settings: AppSettings,
) {
    persistence::save_settings(&settings);
    cx.set_global(GlobalSettings(settings));
}

/// Re-runs `on_change` whenever settings change.
pub fn observe<V: 'static>(
    cx: &mut Context<V>,
    mut on_change: impl FnMut(&mut V, &mut Context<V>) + 'static,
) -> Subscription {
    cx.observe_global::<GlobalSettings>(move |this, cx| on_change(this, cx))
}
