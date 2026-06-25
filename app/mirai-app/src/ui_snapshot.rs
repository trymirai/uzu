//! Headless visual-snapshot harness.
//!
//! Renders a screen through GPUI's offscreen `render_to_image` (real Metal
//! render to an in-memory texture — no on-screen window, no OS `screencapture`,
//! works even with the display locked) and writes a PNG under
//! `target/ui-snapshots/`. Individual views are rendered standalone so the
//! engine is never required.

use std::{path::PathBuf, sync::Arc};

use gpui::{
    AnyView, AnyWindowHandle, App, AppContext, Context, Entity, HeadlessAppContext, Hsla, Platform,
    Render, Window, div, prelude::*, px, size,
};

use crate::{
    assets, components,
    models_store::{ModelKind, ModelsStore},
    screens, settings_state,
    theme::{self, ActiveTheme, FONT_SANS},
    toast,
};

/// Wraps a screen in the same root context the real app provides — background,
/// default text color, and font family — so views that inherit those (e.g.
/// titles without an explicit `text_color`) render correctly headlessly.
struct SnapshotRoot {
    inner: AnyView,
    bg: Hsla,
    fg: Hsla,
}

impl Render for SnapshotRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(self.bg)
            .text_color(self.fg)
            .font_family(FONT_SANS)
            .text_size(px(14.))
            .child(self.inner.clone())
    }
}

/// Render `view` into a `width`×`height` offscreen window and save it as
/// `target/ui-snapshots/<name>.png`.
fn render_png<V: Render + 'static>(
    name: &str,
    width: f32,
    height: f32,
    build: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
) -> PathBuf {
    let text_system = gpui_platform::current_platform(true).text_system();
    let mut cx = HeadlessAppContext::with_platform(
        text_system,
        Arc::new(assets::Assets::new()),
        gpui_platform::current_headless_renderer,
    );

    // Same global setup the real app does at startup (fonts, theme, inputs).
    cx.update(|app| {
        theme::init(app);
        settings_state::init(app);
        toast::init(app);
        components::text_input::register(app);
    });

    let window = cx
        .open_window(size(px(width), px(height)), |window, app| {
            let inner: AnyView = build(window, app).into();
            let theme = app.theme().clone();
            app.new(|_| SnapshotRoot { inner, bg: theme.bg, fg: theme.text })
        })
        .expect("open offscreen window");
    let handle: AnyWindowHandle = window.into();
    // Draw twice with the scheduler drained in between, so async asset/font
    // loading and layout settle before we capture the frame.
    for _ in 0..2 {
        cx.run_until_parked();
        cx.update_window(handle, |_, window, app| {
            window.draw(app);
        })
        .expect("draw window");
    }

    let image = cx.capture_screenshot(handle).expect("render_to_image");
    // Workspace `target/` (gitignored), not the crate dir.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/ui-snapshots");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("{name}.png"));
    image.save(&path).expect("save png");
    eprintln!("ui-snapshot: wrote {} ({}x{})", path.display(), image.width(), image.height());
    path
}

#[test]
fn render_settings() {
    render_png("settings", 1200.0, 800.0, |_, cx| cx.new(screens::SettingsView::new));
}

#[test]
fn render_chats() {
    render_png("chats", 1200.0, 800.0, |_, cx| cx.new(screens::ChatsView::new));
}

#[test]
fn render_chat() {
    // Empty model stores (no engine) render the chat empty state + composer.
    render_png("chat", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        cx.new(|cx| screens::ChatView::new(store, cloud, cx))
    });
}
