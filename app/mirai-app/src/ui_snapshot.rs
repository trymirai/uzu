//! Headless visual-snapshot harness.
//!
//! Renders a screen through GPUI's offscreen `render_to_image` (real Metal
//! render to an in-memory texture — no on-screen window, no OS `screencapture`,
//! works even with the display locked) and writes a PNG under
//! `target/ui-snapshots/`. Individual views are rendered standalone so the
//! engine is never required.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use gpui::{
    AnyView, AnyWindowHandle, App, AppContext, Context, Entity, HeadlessAppContext, Hsla, Platform,
    Render, Window, div, prelude::*, px, size,
};

use crate::{
    assets, components,
    models_store::{ModelKind, ModelsStore},
    persistence::{StoredChat, StoredMessage},
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
    compare_baseline(name, &path);
    path
}

/// Like `render_png`, but boots a real uzu engine + Tokio bridge first so
/// engine-backed screens (Local Models, TTS, Cloud, Routers) populate their
/// model lists. Slower (engine init), and requires a working engine.
fn render_png_with_engine<V: Render + 'static>(
    name: &str,
    width: f32,
    height: f32,
    build: impl FnOnce(&mut Window, &mut App) -> Entity<V>,
) -> PathBuf {
    let runtime =
        tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    let handle = runtime.handle().clone();
    let engine = runtime
        .block_on(async { uzu::engine::Engine::new(uzu::engine::EngineConfig::default()).await })
        .expect("engine init");

    let text_system = gpui_platform::current_platform(true).text_system();
    let mut cx = HeadlessAppContext::with_platform(
        text_system,
        Arc::new(assets::Assets::new()),
        gpui_platform::current_headless_renderer,
    );
    cx.update(|app| {
        theme::init(app);
        settings_state::init(app);
        toast::init(app);
        components::text_input::register(app);
        gpui_tokio::init_from_handle(app, handle.clone());
        crate::engine::init(app, engine.clone());
    });

    let window = cx
        .open_window(size(px(width), px(height)), |window, app| {
            let inner: AnyView = build(window, app).into();
            let theme = app.theme().clone();
            app.new(|_| SnapshotRoot { inner, bg: theme.bg, fg: theme.text })
        })
        .expect("open offscreen window");
    let handle_w: AnyWindowHandle = window.into();

    // Give the async model load time to run on Tokio, then pump + draw.
    std::thread::sleep(Duration::from_secs(3));
    for _ in 0..3 {
        cx.run_until_parked();
        cx.update_window(handle_w, |_, window, app| {
            window.draw(app);
        })
        .expect("draw window");
        std::thread::sleep(Duration::from_millis(300));
    }

    let image = cx.capture_screenshot(handle_w).expect("render_to_image");
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/ui-snapshots");
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("{name}.png"));
    image.save(&path).expect("save png");
    eprintln!("ui-snapshot: wrote {} ({}x{})", path.display(), image.width(), image.height());
    compare_baseline(name, &path);
    drop(runtime);
    path
}

/// Compare a freshly rendered PNG against the committed golden baseline under
/// `tests/ui-baselines/`. Set `UPDATE_BASELINES=1` (or when no baseline exists)
/// to refresh. Tolerates tiny anti-aliasing noise; fails on layout changes.
fn compare_baseline(name: &str, rendered: &Path) {
    let baseline =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ui-baselines").join(format!("{name}.png"));
    if std::env::var("UPDATE_BASELINES").is_ok() || !baseline.exists() {
        std::fs::create_dir_all(baseline.parent().unwrap()).ok();
        std::fs::copy(rendered, &baseline).expect("write baseline");
        eprintln!("ui-snapshot: baseline {} updated", name);
        return;
    }
    let a = image::open(rendered).expect("open rendered").to_rgba8();
    let b = image::open(&baseline).expect("open baseline").to_rgba8();
    assert_eq!((a.width(), a.height()), (b.width(), b.height()), "{name}: dimensions differ");
    let total = (a.width() * a.height()).max(1) as usize;
    let diff = a
        .pixels()
        .zip(b.pixels())
        .filter(|(pa, pb)| pa.0.iter().zip(pb.0.iter()).any(|(x, y)| x.abs_diff(*y) > 8))
        .count();
    let pct = diff as f64 / total as f64 * 100.0;
    assert!(
        pct < 0.5,
        "{name}: {pct:.3}% pixels differ ({diff}/{total}) — run UPDATE_BASELINES=1 to refresh"
    );
}

#[test]
fn render_settings() {
    render_png("settings", 1200.0, 800.0, |_, cx| cx.new(screens::SettingsView::new));
}

#[test]
#[ignore = "boots a real engine; run explicitly with --ignored"]
fn render_local_models() {
    render_png_with_engine("local-models", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        cx.new(|cx| screens::LocalModelsView::new(store, cx))
    });
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

/// A two-message conversation exercising markdown (bold, code block, list,
/// inline code), reasoning, and perf stats.
fn sample_stored_chat() -> StoredChat {
    StoredChat {
        id: "sample".into(),
        title: "Sample".into(),
        model_name: Some("Qwen3.5 0.8B".into()),
        created_at: 0,
        updated_at: 0,
        messages: vec![
            StoredMessage {
                role: "user".into(),
                text: "What is 2+2, and show a code example?".into(),
                reasoning: None,
                tps: None,
                tokens: None,
            },
            StoredMessage {
                role: "assistant".into(),
                text: "**2 + 2 = 4**.\n\nHere's a quick example:\n\n```python\nprint(2 + 2)\n```\n\n- It is basic arithmetic\n- The result is `4`".into(),
                reasoning: Some("The user asks a simple sum.".into()),
                tps: Some(42.0),
                tokens: Some(18),
            },
        ],
    }
}

#[test]
fn render_chat_messages() {
    render_png("chat-messages", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let chat = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        chat.update(cx, |c, cx| c.load_stored(sample_stored_chat(), cx));
        chat
    });
}

#[test]
fn render_welcome() {
    render_png("welcome", 1200.0, 800.0, |_, cx| cx.new(screens::WelcomeView::new));
}

#[test]
fn render_textarea() {
    // Multiline TextInput with seeded content — verifies multi-row rendering,
    // auto-grow, and cursor placement (the composer's editor).
    render_png("textarea", 600.0, 200.0, |_, cx| {
        let input = cx.new(|cx| components::TextInput::new(cx, "Type…").multiline(true, 3, 8));
        input.update(cx, |t, cx| {
            t.set_text("First line of a longer message\nSecond line\nThird line here", cx)
        });
        input
    });
}

#[test]
#[ignore = "boots a real engine; run explicitly with --ignored"]
fn render_model_picker() {
    render_png_with_engine("model-picker", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let chat = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        chat.update(cx, |c, cx| c.open_model_picker(cx));
        chat
    });
}

#[test]
#[ignore = "boots a real engine; run explicitly with --ignored"]
fn render_tts() {
    render_png_with_engine("tts", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::TextToSpeech, cx));
        cx.new(|cx| screens::TtsView::new(store, cx))
    });
}

#[test]
#[ignore = "boots a real engine; run explicitly with --ignored"]
fn render_cloud() {
    render_png_with_engine("cloud", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        cx.new(|cx| screens::CloudModelsView::new(store, cx))
    });
}

#[test]
#[ignore = "boots a real engine; run explicitly with --ignored"]
fn render_routers() {
    render_png_with_engine("routers", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Classification, cx));
        cx.new(|cx| screens::RoutersView::new(store, cx))
    });
}
