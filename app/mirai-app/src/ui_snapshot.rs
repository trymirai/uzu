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
    AnyView, AnyWindowHandle, App, AppContext, Context, Entity, HeadlessAppContext, Hsla, Render,
    Window, div, prelude::*, px, size,
};

use crate::{
    app_shell::MiraiApp,
    assets, components,
    models_store::{ModelKind, ModelsStore},
    persistence::{self, StoredChat, StoredMessage},
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
            let _ = window.draw(app);
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
            let _ = window.draw(app);
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

/// Point persistence at a clean, empty temp dir so the sidebar's recent-chats
/// list renders deterministically empty — these full-app snapshots would
/// otherwise read the developer's real chat history.
fn use_empty_data_dir(tag: &str) {
    let dir = std::env::temp_dir().join(format!("mirai-snap-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok();
    persistence::set_test_data_dir(Some(dir));
}

// Full app shell on the Settings route, reproducing the real sidebar + flex_1
// content-outlet nesting (the isolated `render_settings` can't surface
// outlet-width bugs).
#[test]
fn render_app_settings() {
    use_empty_data_dir("app-settings");
    unsafe { std::env::set_var("MIRAI_SCREEN", "settings") };
    render_png("app-settings", 1400.0, 820.0, |_, cx| {
        cx.new(crate::app_shell::MiraiApp::new)
    });
    unsafe { std::env::remove_var("MIRAI_SCREEN") };
    persistence::set_test_data_dir(None);
}

#[test]
fn render_settings_privacy() {
    render_png("settings-privacy", 1200.0, 800.0, |_, cx| {
        let v = cx.new(screens::SettingsView::new);
        v.update(cx, |s, cx| s.select_tab(1, cx));
        v
    });
}

#[test]
fn render_settings_about() {
    render_png("settings-about", 1200.0, 800.0, |_, cx| {
        let v = cx.new(screens::SettingsView::new);
        v.update(cx, |s, cx| s.select_tab(2, cx));
        v
    });
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
    let dir = std::env::temp_dir().join(format!("mirai-chats-snapshot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    persistence::set_test_data_dir(Some(dir.clone()));

    let fixtures = [
        persistence::StoredChat {
            id: "fixture-a".into(),
            title: "What is Rust?".into(),
            model_name: Some("Qwen3.5 0.8B".into()),
            created_at: 1_700_000_040_000,
            updated_at: 1_700_000_100_000,
            messages: vec![persistence::StoredMessage {
                role: "user".into(),
                text: "What is Rust?".into(),
                reasoning: None,
                tps: None,
                tokens: None,
            }],
        },
        persistence::StoredChat {
            id: "fixture-b".into(),
            title: "Sky color".into(),
            model_name: Some("Qwen3.5 0.8B".into()),
            created_at: 1_700_000_200_000,
            updated_at: 1_700_000_300_000,
            messages: vec![persistence::StoredMessage {
                role: "user".into(),
                text: "Why is the sky blue?".into(),
                reasoning: None,
                tps: None,
                tokens: None,
            }],
        },
    ];
    for chat in fixtures {
        persistence::save_chat(&chat);
    }

    render_png("chats", 1200.0, 800.0, |_, cx| cx.new(screens::ChatsView::new));

    // Instructions card expanded — verifies the +→× rotation.
    render_png("chats-instructions", 1200.0, 800.0, |_, cx| {
        let v = cx.new(screens::ChatsView::new);
        v.update(cx, |c, cx| c.open_instructions(cx));
        v
    });

    persistence::set_test_data_dir(None);
    let _ = std::fs::remove_dir_all(&dir);
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

// The file-upload panel must float above the composer (anchored), not grow it.
#[test]
fn render_chat_upload() {
    render_png("chat-upload", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let v = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        v.update(cx, |c, cx| c.open_file_upload(cx));
        v
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
                text: "**2 + 2 = 4**. This is a deliberately long sentence intended to exceed the content column width so we can verify the assistant message body wraps to the available width instead of overflowing outside the visible area like it used to.\n\n```python\nprint(2 + 2)  # a long code comment that extends well past the content column to confirm code blocks stay constrained\n```\n\n- It is basic arithmetic\n- The result is `4`".into(),
                reasoning: Some("The user asks a simple sum.".into()),
                tps: Some(42.0),
                tokens: Some(18),
            },
        ],
    }
}

#[test]
fn render_sidebar_settings() {
    // Full app shell with the bottom Settings menu expanded.
    use_empty_data_dir("sidebar-settings");
    render_png("sidebar-settings", 1200.0, 800.0, |_, cx| {
        let app = cx.new(MiraiApp::new);
        app.update(cx, |a, cx| a.open_settings_menu(cx));
        app
    });
    persistence::set_test_data_dir(None);
}

#[test]
fn render_chat_thinking() {
    // Expanded reasoning panel with long content — verifies monospace text,
    // the 180px cap, and the bottom fade gradient.
    let reasoning = "Let me think about Paris.\n\
        - Arc de Triomphe (Military).\n\
        - Notre-Dame Cathedral.\n\
        - The Sacré-Cœur Basilica.\n\
        - Champs-Élysées (Fashion hub).\n\
        Cuisine: mention French cuisine and specific dishes.\n\
        Music: Jazz, classical, pop, hip-hop. Mention modern music venues.\n\
        Architecture: Haussmann boulevards and the Eiffel Tower.\n\
        Transport: the Metro and RER connect the center to the suburbs.";
    let chat = StoredChat {
        id: "think".into(),
        title: "Thinking".into(),
        model_name: Some("Qwen3.5 0.8B".into()),
        created_at: 0,
        updated_at: 0,
        messages: vec![
            StoredMessage { role: "user".into(), text: "Tell me about Paris".into(), reasoning: None, tps: None, tokens: None },
            StoredMessage {
                role: "assistant".into(),
                text: "Paris is the capital of France.".into(),
                reasoning: Some(reasoning.into()),
                tps: Some(40.0),
                tokens: Some(64),
            },
        ],
    };
    render_png("chat-thinking", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let view = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        view.update(cx, |c, cx| {
            c.load_stored(chat, cx);
            c.expand_reasoning(1, cx);
        });
        view
    });
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
fn render_chat_perf() {
    // Shows the Performance popover open on the first assistant message.
    render_png("chat-perf", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let chat = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        chat.update(cx, |c, cx| {
            c.load_stored(sample_stored_chat(), cx);
            // message index 1 is the assistant reply with tps stats
            c.open_perf_panel(1, cx);
        });
        chat
    });
}

/// A long conversation that overflows the viewport, used to verify the
/// composer does not overlap the scrolled-to-bottom message content.
fn tall_stored_chat() -> StoredChat {
    let mut messages = Vec::new();
    for i in 0..6 {
        messages.push(StoredMessage {
            role: "user".into(),
            text: format!("Question number {i}: tell me about a famous landmark."),
            reasoning: None,
            tps: None,
            tokens: None,
        });
        messages.push(StoredMessage {
            role: "assistant".into(),
            text: format!(
                "Answer {i}. Here is a comprehensive overview that spans several lines so the \
                 conversation grows tall enough to overflow the viewport. The Eiffel Tower, \
                 completed in 1889, is one of the most recognizable structures in the world. \
                 The Plaza below it draws millions of visitors every year, and the surrounding \
                 gardens are a popular destination for both tourists and locals alike."
            ),
            reasoning: None,
            tps: Some(40.0),
            tokens: Some(64),
        });
    }
    StoredChat {
        id: "tall".into(),
        title: "Tall".into(),
        model_name: Some("Qwen3.5 0.8B".into()),
        created_at: 0,
        updated_at: 0,
        messages,
    }
}

#[test]
fn render_chat_overflow() {
    // Tall chat scrolled to bottom (load_stored pins it there): the last
    // message — text, perf stats, and actions — must sit above the composer
    // with clearance, never behind it.
    render_png("chat-overflow", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let chat = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        chat.update(cx, |c, cx| c.load_stored(tall_stored_chat(), cx));
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
fn render_gen_settings() {
    render_png_with_engine("gen-settings", 1200.0, 800.0, |_, cx| {
        let store = cx.new(|cx| ModelsStore::new(ModelKind::Chat, cx));
        let cloud = cx.new(|cx| ModelsStore::new(ModelKind::CloudChat, cx));
        let chat = cx.new(|cx| screens::ChatView::new(store, cloud, cx));
        chat.update(cx, |c, cx| {
            c.open_gen_settings(cx);
            c.set_stochastic(cx);
        });
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
