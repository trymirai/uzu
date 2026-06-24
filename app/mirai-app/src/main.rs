//! Mirai — a native GPUI reimplementation of the Mirai desktop chat client.
//!
//! Phase A: core experience. Boots a dedicated Tokio runtime, initializes the
//! uzu engine on it, then hands GPUI the main thread to run the UI. The engine
//! is exposed to views as a GPUI global (`engine` module); inference streams
//! are bridged back to the UI via `gpui_tokio`.

mod app_shell;
mod assets;
mod engine;
mod models_store;
mod persistence;
mod screens;
mod settings_state;
mod toast;

// The design system (theme + components) lives in the `ui-kit` crate. Re-export
// it under `crate::theme` / `crate::components` so the rest of the app keeps its
// existing import paths.
pub(crate) use ui_kit::{components, theme};

use gpui::{
    App, Bounds, KeyBinding, Menu, MenuItem, TitlebarOptions, WindowBounds, WindowOptions, actions,
    point, prelude::*, px, size,
};
use gpui_platform::application;

use crate::app_shell::MiraiApp;

actions!(mirai, [Quit]);

fn main() {
    // Dedicated multi-threaded Tokio runtime for uzu. Owned here so it stays
    // alive for the whole app (GPUI's `run` blocks until quit); only its handle
    // is handed to `gpui_tokio`.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build Tokio runtime");
    let handle = runtime.handle().clone();

    // Initialize the engine up-front on the runtime (fast: registry + backends).
    let engine = runtime
        .block_on(async { uzu::engine::Engine::new(uzu::engine::EngineConfig::default()).await });

    application()
        .with_assets(assets::Assets::new())
        .run(move |cx: &mut App| {
            // Mirai design system: bundled fonts + default (dark) palette.
            theme::init(cx);
            settings_state::init(cx);
            toast::init(cx);
            components::text_input::register(cx);

            // Native app menu + ⌘Q quit.
            cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.set_menus([Menu::new("Mirai").items([MenuItem::action("Quit Mirai", Quit)])]);

            // Register our runtime handle so `gpui_tokio::Tokio::spawn` runs uzu
            // futures on it and bridges results back to the UI thread.
            gpui_tokio::init_from_handle(cx, handle.clone());

            match &engine {
                Ok(engine) => {
                    crate::engine::init(cx, engine.clone());
                    eprintln!("[mirai-app] uzu Engine ready");
                }
                Err(err) => eprintln!("[mirai-app] uzu Engine init failed: {err}"),
            }

            let bounds = Bounds::centered(None, size(px(1200.), px(800.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitlebarOptions {
                        title: None,
                        appears_transparent: true,
                        // mirai-chat positions the macOS traffic lights at (20, 18).
                        traffic_light_position: Some(point(px(20.), px(18.))),
                    }),
                    window_min_size: Some(size(px(720.), px(560.))),
                    ..Default::default()
                },
                |_, cx| cx.new(|cx| MiraiApp::new(cx)),
            )
            .unwrap();
            cx.activate(true);
        });
}
