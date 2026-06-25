//! Mirai — native GPUI client for the Mirai desktop chat app.

mod app_shell;
mod assets;
mod engine;
mod models_store;
mod persistence;
mod screens;
mod settings_state;
mod toast;
#[cfg(test)]
mod ui_snapshot;

// Design system lives in the `ui-kit` crate; re-export under the old paths.
pub(crate) use ui_kit::{components, theme};

use gpui::{
    App, Bounds, KeyBinding, Menu, MenuItem, TitlebarOptions, WindowBounds, WindowOptions, actions,
    point, prelude::*, px, size,
};
use gpui_platform::application;

use crate::app_shell::MiraiApp;

actions!(mirai, [Quit]);

fn main() {
    // Tokio runtime for uzu, kept alive for the app's lifetime; GPUI gets its handle.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build Tokio runtime");
    let handle = runtime.handle().clone();

    let engine = runtime
        .block_on(async { uzu::engine::Engine::new(uzu::engine::EngineConfig::default()).await });

    application()
        // Real HTTP client so the image cache can fetch remote provider icons.
        .with_http_client(std::sync::Arc::new(reqwest_client::ReqwestClient::new()))
        .with_assets(assets::Assets::new())
        .run(move |cx: &mut App| {
            theme::init(cx);
            settings_state::init(cx);
            toast::init(cx);
            components::text_input::register(cx);

            cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.set_menus([Menu::new("Mirai").items([MenuItem::action("Quit Mirai", Quit)])]);

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
