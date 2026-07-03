//! Mirai — native GPUI client for the Mirai desktop chat app.

mod app_shell;
mod assets;
mod data_ops;
mod device_info;
mod engine;
mod engine_capabilities;
mod menu_bar;
mod model_recommend;
mod model_sort;
mod models_store;
mod native_dialog;
mod persistence;
mod provider_keys;
mod screens;
mod settings_state;
mod startup;
mod title_gen;
mod toast;
mod tts_history;

// Design system lives in the `ui-kit` crate; re-export under the old paths.
use gpui::{App, KeyBinding, Menu, MenuItem, actions};
use gpui_platform::application;
pub(crate) use ui_kit::{components, theme, tokens};

actions!(mirai, [Quit]);

fn main() {
    // Tokio runtime for uzu, kept alive for the app's lifetime; GPUI gets its handle.
    let runtime =
        tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("failed to build Tokio runtime");
    let handle = runtime.handle().clone();

    let engine = runtime.block_on(async {
        let config =
            uzu::engine::EngineConfig::default().with_application_identifier(provider_keys::APPLICATION_ID.to_string());
        uzu::engine::Engine::new(config).await
    });

    let app = application()
        // Real HTTP client so the image cache can fetch remote provider icons.
        .with_http_client(std::sync::Arc::new(reqwest_client::ReqwestClient::new()))
        .with_assets(assets::Assets::new());
    // macOS keeps the app resident after its last window closes; re-open one
    // when the dock icon is clicked and none remain (Electron's `activate` /
    // `window-all-closed` behaviour).
    app.on_reopen(|cx| {
        if cx.windows().is_empty() {
            app_shell::open_window(cx);
        } else {
            cx.activate(true);
        }
    });
    app.run(move |cx: &mut App| {
        theme::init(cx);
        settings_state::init(cx);
        // Reflect the real OS login-item state in the persisted "run on
        // startup" toggle (the OS is the source of truth; `None` in dev).
        if let Some(enabled) = startup::status() {
            let mut s = settings_state::current(cx);
            if s.run_on_startup != enabled {
                s.run_on_startup = enabled;
                settings_state::set(cx, s);
            }
        }
        toast::init(cx);
        components::text_input::register(cx);

        cx.on_action(|_: &Quit, cx: &mut App| cx.quit());
        cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
        cx.set_menus([Menu::new("Mirai").items([MenuItem::action("Quit Mirai", Quit)])]);

        gpui_tokio::init_from_handle(cx, handle.clone());

        match &engine {
            Ok(engine) => {
                // Honor the privacy opt-in (default off) before any reporting.
                engine.set_usage_reporting(settings_state::current(cx).share_usage_data);
                crate::engine::init(cx, engine.clone());
                eprintln!("[mirai-app] uzu Engine ready");
            },
            Err(err) => eprintln!("[mirai-app] uzu Engine init failed: {err}"),
        }

        // Menu-bar item lives at App level so it survives window close.
        menu_bar::init(cx);
        app_shell::open_window(cx);
    });
}
