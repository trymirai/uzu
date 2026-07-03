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
mod text;
mod title_gen;
mod toast;
mod tts_history;
use gpui::{App, KeyBinding, Menu, MenuItem, actions};
use gpui_platform::application;
pub(crate) use ui_kit::{components, theme, tokens};

actions!(mirai, [Quit]);

fn main() {
    let app = application()
        .with_http_client(std::sync::Arc::new(reqwest_client::ReqwestClient::new()))
        .with_assets(assets::Assets::new());
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

        gpui_tokio::init(cx);
        let engine = gpui_tokio::Tokio::handle(cx).block_on(async {
            let config = uzu::engine::EngineConfig::default()
                .with_application_identifier(provider_keys::APPLICATION_ID.to_string());
            uzu::engine::Engine::new(config).await
        });
        match engine {
            Ok(engine) => {
                engine.set_usage_reporting(settings_state::current(cx).share_usage_data);
                crate::engine::init(cx, engine);
            },
            Err(error) => eprintln!("uzu engine init failed: {error}"),
        }

        menu_bar::init(cx);
        app_shell::open_window(cx);
    });
}
