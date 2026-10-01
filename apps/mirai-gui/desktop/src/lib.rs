mod chat;
mod cli_installer;
mod downloads;
mod engine;
mod error;
mod logger;
mod models;
mod storage;
mod system_ui;
mod updater;

#[tauri::command]
fn open_external(url: String) -> bool {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return false;
    }
    std::process::Command::new("open").arg(&url).spawn().is_ok()
}

fn is_app_url(url: &tauri::Url) -> bool {
    match url.scheme() {
        "tauri" => true,
        "http" | "https" => {
            url.host_str().is_some_and(|h| h == "localhost" || h == "127.0.0.1" || h == "tauri.localhost")
        },
        _ => false,
    }
}

// Backend crashes would otherwise vanish silently; log location + payload to
// stderr and the shared log file.
fn install_panic_logger() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!("[panic] {location}: {payload}");
        logger::write_line("ERROR", "panic", Some(serde_json::json!({ "location": location, "payload": payload })));
        default(info);
    }));
}

pub fn run() {
    install_panic_logger();
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(tauri_plugin_autostart::MacosLauncher::LaunchAgent, None))
        .setup(|app| {
            let window = tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::default())
                .title("Mirai")
                .inner_size(1200.0, 800.0)
                .min_inner_size(390.0, 400.0)
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .traffic_light_position(tauri::LogicalPosition::new(20.0, 28.0))
                // In-window navigation to an external URL would replace the app;
                // hand it to the default browser instead.
                .on_navigation(|url| {
                    if is_app_url(url) {
                        return true;
                    }
                    if matches!(url.scheme(), "http" | "https") {
                        let _ = std::process::Command::new("open").arg(url.as_str()).spawn();
                    }
                    false
                })
                .build()?;
            // Closing hides the window but keeps the process alive so the global
            // shortcut stays registered; ⌘Q quits.
            let hide_target = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested {
                    api,
                    ..
                } = event
                {
                    api.prevent_close();
                    let _ = hide_target.hide();
                }
            });
            logger::info("app:start", Some(serde_json::json!({ "version": app.package_info().version.to_string() })));
            if let Err(error) = system_ui::setup_app_menu(app.handle()) {
                logger::error("menu:setup", Some(serde_json::json!({ "error": error.to_string() })));
            }
            system_ui::restore_from_settings(app.handle());
            chat::restore_auto_eject_config(app.handle());
            cli_installer::trigger_if_needed(app.handle());
            // Warm the engine so the first models/chat request doesn't pay init latency.
            tauri::async_runtime::spawn(async {
                let _ = engine::engine().await;
            });
            downloads::auto_resume_on_startup(app.handle().clone());
            Ok(())
        })
        .manage(chat::ChatState::default())
        .manage(downloads::DownloadsState::default())
        .manage(updater::UpdaterState::default())
        .invoke_handler(tauri::generate_handler![
            open_external,
            cli_installer::cli_install,
            cli_installer::cli_status,
            models::chat_models_get,
            chat::run_stream,
            chat::title_gen,
            chat::cancel_run,
            chat::cancel_title_gen,
            chat::eject_session,
            chat::set_auto_eject_config,
            chat::chat_sampling_defaults,
            downloads::download_resume,
            downloads::download_pause,
            downloads::download_delete,
            storage::chat_list_files,
            storage::chat_load_file,
            storage::chat_save_file,
            storage::chat_delete_file,
            storage::global_instructions_save,
            storage::global_instructions_load,
            storage::save_binary_file,
            storage::read_text_file,
            logger::get_log_file_path,
            storage::settings_load,
            storage::settings_patch,
            storage::model_params_set,
            storage::cleanup::cleanup_preview,
            storage::cleanup::cleanup_execute,
            system_ui::get_run_on_startup,
            system_ui::set_run_on_startup,
            system_ui::register_quick_entry_shortcut,
            system_ui::unregister_quick_entry_shortcut,
            system_ui::get_quick_entry_shortcut,
            system_ui::set_window_theme,
            updater::update_check,
            updater::update_download,
            updater::update_apply,
            updater::update_status,
        ])
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|app, event| {
            // Clicking the dock icon while the window is hidden must bring it
            // back; macOS delivers that as Reopen, not a window event.
            if let tauri::RunEvent::Reopen {
                ..
            } = event
            {
                system_ui::show_and_focus_main(app);
            }
        });
}
