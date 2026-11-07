mod analytics;
mod chat;
#[cfg(target_os = "macos")]
mod cli_installer;
mod downloads;
mod engine;
mod error;
mod logger;
mod models;
mod storage;
mod system_ui;
mod updater;

const MIN_WIDTH: f64 = 390.0;
const MIN_HEIGHT: f64 = 400.0;

#[tauri::command]
fn open_external(url: String) -> bool {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return false;
    }
    open_in_browser(&url)
}

fn open_in_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    #[cfg(windows)]
    let mut command = std::process::Command::new("rundll32");
    #[cfg(windows)]
    command.arg("url.dll,FileProtocolHandler");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = std::process::Command::new("xdg-open");
    // Waiting reaps the opener, which would otherwise stay a zombie.
    command.arg(url).spawn().map(|mut child| std::thread::spawn(move || child.wait())).is_ok()
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

// Without its window the app would quit silently: WebView2 missing or broken is the usual cause on Windows.
fn report_window_failure(
    app: &tauri::AppHandle,
    error: &tauri::Error,
) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

    logger::error("window:create", Some(serde_json::json!({ "error": error.to_string() })));
    let log = logger::log_file_path().map(|path| path.display().to_string()).unwrap_or_default();
    let exit = app.clone();
    app.dialog()
        .message(format!("Mirai could not open its window: {error}\n\nThe log is at {log}"))
        .title("Mirai")
        .kind(MessageDialogKind::Error)
        .show(move |_| exit.exit(1));
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
    use tauri::Manager;

    // Tauri's AppImage exports GTK_THEME as the desktop's light or dark Adwaita, which overrides the window theme the
    // app sets; a user's APPIMAGE_GTK_THEME stays in force.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() && std::env::var_os("APPIMAGE_GTK_THEME").is_none() {
        // SAFETY: no other thread runs yet.
        unsafe { std::env::remove_var("GTK_THEME") };
    }
    install_panic_logger();
    let builder = tauri::Builder::default();
    // A second launch focuses the running window instead of opening another; on macOS LaunchServices already does.
    #[cfg(not(target_vendor = "apple"))]
    let builder = builder.plugin(tauri_plugin_single_instance::init(|app, _, _| system_ui::show_and_focus_main(app)));
    builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let window = tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::default())
                .title("Mirai")
                .inner_size(1200.0, 800.0)
                .min_inner_size(MIN_WIDTH, MIN_HEIGHT);
            #[cfg(target_os = "macos")]
            let window = window
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .traffic_light_position(tauri::LogicalPosition::new(20.0, 28.0));
            // Like the macOS overlay title bar: the client draws the caption buttons and drag regions.
            #[cfg(windows)]
            let window = window.decorations(false);
            let window = window
                // In-window navigation to an external URL would replace the app;
                // hand it to the default browser instead.
                .on_navigation(|url| {
                    if is_app_url(url) {
                        return true;
                    }
                    if matches!(url.scheme(), "http" | "https") {
                        open_in_browser(url.as_str());
                    }
                    false
                })
                // Runs of the page being replaced would keep the engine busy.
                .on_page_load(|window, payload| {
                    if payload.event() == tauri::webview::PageLoadEvent::Started {
                        window.state::<chat::ChatState>().cancel_all_runs();
                    }
                })
                .build();
            let window = match window {
                Ok(window) => window,
                Err(error) => {
                    report_window_failure(app.handle(), &error);
                    return Ok(());
                },
            };
            // The minimum would count the invisible resize borders of the frameless window. On Wayland tao's minimum
            // counts GTK's client-side title bar and shadow, which tao reports as part of the inner size, so the
            // content can be smaller there; that is tao's to fix.
            #[cfg(windows)]
            {
                let (outer, inner) = (window.outer_size()?, window.inner_size()?);
                let borders = tauri::PhysicalSize::new(outer.width - inner.width, outer.height - inner.height)
                    .to_logical::<f64>(window.scale_factor()?);
                window.set_min_size(Some(tauri::LogicalSize::new(
                    MIN_WIDTH + borders.width,
                    MIN_HEIGHT + borders.height,
                )))?;
            }
            // On macOS closing hides the window; the Dock reopens it and ⌘Q quits. Elsewhere nothing reopens a
            // hidden window, so closing keeps quitting.
            #[cfg(target_os = "macos")]
            {
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
            }
            #[cfg(not(target_os = "macos"))]
            let _ = window;
            logger::info("app:start", Some(serde_json::json!({ "version": app.package_info().version.to_string() })));
            #[cfg(target_os = "macos")]
            if let Err(error) = system_ui::remove_legacy_autostart() {
                logger::warn("autostart:cleanup", Some(serde_json::json!({ "error": error.to_string() })));
            }
            #[cfg(target_os = "macos")]
            if let Err(error) = system_ui::setup_app_menu(app.handle()) {
                logger::error("menu:setup", Some(serde_json::json!({ "error": error.to_string() })));
            }
            chat::restore_auto_eject_config(app.handle());
            app.state::<analytics::AnalyticsState>().restore();
            // Warm the engine so the first models/chat request doesn't pay init latency.
            tauri::async_runtime::spawn(async {
                let _ = engine::engine().await;
            });
            downloads::auto_resume_on_startup(app.handle().clone());
            Ok(())
        })
        .manage(chat::ChatState::default())
        .manage(analytics::AnalyticsState::default())
        .manage(downloads::DownloadsState::default())
        .manage(updater::UpdaterState::default())
        .invoke_handler(tauri::generate_handler![
            open_external,
            #[cfg(target_os = "macos")]
            cli_installer::cli_install,
            #[cfg(target_os = "macos")]
            cli_installer::cli_status,
            models::chat_models_get,
            models::chat_models_refresh,
            chat::run_stream,
            chat::title_gen,
            chat::cancel_run,
            chat::cancel_title_gen,
            chat::eject_session,
            chat::resident_session,
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
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen {
                ..
            } = event
            {
                system_ui::show_and_focus_main(app);
            }
            #[cfg(not(target_os = "macos"))]
            let _ = (app, event);
        });
}
