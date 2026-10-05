use tauri::{
    AppHandle, Emitter, Manager,
    menu::{Menu, MenuItem, PredefinedMenuItem},
};
use tauri_plugin_autostart::ManagerExt as AutostartExt;
use tauri_plugin_global_shortcut::GlobalShortcutExt;

use crate::error::{AppError, AppResult};

const MENU_NEW_CHAT_ID: &str = "menu:new-chat";
const MENU_SETTINGS_ID: &str = "menu:settings";

pub(crate) fn show_and_focus_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

#[tauri::command]
pub fn get_run_on_startup(app: AppHandle) -> bool {
    app.autolaunch().is_enabled().unwrap_or(false)
}

#[tauri::command]
pub fn set_run_on_startup(
    app: AppHandle,
    value: bool,
) -> AppResult<()> {
    let manager = app.autolaunch();
    let result = if value {
        manager.enable()
    } else {
        manager.disable()
    };
    result.map_err(AppError::msg)
}

pub fn setup_app_menu(app: &AppHandle) -> tauri::Result<()> {
    let menu = Menu::default(app)?;
    for item in menu.items()? {
        let Some(submenu) = item.as_submenu() else {
            continue;
        };
        if submenu.text().unwrap_or_default() == "File" {
            submenu.insert_items(
                &[
                    &MenuItem::with_id(app, MENU_NEW_CHAT_ID, "New Chat", true, Some("CmdOrCtrl+N"))?,
                    &MenuItem::with_id(app, MENU_SETTINGS_ID, "Settings", true, Some("CmdOrCtrl+,"))?,
                    &PredefinedMenuItem::separator(app)?,
                ],
                0,
            )?;
        }
    }
    app.set_menu(menu)?;
    app.on_menu_event(|app, event| match event.id().as_ref() {
        MENU_NEW_CHAT_ID => {
            show_and_focus_main(app);
            let _ = app.emit("app:new-chat", ());
        },
        MENU_SETTINGS_ID => {
            show_and_focus_main(app);
            let _ = app.emit("app:open-preferences", ());
        },
        _ => {},
    });
    Ok(())
}

fn apply_quick_entry_shortcut(
    app: &AppHandle,
    accelerator: &str,
) -> bool {
    let shortcut = app.global_shortcut();
    if shortcut.is_registered(accelerator) {
        return true;
    }
    let handler_app = app.clone();
    let registered = shortcut
        .on_shortcut(accelerator, move |_app, _shortcut, event| {
            if event.state() == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                show_and_focus_main(&handler_app);
            }
        })
        .is_ok();
    if registered && let Some(previous) = get_quick_entry_shortcut().filter(|p| p != accelerator) {
        let _ = shortcut.unregister(previous.as_str());
    }
    registered
}

#[tauri::command]
pub fn register_quick_entry_shortcut(
    app: AppHandle,
    accelerator: String,
) -> bool {
    let ok = apply_quick_entry_shortcut(&app, &accelerator);
    if ok {
        persist_quick_entry_accelerator(Some(&accelerator));
    }
    ok
}

#[tauri::command]
pub fn unregister_quick_entry_shortcut(app: AppHandle) -> AppResult<()> {
    app.global_shortcut().unregister_all().map_err(AppError::msg)?;
    persist_quick_entry_accelerator(None);
    Ok(())
}

// The shortcut already works for this session; a failed write only loses it on
// the next launch, so log instead of failing the command.
fn persist_quick_entry_accelerator(accelerator: Option<&str>) {
    if let Err(e) = crate::storage::settings_patch(serde_json::json!({ "quickEntryAccelerator": accelerator })) {
        crate::logger::warn("shortcut:settings-write-failed", Some(serde_json::json!({ "error": e })));
    }
}

// Sync the native window appearance to the app theme: otherwise the window
// keeps the system appearance and macOS paints the inactive traffic lights for
// the wrong one (grey lights vanish on a dark background).
#[tauri::command]
pub fn set_window_theme(
    app: AppHandle,
    dark: bool,
) -> bool {
    let theme = if dark {
        tauri::Theme::Dark
    } else {
        tauri::Theme::Light
    };
    app.get_webview_window("main").map(|w| w.set_theme(Some(theme)).is_ok()).unwrap_or(false)
}

#[tauri::command]
pub fn get_quick_entry_shortcut() -> Option<String> {
    let settings = crate::storage::settings_load().ok()?;
    settings.get("quickEntryAccelerator").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(str::to_string)
}

pub fn restore_from_settings(app: &AppHandle) {
    let settings = crate::storage::settings_load().unwrap_or_else(|_| serde_json::json!({}));
    if let Some(accelerator) = settings.get("quickEntryAccelerator").and_then(|v| v.as_str())
        && !accelerator.is_empty()
        && !apply_quick_entry_shortcut(app, accelerator)
    {
        // Otherwise settings keep showing a shortcut that does not work.
        crate::logger::warn("shortcut:restore-failed", Some(serde_json::json!({ "accelerator": accelerator })));
        persist_quick_entry_accelerator(None);
    }
}
