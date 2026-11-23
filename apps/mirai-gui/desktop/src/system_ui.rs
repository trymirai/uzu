use tauri::{
    AppHandle, Emitter, Manager,
    menu::{AboutMetadata, MenuBuilder, MenuItem, SubmenuBuilder, WINDOW_SUBMENU_ID},
};

#[cfg(not(target_os = "macos"))]
use crate::error::{AppError, AppResult};

const MENU_NEW_CHAT_ID: &str = "menu:new-chat";
const MENU_SETTINGS_ID: &str = "menu:settings";
const MENU_ZOOM_IN_ID: &str = "menu:zoom-in";
const MENU_ZOOM_OUT_ID: &str = "menu:zoom-out";
const MENU_ZOOM_RESET_ID: &str = "menu:zoom-reset";

// Before the startup setting was removed, tauri-plugin-autostart used the
// product name (Mirai) and LaunchAgent mode. Retire only that registration for
// this executable; a similarly named agent may belong to something else.
pub fn remove_legacy_autostart() -> std::io::Result<()> {
    let Some(home) = std::env::var_os("HOME") else {
        return Ok(());
    };
    let path = std::path::PathBuf::from(home).join("Library/LaunchAgents/Mirai.plist");
    remove_legacy_autostart_at(&path, &std::env::current_exe()?.canonicalize()?)
}

fn remove_legacy_autostart_at(
    path: &std::path::Path,
    executable: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Read;

    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if !metadata.is_file() || metadata.len() > 16 * 1024 {
        return Ok(());
    }
    let Ok(entry) = plist::Value::from_reader_xml(std::fs::File::open(path)?.take(16 * 1024)) else {
        return Ok(());
    };
    let Some(entry) = entry.as_dictionary() else {
        return Ok(());
    };
    let arguments = entry.get("ProgramArguments").and_then(plist::Value::as_array);
    if entry.len() == 3
        && entry.get("Label").and_then(plist::Value::as_string) == Some("Mirai")
        && entry.get("RunAtLoad").and_then(plist::Value::as_boolean) == Some(true)
        && arguments.is_some_and(|arguments| {
            arguments.len() == 1 && arguments[0].as_string().map(std::path::Path::new) == Some(executable)
        })
    {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

pub(crate) fn show_and_focus_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

pub fn setup_app_menu(app: &AppHandle) -> tauri::Result<()> {
    let package = app.package_info();
    let bundle = &app.config().bundle;
    let about = AboutMetadata {
        name: Some(package.name.clone()),
        version: Some(package.version.to_string()),
        copyright: bundle.copyright.clone(),
        authors: bundle.publisher.clone().map(|publisher| vec![publisher]),
        ..Default::default()
    };
    let settings = MenuItem::with_id(app, MENU_SETTINGS_ID, "Settings…", true, Some("CmdOrCtrl+,"))?;
    let file_menu = SubmenuBuilder::new(app, "File")
        .item(&MenuItem::with_id(app, MENU_NEW_CHAT_ID, "New Chat", true, Some("CmdOrCtrl+N"))?)
        .separator();
    #[cfg(target_os = "macos")]
    let file_menu = file_menu.close_window();
    #[cfg(not(target_os = "macos"))]
    let file_menu = file_menu.item(&settings).separator().quit();
    let file_menu = file_menu.build()?;
    let edit_menu =
        SubmenuBuilder::new(app, "Edit").undo().redo().separator().cut().copy().paste().select_all().build()?;
    let view_menu = SubmenuBuilder::new(app, "View")
        .item(&MenuItem::with_id(app, MENU_ZOOM_RESET_ID, "Actual Size", true, Some("CmdOrCtrl+0"))?)
        .item(&MenuItem::with_id(app, MENU_ZOOM_IN_ID, "Zoom In", true, Some("CmdOrCtrl+="))?)
        .item(&MenuItem::with_id(app, MENU_ZOOM_OUT_ID, "Zoom Out", true, Some("CmdOrCtrl+-"))?);
    #[cfg(target_os = "macos")]
    let view_menu = view_menu.separator().fullscreen();
    let view_menu = view_menu.build()?;
    let window_menu = SubmenuBuilder::with_id(app, WINDOW_SUBMENU_ID, "Window").minimize().maximize();
    #[cfg(target_os = "macos")]
    let window_menu = window_menu.separator().bring_all_to_front();
    let window_menu = window_menu.build()?;
    let menu = MenuBuilder::new(app);
    #[cfg(target_os = "macos")]
    let menu = menu.item(
        &SubmenuBuilder::new(app, &package.name)
            .about(Some(about))
            .separator()
            .item(&settings)
            .separator()
            .services()
            .separator()
            .hide()
            .hide_others()
            .show_all()
            .separator()
            .quit()
            .build()?,
    );
    let menu = menu.items(&[&file_menu, &edit_menu, &view_menu, &window_menu]);
    let help_menu = SubmenuBuilder::with_id(app, tauri::menu::HELP_SUBMENU_ID, "Help");
    #[cfg(not(target_os = "macos"))]
    let help_menu = help_menu.about_with_text(format!("About {}", package.name), Some(about));
    let menu = menu.item(&help_menu.build()?).build()?;
    app.set_menu(menu)?;
    app.on_menu_event(|app, event| {
        on_menu_item(app, event.id().as_ref());
    });
    Ok(())
}

fn on_menu_item(
    app: &AppHandle,
    id: &str,
) -> bool {
    match id {
        MENU_NEW_CHAT_ID => {
            show_and_focus_main(app);
            let _ = app.emit("app:new-chat", ());
        },
        MENU_SETTINGS_ID => {
            show_and_focus_main(app);
            let _ = app.emit("app:open-preferences", ());
        },
        MENU_ZOOM_IN_ID => {
            crate::zoom::step(app, crate::zoom::Step::In);
        },
        MENU_ZOOM_OUT_ID => {
            crate::zoom::step(app, crate::zoom::Step::Out);
        },
        MENU_ZOOM_RESET_ID => {
            crate::zoom::step(app, crate::zoom::Step::Reset);
        },
        _ => return false,
    }
    true
}

#[cfg(not(target_os = "macos"))]
fn submenus(app: &AppHandle) -> AppResult<Vec<tauri::menu::Submenu<tauri::Wry>>> {
    let menu = app.menu().ok_or_else(|| AppError::msg("the app menu is not set up"))?;
    Ok(menu.items().map_err(AppError::msg)?.iter().filter_map(|item| item.as_submenu().cloned()).collect())
}

#[cfg(not(target_os = "macos"))]
#[tauri::command]
pub fn menu_titles(app: AppHandle) -> AppResult<Vec<String>> {
    submenus(&app)?.iter().map(|submenu| submenu.text().map_err(AppError::msg)).collect()
}

#[cfg(windows)]
#[tauri::command]
pub fn run_menu_item(
    app: AppHandle,
    id: String,
) -> AppResult<()> {
    if on_menu_item(&app, &id) {
        Ok(())
    } else {
        Err(AppError::msg(format!("no menu item {id}")))
    }
}

#[cfg(not(target_os = "macos"))]
#[tauri::command]
pub fn open_menu(
    app: AppHandle,
    window: tauri::Window,
    index: usize,
    left: f64,
    bottom: f64,
) -> AppResult<()> {
    let submenus = submenus(&app)?;
    let submenu = submenus.get(index).ok_or_else(|| AppError::msg(format!("no app menu {index}")))?;
    let zoom = crate::zoom::current(&app);
    window.popup_menu_at(submenu, tauri::LogicalPosition::new(left * zoom, bottom * zoom)).map_err(AppError::msg)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindowTheme {
    System,
    Light,
    Dark,
}

// Follow the OS in system mode. Explicit app themes also set the native
// appearance so macOS paints the traffic lights for the matching background.
#[tauri::command]
pub fn set_window_theme(
    app: AppHandle,
    theme: WindowTheme,
) -> bool {
    let theme = match theme {
        WindowTheme::System => None,
        WindowTheme::Light => Some(tauri::Theme::Light),
        WindowTheme::Dark => Some(tauri::Theme::Dark),
    };
    app.get_webview_window("main").map(|w| w.set_theme(theme).is_ok()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autostart_cleanup_only_removes_the_released_apps_own_registration() {
        let directory = std::env::temp_dir().join(format!("mirai-autostart-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("Mirai.plist");
        let executable = std::path::Path::new("/Applications/Mirai.app/Contents/MacOS/Mirai");
        let original = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>Mirai</string>
<key>ProgramArguments</key><array><string>{}</string></array>
<key>RunAtLoad</key><true/>
</dict></plist>"#,
            executable.display()
        );
        remove_legacy_autostart_at(&path, executable).unwrap();
        for contents in [
            original.replace("<string>Mirai</string>", "<string>Other</string>"),
            original.replace("/Applications/Mirai.app", "/Applications/Other.app"),
            original.replace("</array>", "<string>--custom</string></array>"),
            original.replace("<true/>", "<false/>"),
            original.replace("</dict>", "<key>KeepAlive</key><true/></dict>"),
            "invalid plist".into(),
            " ".repeat(16 * 1024 + 1),
        ] {
            std::fs::write(&path, &contents).unwrap();
            remove_legacy_autostart_at(&path, executable).unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap(), contents);
        }
        std::fs::write(&path, &original).unwrap();
        remove_legacy_autostart_at(&path, executable).unwrap();
        assert!(!path.exists());
        #[cfg(unix)]
        {
            let other = directory.join("other.plist");
            std::fs::write(&other, &original).unwrap();
            std::os::unix::fs::symlink(&other, &path).unwrap();
            remove_legacy_autostart_at(&path, executable).unwrap();
            assert_eq!(std::fs::read_to_string(&other).unwrap(), original);
            assert!(std::fs::symlink_metadata(&path).unwrap().is_symlink());
        }
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
