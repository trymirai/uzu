use tauri::{
    AppHandle, Emitter, Manager,
    menu::{AboutMetadata, HELP_SUBMENU_ID, MenuBuilder, MenuItem, SubmenuBuilder, WINDOW_SUBMENU_ID},
};

const MENU_NEW_CHAT_ID: &str = "menu:new-chat";
const MENU_SETTINGS_ID: &str = "menu:settings";

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
    let app_menu = SubmenuBuilder::new(app, &package.name)
        .about(Some(AboutMetadata {
            name: Some(package.name.clone()),
            version: Some(package.version.to_string()),
            copyright: bundle.copyright.clone(),
            authors: bundle.publisher.clone().map(|publisher| vec![publisher]),
            ..Default::default()
        }))
        .separator()
        .item(&MenuItem::with_id(app, MENU_SETTINGS_ID, "Settings…", true, Some("CmdOrCtrl+,"))?)
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .quit()
        .build()?;
    let file_menu = SubmenuBuilder::new(app, "File")
        .item(&MenuItem::with_id(app, MENU_NEW_CHAT_ID, "New Chat", true, Some("CmdOrCtrl+N"))?)
        .separator()
        .close_window()
        .build()?;
    let edit_menu =
        SubmenuBuilder::new(app, "Edit").undo().redo().separator().cut().copy().paste().select_all().build()?;
    let view_menu = SubmenuBuilder::new(app, "View").fullscreen().build()?;
    let window_menu = SubmenuBuilder::with_id(app, WINDOW_SUBMENU_ID, "Window")
        .minimize()
        .maximize()
        .separator()
        .bring_all_to_front()
        .build()?;
    let help_menu = SubmenuBuilder::with_id(app, HELP_SUBMENU_ID, "Help").build()?;
    let menu = MenuBuilder::new(app)
        .items(&[&app_menu, &file_menu, &edit_menu, &view_menu, &window_menu, &help_menu])
        .build()?;
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
