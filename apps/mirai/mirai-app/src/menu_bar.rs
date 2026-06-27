//! Menu-bar status item ("Show in menu bar", Electron's `Tray`). GPUI has no
//! status-item API, so this uses the `tray-icon` crate to own an `NSStatusItem`
//! with a context menu (Open Mirai / New Chat / Open Chats / Settings / Quit).
//!
//! Menu clicks arrive on `tray_icon::menu::MenuEvent`'s global channel, which is
//! not wired into GPUI's run loop — `app_shell` drains it from a short timer and
//! routes each id to a navigation action. The returned `TrayIcon` must be kept
//! alive (dropping it removes the item); `app_shell` stores it and drops it when
//! the toggle turns off.

use tray_icon::{
    Icon, TrayIcon, TrayIconBuilder,
    menu::{Menu, MenuItem, PredefinedMenuItem},
};

pub const OPEN: &str = "mirai.open";
pub const NEW_CHAT: &str = "mirai.new_chat";
pub const OPEN_CHATS: &str = "mirai.open_chats";
pub const SETTINGS: &str = "mirai.settings";
pub const QUIT: &str = "mirai.quit";

/// Build the status item, or `None` if the platform rejects it (e.g. headless).
pub fn build() -> Option<TrayIcon> {
    let menu = Menu::new();
    menu.append_items(&[
        &MenuItem::with_id(OPEN, "Open Mirai", true, None),
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id(NEW_CHAT, "New Chat", true, None),
        &MenuItem::with_id(OPEN_CHATS, "Open Chats", true, None),
        &MenuItem::with_id(SETTINGS, "Settings", true, None),
        &PredefinedMenuItem::separator(),
        &MenuItem::with_id(QUIT, "Quit Mirai", true, None),
    ])
    .ok()?;

    TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Mirai")
        .with_icon_as_template(true)
        .with_icon(icon()?)
        .build()
        .ok()
}

fn icon() -> Option<Icon> {
    let rgba = image::load_from_memory(include_bytes!("../assets/icons/tray.png")).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    Icon::from_rgba(rgba.into_raw(), w, h).ok()
}
