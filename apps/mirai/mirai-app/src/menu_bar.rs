//! Menu-bar status item ("Show in menu bar", Electron's `Tray`). GPUI has no
//! status-item API, so this uses `tray-icon` to own an `NSStatusItem`.
//!
//! The item and its event poll live at the *App* level (a global), not on the
//! window's root view — so closing the last window (the app stays resident; see
//! `on_reopen`) doesn't remove the menu-bar item. Clicks arrive on
//! `tray_icon::menu::MenuEvent`'s global channel, which isn't wired into GPUI's
//! run loop, so a short timer drains it and routes each id.

use gpui::{App, AsyncApp, Global, WeakEntity};
use tray_icon::{
    Icon, TrayIcon, TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
};

use crate::app_shell::{self, MiraiApp};

const OPEN: &str = "mirai.open";
const NEW_CHAT: &str = "mirai.new_chat";
const OPEN_CHATS: &str = "mirai.open_chats";
const SETTINGS: &str = "mirai.settings";
const QUIT: &str = "mirai.quit";

/// What a status-item menu click routes to in the window's root view.
#[derive(Clone, Copy)]
pub enum TrayAction {
    NewChat,
    OpenChats,
    Settings,
}

/// App-level state: the live status item (kept alive past window close) and a
/// weak handle to the current window's root view for routing clicks.
#[derive(Default)]
struct MenuBar {
    tray: Option<TrayIcon>,
    app: Option<WeakEntity<MiraiApp>>,
}
impl Global for MenuBar {}

/// Start the menu-bar reconcile + click poll. Call once at launch.
pub fn init(cx: &mut App) {
    cx.set_global(MenuBar::default());
    cx.spawn(async move |cx: &mut AsyncApp| {
        loop {
            cx.background_executor().timer(std::time::Duration::from_millis(250)).await;
            cx.update(tick);
        }
    })
    .detach();
}

/// Register the current window's root view so menu clicks can route to it.
pub fn register_app(
    cx: &mut App,
    app: WeakEntity<MiraiApp>,
) {
    if !cx.has_global::<MenuBar>() {
        cx.set_global(MenuBar::default());
    }
    cx.global_mut::<MenuBar>().app = Some(app);
}

fn tick(cx: &mut App) {
    // Reconcile the item with the "Show in menu bar" toggle.
    let want = crate::settings_state::current(cx).show_in_menu_bar;
    {
        let mb = cx.global_mut::<MenuBar>();
        if want && mb.tray.is_none() {
            mb.tray = build();
        } else if !want {
            mb.tray = None;
        }
    }
    while let Ok(event) = MenuEvent::receiver().try_recv() {
        match event.id.0.as_str() {
            QUIT => cx.quit(),
            OPEN => focus_window(cx, None),
            NEW_CHAT => focus_window(cx, Some(TrayAction::NewChat)),
            OPEN_CHATS => focus_window(cx, Some(TrayAction::OpenChats)),
            SETTINGS => focus_window(cx, Some(TrayAction::Settings)),
            _ => {},
        }
    }
}

/// Bring the app forward (opening a window if all are closed) and optionally
/// route a navigation action to the root view.
fn focus_window(
    cx: &mut App,
    action: Option<TrayAction>,
) {
    if cx.windows().is_empty() {
        app_shell::open_window(cx);
    }
    cx.activate(true);
    let Some(action) = action else {
        return;
    };
    let app = cx.global::<MenuBar>().app.clone().and_then(|app| app.upgrade());
    if let Some(app) = app {
        app.update(cx, |app, cx| app.handle_tray_action(action, cx));
    }
}

/// Build the status item, or `None` if the platform rejects it (e.g. headless).
fn build() -> Option<TrayIcon> {
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
