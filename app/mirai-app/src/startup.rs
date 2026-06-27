//! "Run on startup" — registers the installed `Mirai.app` as a macOS login
//! item via the `auto-launch` crate (Electron parity for
//! `app.setLoginItemSettings`). Only meaningful when running from a `.app`
//! bundle; a bare `cargo run` binary has no bundle path, so the helpers no-op
//! and `status()` returns `None` (so we never clobber the persisted toggle in
//! dev). The OS is the source of truth — `status()` reads the live login-item
//! state, mirroring Electron's `getLoginItemSettings().openAtLogin`.

use auto_launch::AutoLaunchBuilder;

/// Build an `AutoLaunch` targeting the enclosing `.app` bundle, or `None` when
/// not running from a bundle.
fn auto_launch() -> Option<auto_launch::AutoLaunch> {
    let exe = std::env::current_exe().ok()?;
    // `.../Mirai.app/Contents/MacOS/mirai-app` → `.../Mirai.app`
    let bundle = exe
        .ancestors()
        .find(|p| p.extension().is_some_and(|e| e == "app"))?;
    AutoLaunchBuilder::new()
        .set_app_name("Mirai")
        .set_app_path(bundle.to_str()?)
        .build()
        .ok()
}

/// Current OS login-item state, or `None` if it can't be determined (e.g. not
/// bundled). Callers should leave the persisted preference untouched on `None`.
pub fn status() -> Option<bool> {
    auto_launch().and_then(|al| al.is_enabled().ok())
}

/// Enable or disable launch-at-login. Best-effort: silently no-ops when not
/// bundled or if the OS call fails.
pub fn set(enabled: bool) {
    if let Some(al) = auto_launch() {
        let _ = if enabled { al.enable() } else { al.disable() };
    }
}
