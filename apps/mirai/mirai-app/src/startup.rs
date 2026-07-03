use auto_launch::AutoLaunchBuilder;

fn auto_launch() -> Option<auto_launch::AutoLaunch> {
    let exe = std::env::current_exe().ok()?;

    let bundle = exe.ancestors().find(|p| p.extension().is_some_and(|e| e == "app"))?;
    AutoLaunchBuilder::new().set_app_name("Mirai").set_app_path(bundle.to_str()?).build().ok()
}

pub fn status() -> Option<bool> {
    auto_launch().and_then(|al| al.is_enabled().ok())
}

pub fn set(enabled: bool) {
    if let Some(al) = auto_launch() {
        let _ = if enabled {
            al.enable()
        } else {
            al.disable()
        };
    }
}
