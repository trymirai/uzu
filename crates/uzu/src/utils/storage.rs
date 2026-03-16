use std::path::PathBuf;

const STORAGE_DIR_NAME: &str = "com.mirai.sdk.storage";

/// Platform-specific directory the SDK uses for persistent storage.
///
/// * macOS:   `~/Library/Caches/<STORAGE_DIR_NAME>/<VERSION>/`
/// * iOS:     `~/Documents/<STORAGE_DIR_NAME>/<VERSION>/`
/// * other:   `$HOME/.cache/<STORAGE_DIR_NAME>/<VERSION>/`
pub fn storage_path() -> PathBuf {
    #[cfg(target_os = "macos")]
    let base = {
        // In a sandboxed macOS application, NSHomeDirectory() returns
        // `~/Library/Containers/<bundle-id>/Data`. Persist model files inside
        // the app-scoped caches directory to comply with sandbox rules:
        // `~/Library/Containers/<bundle-id>/Data/Library/Caches/…`.
        let home = unsafe { objc2_foundation::NSHomeDirectory() }.to_string();
        PathBuf::from(&home).join("Library").join("Caches")
    };

    #[cfg(target_os = "ios")]
    let base = {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(&home).join("Documents")
    };

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let base = {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join(".cache")
    };

    let full = base.join(STORAGE_DIR_NAME).join(crate::VERSION);
    let _ = std::fs::create_dir_all(&full);
    full
}
