use std::path::PathBuf;

use objc2::rc::Retained;
use objc2_foundation::{NSHomeDirectory, NSSearchPathDirectory, NSString};

const STORAGE_DIR_NAME: &str = "com.mirai.sdk.storage";

/// Returns the *user-domain* path for a given directory without any
/// application-specific suffixes.
pub fn user_domain_path(dir: NSSearchPathDirectory) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use objc2_foundation::{NSFileManager, NSSearchPathDomainMask};

        if let Some(path) = objc2::rc::autoreleasepool(|_| {
            let fm = NSFileManager::defaultManager();
            let urls = fm.URLsForDirectory_inDomains(dir, NSSearchPathDomainMask::UserDomainMask);

            urls.firstObject().map(|url| {
                let ns_path: Retained<NSString> = url.path().expect("URL should have a path");
                PathBuf::from(ns_path.to_string())
            })
        }) {
            return path;
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    if dir == NSSearchPathDirectory::CachesDirectory {
        #[cfg(target_os = "macos")]
        {
            PathBuf::from(&home).join("Library").join("Caches")
        }
        #[cfg(not(target_os = "macos"))]
        {
            PathBuf::from(&home).join(".cache")
        }
    } else if dir == NSSearchPathDirectory::DownloadsDirectory {
        PathBuf::from(&home).join("Downloads")
    } else if dir == NSSearchPathDirectory::DocumentDirectory {
        PathBuf::from(&home).join("Documents")
    } else {
        PathBuf::from(home)
    }
}

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
        let home = NSHomeDirectory().to_string();
        PathBuf::from(&home).join("Library").join("Caches")
    };

    #[cfg(target_os = "ios")]
    let base = user_domain_path(NSSearchPathDirectory::DocumentDirectory);

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let base = {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join(".cache")
    };

    let full = base.join(STORAGE_DIR_NAME).join(crate::VERSION);
    let _ = std::fs::create_dir_all(&full);
    full
}

/// Deprecated: use `user_domain_path` or `storage_path` instead. Retained for
/// compatibility with existing code paths.
pub fn root_dir(location: NSSearchPathDirectory) -> PathBuf {
    if location == NSSearchPathDirectory::CachesDirectory {
        storage_path()
    } else {
        user_domain_path(location)
    }
}
