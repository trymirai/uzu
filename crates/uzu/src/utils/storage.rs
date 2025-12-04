use std::path::PathBuf;

use objc2::rc::Retained;
use objc2_foundation::{NSHomeDirectory, NSString};

/// `NSSearchPathDirectory` values (Darwin) so they can be passed directly to
/// `NSFileManager::URLsForDirectory:inDomains:` via an `as u64` cast.
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum NSSearchPathDirectory {
    Application = 1,
    DemoApplication = 2,
    DeveloperApplication = 3,
    AdminApplication = 4,
    Library = 5,
    Developer = 6,
    User = 7,
    Documentation = 8,
    Documents = 9,
    CoreServices = 10,
    AutosavedInformation = 11,
    Desktop = 12,
    Caches = 13,
    ApplicationSupport = 14,
    Downloads = 15,
    InputMethods = 16,
    Movies = 17,
    Music = 18,
    Pictures = 19,
    PrinterDescription = 20,
    SharedPublic = 21,
    PreferencePanes = 22,
    ApplicationScripts = 23,
    ItemReplacement = 99,
    AllApplications = 100,
    AllLibraries = 101,
}

/// Directory name used for cached model storage.
const STORAGE_DIR_NAME: &str = "com.mirai.sdk.storage";

/// Returns the *user-domain* path for a given directory without any
/// application-specific suffixes.
pub fn user_domain_path(dir: NSSearchPathDirectory) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use objc2::{class, msg_send, rc::Retained};
        use objc2_foundation::{NSArray, NSFileManager, NSString, NSURL};

        unsafe {
            if let Some(path) = objc2::rc::autoreleasepool(|_| {
                let fm: Retained<NSFileManager> =
                    msg_send![class!(NSFileManager), defaultManager];

                const NS_USER_DOMAIN_MASK: u64 = 1;
                let urls: Retained<NSArray<NSURL>> = msg_send![&*fm, URLsForDirectory: dir as u64, inDomains: NS_USER_DOMAIN_MASK];
                let url_opt: Option<Retained<NSURL>> =
                    msg_send![&*urls, firstObject];

                url_opt.map(|url| {
                    let ns_path: Retained<NSString> = msg_send![&*url, path];
                    PathBuf::from(ns_path.to_string())
                })
            }) {
                return path;
            }
        }
    }

    // Fallback for non-Darwin or if the Objective-C call failed.
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    match dir {
        NSSearchPathDirectory::Caches => {
            #[cfg(target_os = "macos")]
            {
                PathBuf::from(&home).join("Library").join("Caches")
            }
            #[cfg(not(target_os = "macos"))]
            {
                PathBuf::from(&home).join(".cache")
            }
        },
        NSSearchPathDirectory::Downloads => {
            PathBuf::from(&home).join("Downloads")
        },
        NSSearchPathDirectory::Documents => {
            PathBuf::from(&home).join("Documents")
        },
        _ => PathBuf::from(home),
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
        let home: String = unsafe {
            let ns_str: Retained<NSString> = NSHomeDirectory();
            ns_str.to_string()
        };

        PathBuf::from(&home).join("Library").join("Caches")
    };

    #[cfg(target_os = "ios")]
    let base = user_domain_path(NSSearchPathDirectory::Documents);

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
    match location {
        NSSearchPathDirectory::Caches => storage_path(),
        NSSearchPathDirectory::Downloads => {
            user_domain_path(NSSearchPathDirectory::Downloads)
        },
        NSSearchPathDirectory::Documents => {
            user_domain_path(NSSearchPathDirectory::Documents)
        },
        _ => user_domain_path(location),
    }
}
