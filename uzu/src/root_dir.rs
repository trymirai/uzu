use std::path::PathBuf;

/// Possible base folders for storing or loading model files.
#[derive(Clone, Copy, Debug)]
pub enum RootLocation {
    /// The cache directory (`~/Library/Caches` on macOS).
    Caches,
    /// The user downloads directory (`~/Downloads`).
    Downloads,
}

pub fn root_dir(location: RootLocation) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use objc2::{class, msg_send, rc::Retained};
        use objc2_foundation::{NSArray, NSFileManager, NSString, NSURL};

        unsafe {
            if let Some(root) = objc2::rc::autoreleasepool(|_| {
                let fm_class = class!(NSFileManager);
                let fm: Retained<NSFileManager> =
                    msg_send![fm_class, defaultManager];

                // NSSearchPathDirectory constants.
                const NS_CACHES_DIRECTORY: u64 = 13;
                const NS_DOWNLOADS_DIRECTORY: u64 = 15;
                const NS_USER_DOMAIN_MASK: u64 = 1;

                let dir: u64 = match location {
                    RootLocation::Caches => NS_CACHES_DIRECTORY,
                    RootLocation::Downloads => NS_DOWNLOADS_DIRECTORY,
                };

                let urls: Retained<NSArray<NSURL>> = msg_send![&*fm, URLsForDirectory: dir, inDomains: NS_USER_DOMAIN_MASK];
                let url_opt: Option<Retained<NSURL>> =
                    msg_send![&*urls, firstObject];

                url_opt.map(|url| {
                    let ns_path: Retained<NSString> = msg_send![&*url, path];
                    let mut root = PathBuf::from(ns_path.to_string());
                    if matches!(location, RootLocation::Caches) {
                        root = root.join("com.mirai.sdk.storage");
                    }
                    if matches!(location, RootLocation::Caches) {
                        let _ = std::fs::create_dir_all(&root);
                    }
                    root
                })
            }) {
                return root;
            }
        }
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    let root = match location {
        RootLocation::Caches => {
            let base = if cfg!(target_os = "macos") {
                PathBuf::from(&home).join("Library").join("Caches")
            } else {
                PathBuf::from(&home).join(".cache")
            };
            base.join("com.mirai.sdk.storage")
        },
        RootLocation::Downloads => PathBuf::from(&home).join("Downloads"),
    };
    if matches!(location, RootLocation::Caches) {
        let _ = std::fs::create_dir_all(&root);
    }
    root
}
