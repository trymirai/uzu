use std::path::PathBuf;

#[allow(unreachable_code)]
pub fn home_path() -> Option<PathBuf> {
    #[cfg(all(target_os = "ios", not(target_abi = "macabi")))]
    {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::{NSFileManager, NSSearchPathDirectory, NSSearchPathDomainMask};

        let path = autoreleasepool(|_| {
            let file_manager = NSFileManager::defaultManager();
            let urls = file_manager.URLsForDirectory_inDomains(
                NSSearchPathDirectory::CachesDirectory,
                NSSearchPathDomainMask::UserDomainMask,
            );
            urls.firstObject().and_then(|url| url.path().map(|path| PathBuf::from(path.to_string())))
        });
        return path;
    }
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSHomeDirectory;

        let path = NSHomeDirectory();
        return Some(PathBuf::from(path.to_string()));
    }
    // Inside an app, Kotlin's `Uzu.initialize` sets the storage root; shell processes keep HOME.
    #[cfg(target_os = "android")]
    if let Some(path) = crate::android::home_path() {
        return Some(path);
    }
    #[cfg(not(target_family = "wasm"))]
    return dirs::home_dir();

    #[cfg(target_family = "wasm")]
    Some(PathBuf::from("/"))
}
