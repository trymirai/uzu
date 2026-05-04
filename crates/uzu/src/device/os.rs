use std::path::PathBuf;

#[allow(unreachable_code)]
pub fn home_path() -> Option<PathBuf> {
    #[cfg(target_os = "ios")]
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
    dirs::home_dir()
}
