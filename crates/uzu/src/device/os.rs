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

#[allow(unreachable_code)]
pub fn is_environment_sandboxed() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSString;
        use objc2_security::SecTask;

        unsafe {
            let Some(task) = SecTask::from_self(None) else {
                return false;
            };
            let key = NSString::from_str("com.apple.security.app-sandbox");
            let value = task.value_for_entitlement(key.as_ref(), std::ptr::null_mut());
            return value.is_some();
        }
    }
    false
}

#[allow(unreachable_code)]
pub fn application_identifier() -> String {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSBundle;

        let bundle = NSBundle::mainBundle();
        if let Some(identifier) = bundle.bundleIdentifier() {
            return identifier.to_string();
        }
    }
    if let Ok(path) = std::env::current_exe() {
        if let Some(name) = path.file_name() {
            return name.to_string_lossy().into_owned();
        }
    }
    "default".to_string()
}

#[allow(unreachable_code)]
pub fn is_keyring_available() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_security::SecTask;

        const CS_VALID: u32 = 0x0000_0001;
        const CS_ADHOC: u32 = 0x0000_0002;
        const CS_LINKER_SIGNED: u32 = 0x0002_0000;

        unsafe {
            let Some(task) = SecTask::from_self(None) else {
                return false;
            };
            let status = task.code_sign_status();
            return (status & CS_VALID) != 0 && (status & CS_ADHOC) == 0 && (status & CS_LINKER_SIGNED) == 0;
        }
    }
    true
}
