use std::path::PathBuf;

#[allow(unreachable_code)]
pub fn home_path() -> Option<PathBuf> {
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
