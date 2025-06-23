use std::path::PathBuf;

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn foundation_base_dir() -> Option<PathBuf> {
    use objc2::{class, msg_send, rc::Retained};
    use objc2_foundation::{NSArray, NSFileManager, NSString, NSURL};

    unsafe {
        objc2::rc::autoreleasepool(|_| {
            let fm_class = class!(NSFileManager);
            let fm: Retained<NSFileManager> =
                msg_send![fm_class, defaultManager];

            const NS_CACHES_DIRECTORY: u64 = 13;
            const NS_USER_DOMAIN_MASK: u64 = 1;

            let urls: Retained<NSArray<NSURL>> = msg_send![&*fm, URLsForDirectory: NS_CACHES_DIRECTORY, inDomains: NS_USER_DOMAIN_MASK];

            let url_opt: Option<Retained<NSURL>> =
                msg_send![&*urls, firstObject];
            let url = match url_opt {
                Some(u) => u,
                None => return None,
            };

            let ns_path: Retained<NSString> = msg_send![&*url, path];
            let rust_str: String = ns_path.to_string();
            Some(PathBuf::from(rust_str))
        })
    }
}

fn default_root_dir() -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if let Some(base) = foundation_base_dir() {
        return base.join("com.mirai.sdk.storage");
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/com.mirai.sdk.storage")
}

pub fn get_test_model_path() -> PathBuf {
    let model_path = default_root_dir().join("Llama-3.2-1B-Instruct-FP16");
    if !model_path.exists() {
        panic!(
            "Test model not found at {:?}. Please make sure the model is downloaded to the SDK storage area.",
            model_path
        );
    }
    model_path
}
