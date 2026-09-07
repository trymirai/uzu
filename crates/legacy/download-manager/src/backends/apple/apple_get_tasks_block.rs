use std::ptr::NonNull;

use objc2_foundation::{NSArray, NSURLSessionDataTask, NSURLSessionDownloadTask, NSURLSessionUploadTask};

pub type AppleGetTasksBlock = dyn Fn(
    NonNull<NSArray<NSURLSessionDataTask>>,
    NonNull<NSArray<NSURLSessionUploadTask>>,
    NonNull<NSArray<NSURLSessionDownloadTask>>,
);
