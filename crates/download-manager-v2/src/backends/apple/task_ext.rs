use std::path::PathBuf;

use objc2::msg_send;
use objc2_foundation::{NSString, NSURLSessionDownloadTask, NSURLSessionTaskState};

use crate::{DownloadId, DownloadInfo, compute_download_id};

#[allow(dead_code)]
pub trait AppleDownloadTaskExt {
    fn set_download_info(
        &self,
        info: &DownloadInfo,
    );
    fn download_info(&self) -> Option<DownloadInfo>;
    fn download_id(&self) -> Option<DownloadId>;
    fn state(&self) -> NSURLSessionTaskState;
    fn count_of_bytes_expected_to_receive(&self) -> u64;
    fn count_of_bytes_received(&self) -> u64;
}

impl AppleDownloadTaskExt for NSURLSessionDownloadTask {
    fn set_download_info(
        &self,
        info: &DownloadInfo,
    ) {
        if let Ok(json) = info.to_json() {
            let ns_string = NSString::from_str(&json);
            self.setTaskDescription(Some(&ns_string));
        }
    }

    fn download_info(&self) -> Option<DownloadInfo> {
        self.taskDescription()
            .map(|description| description.to_string())
            .and_then(|description| DownloadInfo::from_json(&description).ok())
    }

    fn download_id(&self) -> Option<DownloadId> {
        self.download_info()
            .map(|info| compute_download_id(&info.source_url, PathBuf::from(info.destination_path).as_path()))
    }

    fn state(&self) -> NSURLSessionTaskState {
        unsafe { msg_send![self, state] }
    }

    fn count_of_bytes_expected_to_receive(&self) -> u64 {
        let bytes_expected: i64 = unsafe { msg_send![self, countOfBytesExpectedToReceive] };
        bytes_expected.max(0) as u64
    }

    fn count_of_bytes_received(&self) -> u64 {
        let bytes_received: i64 = unsafe { msg_send![self, countOfBytesReceived] };
        bytes_received.max(0) as u64
    }
}
