use objc2::msg_send;
use objc2_foundation::{NSString, NSURLSessionDownloadTask, NSURLSessionTaskState};

use crate::{DownloadId, recovery_metadata::RecoveryMetadata};

pub(crate) trait AppleDownloadTaskExt {
    fn set_recovery_metadata(
        &self,
        metadata: &RecoveryMetadata,
    );
    fn recovery_metadata(&self) -> Option<RecoveryMetadata>;
    fn download_id(&self) -> Option<DownloadId>;
    fn state(&self) -> NSURLSessionTaskState;
    fn count_of_bytes_expected_to_receive(&self) -> u64;
    fn count_of_bytes_received(&self) -> u64;
    fn task_identifier(&self) -> u64;
}

impl AppleDownloadTaskExt for NSURLSessionDownloadTask {
    fn set_recovery_metadata(
        &self,
        metadata: &RecoveryMetadata,
    ) {
        if let Ok(json) = metadata.to_json() {
            let ns_string = NSString::from_str(&json);
            self.setTaskDescription(Some(&ns_string));
        }
    }

    fn recovery_metadata(&self) -> Option<RecoveryMetadata> {
        self.taskDescription()
            .map(|description| description.to_string())
            .and_then(|description| RecoveryMetadata::from_json(&description).ok())
    }

    fn download_id(&self) -> Option<DownloadId> {
        self.recovery_metadata().and_then(|metadata| metadata.download_id())
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

    fn task_identifier(&self) -> u64 {
        unsafe { msg_send![self, taskIdentifier] }
    }
}
