use std::path::PathBuf;

use async_trait::async_trait;

use crate::{
    DownloadId, DownloadInfo, TaskID, compute_download_id,
    managers::apple::{URLSessionDownloadTaskResumeData, URLSessionResumeDataHandler},
    prelude::*,
};

#[allow(unused)]
#[async_trait]
pub trait UrlSessionDownloadTaskExt {
    fn task_identifier(&self) -> TaskID;
    fn task_description(&self) -> Option<String>;
    fn set_task_description(
        &self,
        s: &str,
    );
    fn download_info(&self) -> Option<DownloadInfo>;
    fn set_download_info(
        &self,
        info: &DownloadInfo,
    );
    fn download_id(&self) -> Option<DownloadId>;
    fn state(&self) -> NSURLSessionTaskState;
    fn count_of_bytes_expected_to_receive(&self) -> u64;
    fn count_of_bytes_received(&self) -> u64;
    async fn cancel_by_producing_resume_data(&self) -> Result<URLSessionDownloadTaskResumeData, crate::DownloadError>;
}

#[async_trait]
impl UrlSessionDownloadTaskExt for NSURLSessionDownloadTask {
    fn task_identifier(&self) -> TaskID {
        unsafe { msg_send![self, taskIdentifier] }
    }

    fn task_description(&self) -> Option<String> {
        let desc_opt = self.taskDescription();
        desc_opt.map(|d| d.to_string())
    }

    fn set_task_description(
        &self,
        s: &str,
    ) {
        let ns = NSString::from_str(s);
        self.setTaskDescription(Some(&ns));
    }

    fn download_info(&self) -> Option<DownloadInfo> {
        self.task_description().and_then(|desc| serde_json::from_str::<crate::DownloadInfo>(&desc).ok())
    }

    fn set_download_info(
        &self,
        info: &DownloadInfo,
    ) {
        let ns = NSString::from_str(&serde_json::to_string(info).unwrap());
        self.setTaskDescription(Some(&ns));
    }

    fn download_id(&self) -> Option<DownloadId> {
        self.download_info().and_then(|info| {
            Some(compute_download_id(&info.source_url, PathBuf::from(info.destination_path).as_path()))
        })
    }

    fn state(&self) -> NSURLSessionTaskState {
        unsafe { msg_send![self, state] }
    }

    fn count_of_bytes_expected_to_receive(&self) -> u64 {
        let bytes_expected: i64 = unsafe { msg_send![self, countOfBytesExpectedToReceive] };
        if bytes_expected > 0 {
            bytes_expected as u64
        } else {
            0
        }
    }

    fn count_of_bytes_received(&self) -> u64 {
        let bytes_received: i64 = unsafe { msg_send![self, countOfBytesReceived] };
        if bytes_received > 0 {
            bytes_received as u64
        } else {
            0
        }
    }

    async fn cancel_by_producing_resume_data(&self) -> Result<URLSessionDownloadTaskResumeData, crate::DownloadError> {
        let (sender, receiver) = tokio_oneshot_channel::<Box<[u8]>>();
        let sender_cell: Arc<Mutex<Option<TokioOneshotSender<Box<[u8]>>>>> = Arc::new(Mutex::new(Some(sender)));
        {
            let sender_cell = Arc::clone(&sender_cell);
            let handler = URLSessionResumeDataHandler::new_bytes(move |bytes| {
                if let Some(s) = sender_cell.lock().unwrap().take() {
                    let _ = s.send(bytes);
                }
            });
            unsafe {
                self.cancelByProducingResumeData(&handler);
            }
        }
        let resume_data_bytes = receiver.await.map_err(|_| crate::DownloadError::ResumeDataError)?;

        let mut resume_data = URLSessionDownloadTaskResumeData::from_bytes(&resume_data_bytes)
            .map_err(|_| crate::DownloadError::ResumeDataError)?;

        resume_data.bytes_received = Some(self.count_of_bytes_received());
        resume_data.bytes_expected_to_receive = Some(self.count_of_bytes_expected_to_receive());

        Ok(resume_data)
    }
}
