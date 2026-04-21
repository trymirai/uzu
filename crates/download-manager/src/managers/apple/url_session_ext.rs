use crate::{
    DownloadError,
    managers::apple::{URLSessionDownloadTaskResumeData, URLSessionGetTasksCompletionHandler},
    prelude::*,
};

pub struct URLSessionTasks {
    pub download_tasks: Box<[Retained<NSURLSessionDownloadTask>]>,
}

impl Default for URLSessionTasks {
    fn default() -> Self {
        Self {
            download_tasks: Box::default(),
        }
    }
}

#[async_trait::async_trait]
pub trait URLSessionExt {
    async fn get_tasks(&self) -> URLSessionTasks;
    fn download_task_with_url(
        &self,
        source_url: &str,
    ) -> Result<Retained<NSURLSessionDownloadTask>, DownloadError>;
    fn download_task_with_resume_data(
        &self,
        resume_data: &URLSessionDownloadTaskResumeData,
    ) -> Result<Retained<NSURLSessionDownloadTask>, DownloadError>;
}

#[async_trait::async_trait]
impl URLSessionExt for NSURLSession {
    async fn get_tasks(&self) -> URLSessionTasks {
        let receiver = {
            let (sender, receiver) = tokio_oneshot_channel();
            let sender = Arc::new(Mutex::new(Some(sender)));
            let handler = URLSessionGetTasksCompletionHandler::new({
                let sender = sender.clone();
                move |_data_tasks, _upload_tasks, download_tasks| {
                    if let Some(s) = sender.lock().unwrap().take() {
                        let _ = s.send(URLSessionTasks {
                            download_tasks,
                        });
                    }
                }
            });
            unsafe { self.getTasksWithCompletionHandler(&handler) };
            receiver
        };
        receiver.await.unwrap_or(URLSessionTasks::default())
    }

    fn download_task_with_url(
        &self,
        source_url: &str,
    ) -> Result<Retained<NSURLSessionDownloadTask>, DownloadError> {
        autoreleasepool(|_| {
            let nsurl = NSURL::URLWithString(&NSString::from_str(source_url)).ok_or(DownloadError::BadUrl)?;
            Ok::<_, DownloadError>(self.downloadTaskWithURL(&nsurl))
        })
    }

    fn download_task_with_resume_data(
        &self,
        resume_data: &URLSessionDownloadTaskResumeData,
    ) -> Result<Retained<NSURLSessionDownloadTask>, DownloadError> {
        let bytes = resume_data.to_bytes().map_err(|_| DownloadError::ResumeDataError)?;
        autoreleasepool(|_| {
            let nsdata = NSData::with_bytes(&bytes);
            Ok::<_, DownloadError>(self.downloadTaskWithResumeData(&nsdata))
        })
    }
}
