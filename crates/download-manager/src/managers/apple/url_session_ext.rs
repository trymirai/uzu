use crate::{
    DownloadError,
    managers::apple::{URLSessionDownloadTaskResumeData, URLSessionGetTasksCompletionHandler},
    prelude::*,
};

#[allow(unused)]
pub struct URLSessionTasks {
    pub data_tasks: Box<[Retained<NSURLSessionDataTask>]>,
    pub upload_tasks: Box<[Retained<NSURLSessionUploadTask>]>,
    pub download_tasks: Box<[Retained<NSURLSessionDownloadTask>]>,
}

impl Default for URLSessionTasks {
    fn default() -> Self {
        Self {
            data_tasks: Box::default(),
            upload_tasks: Box::default(),
            download_tasks: Box::default(),
        }
    }
}

#[allow(unused)]
#[async_trait::async_trait]
pub trait URLSessionExt {
    async fn get_tasks(&self) -> URLSessionTasks;
    fn get_tasks_blocking(&self) -> URLSessionTasks;
    fn description(&self) -> Option<String>;
    fn set_description(
        &self,
        s: &str,
    );
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
                move |data_tasks, upload_tasks, download_tasks| {
                    if let Some(s) = sender.lock().unwrap().take() {
                        let _ = s.send(URLSessionTasks {
                            data_tasks,
                            upload_tasks,
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

    fn get_tasks_blocking(&self) -> URLSessionTasks {
        use std::sync::{Arc, Mutex, mpsc::sync_channel};
        let (sender, receiver) = sync_channel(1);
        let sender = Arc::new(Mutex::new(Some(sender)));
        let handler = URLSessionGetTasksCompletionHandler::new({
            let sender = sender.clone();
            move |data_tasks, upload_tasks, download_tasks| {
                if let Some(s) = sender.lock().unwrap().take() {
                    let _ = s.send(URLSessionTasks {
                        data_tasks,
                        upload_tasks,
                        download_tasks,
                    });
                }
            }
        });
        unsafe { self.getTasksWithCompletionHandler(&handler) };
        receiver.recv().unwrap_or_default()
    }

    fn description(&self) -> Option<String> {
        let desc_opt = self.sessionDescription();
        desc_opt.map(|d| d.to_string())
    }

    fn set_description(
        &self,
        s: &str,
    ) {
        let ns = NSString::from_str(s);
        self.setSessionDescription(Some(&ns));
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
