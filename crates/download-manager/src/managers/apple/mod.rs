mod file_download_task;
mod session_config;
mod url_session_delegate;
mod url_session_download_manager;
mod url_session_download_task_ext;
mod url_session_download_task_resume_data;
mod url_session_error;
mod url_session_ext;
mod url_session_get_tasks_handler;
mod url_session_resume_data_handler;
mod url_session_state_reducer;
mod url_session_task_ext;

pub use file_download_task::FileDownloadTask;
pub use session_config::SessionConfig;
pub use url_session_delegate::URLSessionDelegate;
pub use url_session_download_manager::{URLSessionDownloadManager, URLSessionDropPolicy};
pub use url_session_download_task_resume_data::URLSessionDownloadTaskResumeData;
pub use url_session_error::URLSessionError;
pub use url_session_get_tasks_handler::URLSessionGetTasksCompletionHandler;
pub use url_session_resume_data_handler::URLSessionResumeDataHandler;

use crate::prelude::*;

pub type URLSessionDownloadTaskState = Option<NSURLSessionTaskState>;

pub use url_session_download_task_ext::UrlSessionDownloadTaskExt;
pub use url_session_ext::URLSessionExt;
