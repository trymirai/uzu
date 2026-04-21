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

pub(crate) use file_download_task::FileDownloadTask;
pub(crate) use session_config::SessionConfig;
pub(crate) use url_session_delegate::URLSessionDelegate;
pub(crate) use url_session_download_manager::URLSessionDownloadManager;
pub use url_session_download_task_resume_data::URLSessionDownloadTaskResumeData;
pub(crate) use url_session_error::URLSessionError;
pub(crate) use url_session_get_tasks_handler::URLSessionGetTasksCompletionHandler;
pub(crate) use url_session_resume_data_handler::URLSessionResumeDataHandler;

use crate::prelude::*;

pub(crate) type URLSessionDownloadTaskState = Option<NSURLSessionTaskState>;

pub(crate) use url_session_download_task_ext::UrlSessionDownloadTaskExt;
pub(crate) use url_session_ext::URLSessionExt;
