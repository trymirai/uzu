use std::path::PathBuf;

use crate::{
    FileDownloadEvent, SharedDownloadEventSender,
    managers::apple::{URLSessionError, UrlSessionDownloadTaskExt},
    prelude::*,
};

#[derive(Debug, Clone)]
pub struct URLSessionDelegateIvars {
    pub global_broadcast_sender: SharedDownloadEventSender,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    #[ivars = URLSessionDelegateIvars]
    pub struct URLSessionDelegate;

    unsafe impl NSObjectProtocol for URLSessionDelegate {}

    unsafe impl NSURLSessionDelegate for URLSessionDelegate {
        #[unsafe(method(URLSession:didBecomeInvalidWithError:))]
        fn did_become_invalid_with_error(
            &self,
            _session: &NSURLSession,
            _error: Option<&NSError>,
        ) {
        }

        #[unsafe(method(URLSessionDidFinishEventsForBackgroundURLSession:))]
        fn did_finish_events_for_background_url_session(
            &self,
            _session: &NSURLSession,
        ) {
        }
    }

    unsafe impl NSURLSessionTaskDelegate for URLSessionDelegate {
        #[unsafe(method(URLSession:task:didCompleteWithError:))]
        fn did_complete_with_error(
            &self,
            _session: &NSURLSession,
            task: &NSURLSessionTask,
            error: Option<&NSError>,
        ) {
            if let Some(download_task) = task.downcast_ref::<NSURLSessionDownloadTask>() {
                if let Some(err) = error {
                    let parsed_error = URLSessionError::from_nserror(err);

                    if parsed_error.should_ignore() {
                        return;
                    }

                    if let Some(download_id) = download_task.download_id() {
                        let error_message = parsed_error.user_message();
                        let event = FileDownloadEvent::Error {
                            message: error_message,
                        };
                        let _ = Self::ivars(self).global_broadcast_sender.send((download_id, event));
                    }
                }
            }
        }
    }

    unsafe impl NSURLSessionDownloadDelegate for URLSessionDelegate {
        #[unsafe(method(URLSession:downloadTask:didFinishDownloadingToURL:))]
        fn did_finish_downloading_to_url(
            &self,
            _session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            location: &NSURL,
        ) {
            tracing::debug!("[DELEGATE] didFinishDownloadingToURL called");

            let tmp_path = PathBuf::from(location.path().unwrap().to_string());
            if let Some(download_info) = download_task.download_info() {
                let final_destination = PathBuf::from(&download_info.destination_path);
                if let Some(parent_dir) = final_destination.parent() {
                    let _ = fs::create_dir_all(parent_dir);

                    tracing::debug!("[DELEGATE] Moving file from temp to final destination...");

                    let move_successful = match std::fs::rename(&tmp_path, &final_destination) {
                        Ok(_) => true,
                        Err(_) => match std::fs::copy(&tmp_path, &final_destination) {
                            Ok(_) => {
                                let _ = fs::remove_file(&tmp_path);
                                true
                            },
                            Err(_) => false,
                        },
                    };

                    if !move_successful {
                        tracing::debug!("[DELEGATE] ✗ File move failed");

                        if let Some(download_id) = download_task.download_id() {
                            let event = FileDownloadEvent::Error {
                                message: "move into destination failed".to_string(),
                            };
                            let _ = Self::ivars(self).global_broadcast_sender.send((download_id, event));
                        }
                        return;
                    }

                    tracing::debug!("[DELEGATE] ✓ File moved successfully");

                    if let Some(download_id) = download_task.download_id() {
                        let event = FileDownloadEvent::DownloadCompleted {
                            tmp_path: tmp_path.clone(),
                            final_destination: final_destination.clone(),
                        };

                        tracing::debug!("[DELEGATE] Broadcasting download completion event");

                        let _ = Self::ivars(self).global_broadcast_sender.send((download_id, event));

                        tracing::debug!("[DELEGATE] ✓ didFinishDownloadingToURL complete");
                    }
                }
            }
        }

        #[unsafe(method(URLSession:downloadTask:didWriteData:totalBytesWritten:totalBytesExpectedToWrite:))]
        fn did_write_data(
            &self,
            _session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            bytes_written_since_last_callback: i64,
            cumulative_bytes_written: i64,
            total_expected_bytes_to_write: i64,
        ) {
            // Clean up resume_data file on first progress event
            // This confirms NSURLSession has processed the resume_data
            if let Some(info) = download_task.download_info() {
                let dest = PathBuf::from(&info.destination_path);
                let resume_path = format!("{}.resume_data", dest.display());
                if PathBuf::from(&resume_path).exists() {
                    let _ = fs::remove_file(&resume_path);
                }
            }

            if let Some(download_id) = download_task.download_id() {
                let bytes_written = if bytes_written_since_last_callback > 0 {
                    bytes_written_since_last_callback as u64
                } else {
                    0
                };
                let total_bytes_written = if cumulative_bytes_written > 0 {
                    cumulative_bytes_written as u64
                } else {
                    0
                };
                let total_bytes_expected = if total_expected_bytes_to_write > 0 {
                    total_expected_bytes_to_write as u64
                } else {
                    0
                };

                let event = FileDownloadEvent::ProgressUpdate {
                    bytes_written,
                    total_bytes_written,
                    total_bytes_expected,
                };

                tracing::debug!(
                    "[DELEGATE] Progress: download_id={}, written={}, total={}/{} bytes",
                    download_id,
                    bytes_written,
                    total_bytes_written,
                    total_bytes_expected
                );

                let send_result = Self::ivars(self).global_broadcast_sender.send((download_id, event));

                if let Err(e) = send_result {
                    tracing::debug!("[DELEGATE] ⚠️ Failed to send progress event: {:?}", e);
                }
            }
        }
    }
);

impl URLSessionDelegate {
    pub fn new(global_broadcast_sender: SharedDownloadEventSender) -> Retained<Self> {
        unsafe {
            let class = Self::class();
            let allocated: Allocated<Self> = msg_send![class, alloc];
            let allocated_with_ivars = allocated.set_ivars(URLSessionDelegateIvars {
                global_broadcast_sender,
            });
            let initialized: Retained<Self> = msg_send![super(allocated_with_ivars), init];

            initialized
        }
    }
}
