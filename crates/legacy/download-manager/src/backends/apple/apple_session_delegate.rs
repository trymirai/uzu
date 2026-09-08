use std::{path::PathBuf, sync::PoisonError};

use objc2::{
    ClassType, DefinedClass, define_class, msg_send,
    rc::{Allocated, Retained},
};
use objc2_foundation::{
    NSError, NSObject, NSObjectProtocol, NSURL, NSURLSession, NSURLSessionDelegate, NSURLSessionDownloadDelegate,
    NSURLSessionDownloadTask, NSURLSessionTask, NSURLSessionTaskDelegate,
};

use crate::backends::{
    BackendEvent,
    apple::{AppleEventRegistry, AppleSessionDelegateIvars},
};

define_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    #[ivars = AppleSessionDelegateIvars]
    pub struct AppleSessionDelegate;

    unsafe impl NSObjectProtocol for AppleSessionDelegate {}

    unsafe impl NSURLSessionDelegate for AppleSessionDelegate {}

    unsafe impl NSURLSessionTaskDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:task:didCompleteWithError:))]
        fn did_complete_with_error(
            &self,
            _session: &NSURLSession,
            task: &NSURLSessionTask,
            error: Option<&NSError>,
        ) {
            let Some(error) = error else {
                return;
            };
            let Some(sink) = Self::ivars(self)
                .event_registry
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&task.taskIdentifier())
            else {
                return;
            };
            let message = error.localizedDescription().to_string();
            sink.runtime_handle.clone().spawn(async move {
                sink.events
                    .send_terminal(BackendEvent::Error {
                        generation: sink.generation,
                        message,
                    })
                    .await;
            });
        }
    }

    unsafe impl NSURLSessionDownloadDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:downloadTask:didFinishDownloadingToURL:))]
        fn did_finish_downloading_to_url(
            &self,
            _session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            location: &NSURL,
        ) {
            let Some(sink) = Self::ivars(self)
                .event_registry
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&download_task.taskIdentifier())
            else {
                return;
            };
            let Some(temporary_path) = location.path().map(|path| PathBuf::from(path.to_string())) else {
                return;
            };
            if let Some(parent) = sink.destination.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let moved = std::fs::rename(&temporary_path, &sink.destination).or_else(|_| {
                std::fs::copy(&temporary_path, &sink.destination)?;
                let _ = std::fs::remove_file(&temporary_path);
                Ok::<(), std::io::Error>(())
            });
            let event = match moved {
                Ok(()) => BackendEvent::Completed {
                    generation: sink.generation,
                },
                Err(error) => BackendEvent::Error {
                    generation: sink.generation,
                    message: format!("move into destination failed: {error}"),
                },
            };
            sink.runtime_handle.clone().spawn(async move {
                sink.events.send_terminal(event).await;
            });
        }

        #[unsafe(method(URLSession:downloadTask:didWriteData:totalBytesWritten:totalBytesExpectedToWrite:))]
        fn did_write_data(
            &self,
            _session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            _bytes_written_since_last_callback: i64,
            cumulative_bytes_written: i64,
            total_expected_bytes_to_write: i64,
        ) {
            let Some(sink) = Self::ivars(self)
                .event_registry
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(&download_task.taskIdentifier())
                .cloned()
            else {
                return;
            };
            let downloaded_bytes = cumulative_bytes_written.max(0) as u64;
            let total_bytes = (total_expected_bytes_to_write > 0).then_some(total_expected_bytes_to_write as u64);
            sink.events.send_progress(sink.generation, downloaded_bytes, total_bytes);
        }
    }
);

impl AppleSessionDelegate {
    pub fn new(event_registry: AppleEventRegistry) -> Retained<Self> {
        unsafe {
            let allocated: Allocated<Self> = msg_send![Self::class(), alloc];
            let allocated_with_ivars = allocated.set_ivars(AppleSessionDelegateIvars {
                event_registry,
            });
            msg_send![super(allocated_with_ivars), init]
        }
    }
}
