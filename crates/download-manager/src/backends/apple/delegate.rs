use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use objc2::{
    ClassType, DefinedClass, define_class, msg_send,
    rc::{Allocated, Retained},
    runtime::ProtocolObject,
};
use objc2_foundation::{
    NSError, NSObject, NSObjectProtocol, NSURL, NSURLSession, NSURLSessionDelegate, NSURLSessionDownloadDelegate,
    NSURLSessionDownloadTask, NSURLSessionTask, NSURLSessionTaskDelegate,
};
use tokio::runtime::Handle as TokioHandle;

use crate::{
    DownloadId,
    backends::apple::task_ext::AppleDownloadTaskExt,
    file_download_task_actor::BackendEvent,
    traits::{ActiveDownloadGeneration, BackendEventSender},
};

#[derive(Clone, Debug)]
pub struct AppleEventSink {
    pub generation: ActiveDownloadGeneration,
    pub destination: PathBuf,
    pub backend_event_sender: BackendEventSender,
    pub tokio_handle: TokioHandle,
}

pub type AppleEventRegistry = Arc<Mutex<HashMap<DownloadId, AppleEventSink>>>;

#[derive(Debug, Clone)]
pub struct AppleSessionDelegateIvars {
    pub event_registry: AppleEventRegistry,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[derive(Debug)]
    #[ivars = AppleSessionDelegateIvars]
    pub struct AppleSessionDelegate;

    unsafe impl NSObjectProtocol for AppleSessionDelegate {}

    unsafe impl NSURLSessionDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:didBecomeInvalidWithError:))]
        fn did_become_invalid_with_error(
            &self,
            _session: &NSURLSession,
            _error: Option<&NSError>,
        ) {
        }
    }

    unsafe impl NSURLSessionTaskDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:task:didCompleteWithError:))]
        fn did_complete_with_error(
            &self,
            _session: &NSURLSession,
            task: &NSURLSessionTask,
            error: Option<&NSError>,
        ) {
            let Some(download_task) = task.downcast_ref::<NSURLSessionDownloadTask>() else {
                return;
            };
            let Some(error) = error else {
                return;
            };
            let Some(download_id) = download_task.download_id() else {
                return;
            };
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&download_id))
            else {
                return;
            };
            let message = error.localizedDescription().to_string();
            sink.tokio_handle.clone().spawn(async move {
                let _ = sink.backend_event_sender.send_terminal(BackendEvent::error(sink.generation, message)).await;
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
            let Some(download_id) = download_task.download_id() else {
                return;
            };
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&download_id))
            else {
                return;
            };
            let Some(temporary_path) = location.path().map(|path| PathBuf::from(path.to_string())) else {
                return;
            };

            sink.tokio_handle.clone().spawn(async move {
                if let Some(parent) = sink.destination.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }

                let move_result = tokio::fs::rename(&temporary_path, &sink.destination).await.or_else(|_| {
                    std::fs::copy(&temporary_path, &sink.destination)?;
                    std::fs::remove_file(&temporary_path)?;
                    Ok::<(), std::io::Error>(())
                });

                let terminal_event = match move_result {
                    Ok(()) => BackendEvent::completed(sink.generation),
                    Err(error) => {
                        BackendEvent::error(sink.generation, format!("move into destination failed: {error}"))
                    },
                };
                let _ = sink.backend_event_sender.send_terminal(terminal_event).await;
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
            let Some(download_id) = download_task.download_id() else {
                return;
            };
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|registry| registry.get(&download_id).cloned())
            else {
                return;
            };
            let downloaded_bytes = cumulative_bytes_written.max(0) as u64;
            let total_bytes = (total_expected_bytes_to_write > 0).then_some(total_expected_bytes_to_write as u64);

            sink.tokio_handle.clone().spawn(async move {
                sink.backend_event_sender.send_progress(sink.generation, downloaded_bytes, total_bytes).await;
            });
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

    pub fn protocol_object(delegate: Retained<Self>) -> Retained<ProtocolObject<dyn NSURLSessionDelegate>> {
        ProtocolObject::<dyn NSURLSessionDelegate>::from_retained(delegate)
    }
}
