use std::{
    collections::HashMap,
    fs::{copy, create_dir_all, remove_file, rename},
    io::Error as IoError,
    path::PathBuf,
    ptr::{from_ref, null_mut},
    sync::{Arc, Mutex},
};

use block2::DynBlock;
use http::{StatusCode, uri::Scheme};
use kiban::rt::RuntimeHandle;
use objc2::{
    ClassType, DefinedClass, define_class, msg_send,
    rc::{Allocated, Retained},
    runtime::ProtocolObject,
};
use objc2_foundation::{
    NSError, NSHTTPURLResponse, NSMutableCopying, NSObject, NSObjectProtocol, NSString, NSURL, NSURLRequest,
    NSURLSession, NSURLSessionDelegate, NSURLSessionDownloadDelegate, NSURLSessionDownloadTask, NSURLSessionTask,
    NSURLSessionTaskDelegate,
};
use reqwest::Url;

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
    pub expected_bytes: Option<u64>,
    pub backend_event_sender: BackendEventSender,
    pub runtime_handle: RuntimeHandle,
}

impl AppleEventSink {
    fn send_terminal(
        self,
        event: BackendEvent,
    ) {
        self.runtime_handle.clone().spawn(async move {
            let _ = self.backend_event_sender.send_terminal(event).await;
        });
    }
}

pub type AppleSinkKey = (DownloadId, u64);
pub type AppleEventRegistry = Arc<Mutex<HashMap<AppleSinkKey, AppleEventSink>>>;

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

    unsafe impl NSURLSessionDelegate for AppleSessionDelegate {}

    unsafe impl NSURLSessionTaskDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:task:willPerformHTTPRedirection:newRequest:completionHandler:))]
        unsafe fn will_perform_http_redirection(
            &self,
            _session: &NSURLSession,
            _task: &NSURLSessionTask,
            response: &NSHTTPURLResponse,
            request: &NSURLRequest,
            completion_handler: &DynBlock<dyn Fn(*mut NSURLRequest)>,
        ) {
            let current_url =
                response.URL().and_then(|url| url.absoluteString()).and_then(|url| Url::parse(&url.to_string()).ok());
            let next_url =
                request.URL().and_then(|url| url.absoluteString()).and_then(|url| Url::parse(&url.to_string()).ok());
            let redirected_request = match (current_url, next_url) {
                (Some(current), Some(next)) => {
                    match (Scheme::try_from(current.scheme()), Scheme::try_from(next.scheme())) {
                        (Ok(current_scheme), Ok(next_scheme))
                            if (current_scheme == Scheme::HTTP
                                && (next_scheme == Scheme::HTTP || next_scheme == Scheme::HTTPS))
                                || (current_scheme == Scheme::HTTPS && next_scheme == Scheme::HTTPS) =>
                        {
                            if current.origin() == next.origin() {
                                from_ref(request).cast_mut()
                            } else {
                                let redirected = request.mutableCopy();
                                redirected.setValue_forHTTPHeaderField(None, &NSString::from_str("Authorization"));
                                let pointer: *const _ = &*redirected;
                                completion_handler.call((pointer.cast::<NSURLRequest>().cast_mut(),));
                                return;
                            }
                        },
                        _ => null_mut(),
                    }
                },
                _ => null_mut(),
            };
            completion_handler.call((redirected_request,));
        }

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
            let key: AppleSinkKey = (download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&key))
            else {
                return;
            };
            let message = error.localizedDescription().to_string();
            let event = BackendEvent::error(sink.generation, message);
            sink.send_terminal(event);
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
            let key: AppleSinkKey = (download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&key))
            else {
                return;
            };
            let status_code = download_task
                .response()
                .and_then(|response| response.downcast_ref::<NSHTTPURLResponse>().map(|response| response.statusCode()))
                .and_then(|status_code| u16::try_from(status_code).ok())
                .and_then(|status_code| StatusCode::from_u16(status_code).ok());
            if !status_code.is_some_and(|status_code| status_code.is_success()) {
                let message = status_code.map_or_else(
                    || "download response was not HTTP".to_string(),
                    |status_code| format!("download failed with HTTP status {}", status_code.as_u16()),
                );
                let event = BackendEvent::error(sink.generation, message);
                sink.send_terminal(event);
                return;
            }
            let Some(temporary_path) = location.path().map(|path| PathBuf::from(path.to_string())) else {
                let event =
                    BackendEvent::error(sink.generation, "download temporary location was unavailable".to_string());
                sink.send_terminal(event);
                return;
            };

            // Move synchronously: NSURLSession deletes the temp file as soon as this method returns.
            if let Some(parent) = sink.destination.parent() {
                let _ = create_dir_all(parent);
            }
            let move_result = rename(&temporary_path, &sink.destination).or_else(|_| {
                copy(&temporary_path, &sink.destination)?;
                let _ = remove_file(&temporary_path);
                Ok::<(), IoError>(())
            });

            let terminal_event = match move_result {
                Ok(()) => BackendEvent::completed(sink.generation),
                Err(error) => BackendEvent::error(sink.generation, format!("move into destination failed: {error}")),
            };
            sink.send_terminal(terminal_event);
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
            let key: AppleSinkKey = (download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|registry| registry.get(&key).cloned())
            else {
                return;
            };
            let downloaded_bytes = u64::try_from(cumulative_bytes_written).unwrap_or_default();
            let total_bytes = u64::try_from(total_expected_bytes_to_write).ok().filter(|total_bytes| *total_bytes > 0);
            if let Some(expected_bytes) = sink.expected_bytes
                && downloaded_bytes > expected_bytes
            {
                download_task.cancel();
                if let Ok(mut registry) = Self::ivars(self).event_registry.lock() {
                    registry.remove(&key);
                }
                let event = BackendEvent::error(
                    sink.generation,
                    format!("response exceeded the registry size of {expected_bytes} bytes"),
                );
                sink.send_terminal(event);
                return;
            }

            sink.runtime_handle.clone().spawn(async move {
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
