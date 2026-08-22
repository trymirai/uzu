use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use kiban::rt::RuntimeHandle;
use objc2::{
    ClassType, DefinedClass, define_class, msg_send,
    rc::{Allocated, Retained},
    runtime::ProtocolObject,
};
use objc2_foundation::{
    NSError, NSHTTPURLResponse, NSObject, NSObjectProtocol, NSURL, NSURLRequest, NSURLSession, NSURLSessionDelegate,
    NSURLSessionDownloadDelegate, NSURLSessionDownloadTask, NSURLSessionTask, NSURLSessionTaskDelegate,
};

use crate::{
    DownloadError, DownloadId,
    backends::apple::task_ext::AppleDownloadTaskExt,
    file_download_task_actor::BackendEvent,
    traits::{ActiveDownloadGeneration, BackendEventSender},
};

#[derive(Clone, Debug)]
pub struct AppleEventSink {
    pub generation: ActiveDownloadGeneration,
    pub destination: PathBuf,
    pub installation_artifact: PathBuf,
    pub backend_event_sender: BackendEventSender,
    pub runtime_handle: RuntimeHandle,
    pub destination_install_barrier: DestinationInstallBarrier,
}

#[derive(Clone, Debug, Default)]
pub struct DestinationInstallBarrier {
    cancelled: Arc<Mutex<bool>>,
}

impl DestinationInstallBarrier {
    pub fn prevent_installation(&self) {
        *self.cancelled.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
    }

    fn install<T>(
        &self,
        install: impl FnOnce() -> T,
    ) -> Option<T> {
        let cancelled = self.cancelled.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if *cancelled {
            return None;
        }
        let result = install();
        drop(cancelled);
        Some(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AppleSinkKey {
    session_identity: usize,
    download_id: DownloadId,
    task_identifier: u64,
}

impl AppleSinkKey {
    pub(crate) fn new(
        session: &NSURLSession,
        download_id: DownloadId,
        task_identifier: u64,
    ) -> Self {
        Self {
            session_identity: std::ptr::from_ref(session).addr(),
            download_id,
            task_identifier,
        }
    }
}

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
        // URLSession only calls this for default and ephemeral sessions;
        // background-session redirects remain controlled by the OS.
        #[unsafe(method(URLSession:task:willPerformHTTPRedirection:newRequest:completionHandler:))]
        unsafe fn will_perform_http_redirection(
            &self,
            _session: &NSURLSession,
            _task: &NSURLSessionTask,
            response: &NSHTTPURLResponse,
            request: &NSURLRequest,
            completion_handler: &block2::DynBlock<dyn Fn(*mut NSURLRequest)>,
        ) {
            let request = if is_https_to_http_redirect(response, request) {
                std::ptr::null_mut()
            } else {
                std::ptr::from_ref(request).cast_mut()
            };
            completion_handler.call((request,));
        }

        #[unsafe(method(URLSession:task:didCompleteWithError:))]
        fn did_complete_with_error(
            &self,
            session: &NSURLSession,
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
            let key = AppleSinkKey::new(session, download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&key))
            else {
                return;
            };
            let error = DownloadError::Transport(error.localizedDescription().to_string());
            sink.runtime_handle.clone().spawn(async move {
                let _ = sink.backend_event_sender.send_terminal(BackendEvent::error(sink.generation, error)).await;
            });
        }
    }

    unsafe impl NSURLSessionDownloadDelegate for AppleSessionDelegate {
        #[unsafe(method(URLSession:downloadTask:didFinishDownloadingToURL:))]
        fn did_finish_downloading_to_url(
            &self,
            session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            location: &NSURL,
        ) {
            let Some(download_id) = download_task.download_id() else {
                return;
            };
            let key = AppleSinkKey::new(session, download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|mut registry| registry.remove(&key))
            else {
                return;
            };
            let status = download_task
                .response()
                .and_then(|response| response.downcast::<NSHTTPURLResponse>().ok())
                .map(|response| response.statusCode());
            if !status.is_some_and(is_success_http_status) {
                let error = status
                    .and_then(|status| u16::try_from(status).ok())
                    .map(DownloadError::from_http_status)
                    .unwrap_or_else(|| {
                        DownloadError::Protocol("download completed without a valid HTTP response".to_string())
                    });
                sink.runtime_handle.clone().spawn(async move {
                    let _ = sink.backend_event_sender.send_terminal(BackendEvent::error(sink.generation, error)).await;
                });
                return;
            }
            let temporary_path = match download_location_path(location.path().map(|path| path.to_string())) {
                Ok(path) => path,
                Err(error) => {
                    sink.runtime_handle.clone().spawn(async move {
                        let _ =
                            sink.backend_event_sender.send_terminal(BackendEvent::error(sink.generation, error)).await;
                    });
                    return;
                },
            };

            // Move synchronously: NSURLSession deletes the temp file as soon as this method returns.
            let Some(move_result) = sink
                .destination_install_barrier
                .install(|| install_downloaded_file(&temporary_path, &sink.installation_artifact, &sink.destination))
            else {
                return;
            };

            let terminal_event = match move_result {
                Ok(()) => BackendEvent::completed(sink.generation),
                Err(error) => BackendEvent::error(sink.generation, DownloadError::Io(error.to_string())),
            };
            sink.runtime_handle.clone().spawn(async move {
                let _ = sink.backend_event_sender.send_terminal(terminal_event).await;
            });
        }

        #[unsafe(method(URLSession:downloadTask:didWriteData:totalBytesWritten:totalBytesExpectedToWrite:))]
        fn did_write_data(
            &self,
            session: &NSURLSession,
            download_task: &NSURLSessionDownloadTask,
            _bytes_written_since_last_callback: i64,
            cumulative_bytes_written: i64,
            total_expected_bytes_to_write: i64,
        ) {
            let Some(download_id) = download_task.download_id() else {
                return;
            };
            let key = AppleSinkKey::new(session, download_id, download_task.task_identifier());
            let Some(sink) =
                Self::ivars(self).event_registry.lock().ok().and_then(|registry| registry.get(&key).cloned())
            else {
                return;
            };
            let downloaded_bytes = cumulative_bytes_written.max(0) as u64;
            let total_bytes = (total_expected_bytes_to_write > 0).then_some(total_expected_bytes_to_write as u64);

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

fn download_location_path(path: Option<String>) -> Result<PathBuf, DownloadError> {
    path.map(PathBuf::from)
        .ok_or_else(|| DownloadError::Protocol("download completed without a temporary file path".to_string()))
}

fn install_downloaded_file(
    temporary_path: &Path,
    installation_artifact: &Path,
    destination: &Path,
) -> Result<(), std::io::Error> {
    if let Some(parent) = destination.parent() {
        reject_symlink_components(parent)?;
        std::fs::create_dir_all(parent)?;
        reject_symlink_components(parent)?;
    }
    if let Some(parent) = installation_artifact.parent() {
        reject_symlink_components(parent)?;
    }
    match std::fs::remove_file(installation_artifact) {
        Ok(()) => {},
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
        Err(error) => return Err(error),
    }

    if std::fs::rename(temporary_path, installation_artifact).is_err() {
        if let Err(error) = std::fs::copy(temporary_path, installation_artifact) {
            let _ = std::fs::remove_file(installation_artifact);
            return Err(error);
        }
        let _ = std::fs::remove_file(temporary_path);
    }

    if let Some(parent) = destination.parent() {
        reject_symlink_components(parent)?;
    }
    if let Some(parent) = installation_artifact.parent() {
        reject_symlink_components(parent)?;
    }
    std::fs::rename(installation_artifact, destination)
}

fn reject_symlink_components(path: &Path) -> Result<(), std::io::Error> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("download installation path contains a symlink: {}", current.display()),
                ));
            },
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn is_platform_path_alias(path: &Path) -> bool {
    matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
}

fn is_success_http_status(status: isize) -> bool {
    (200..=299).contains(&status)
}

fn is_https_to_http_redirect(
    response: &NSHTTPURLResponse,
    request: &NSURLRequest,
) -> bool {
    let previous_scheme = response.URL().and_then(|url| url.scheme()).map(|scheme| scheme.to_string());
    let next_scheme = request.URL().and_then(|url| url.scheme()).map(|scheme| scheme.to_string());
    is_https_to_http(previous_scheme.as_deref(), next_scheme.as_deref())
}

fn is_https_to_http(
    previous_scheme: Option<&str>,
    next_scheme: Option<&str>,
) -> bool {
    matches!((previous_scheme, next_scheme), (Some("https"), Some("http")))
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/apple/delegate_test.rs"]
mod tests;
