use std::{
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};

use block2::RcBlock;
use kiban::{fs, rt::RuntimeHandle};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{
    NSArray, NSBundle, NSData, NSMutableURLRequest, NSString, NSURL, NSURLSession, NSURLSessionConfiguration,
    NSURLSessionDataTask, NSURLSessionDelegate, NSURLSessionDownloadTask, NSURLSessionTaskState,
    NSURLSessionUploadTask,
};
use tokio::sync::{OnceCell as TokioOnceCell, oneshot::channel as tokio_oneshot_channel};

use crate::{
    backends::{
        ActiveTask, Backend, BackendError, BackendEventSender, DownloadGeneration,
        apple::{
            AppleActiveTask, AppleBackendError, AppleEventRegistry, AppleEventSink, AppleSessionDelegate,
            AppleTaskDescription, ResumeData,
        },
    },
    file_download::DownloadConfig,
};

#[cfg(target_os = "macos")]
const CALLER_SECURITY_SESSION: u32 = u32::MAX;
#[cfg(target_os = "macos")]
const SESSION_IS_ROOT: u32 = 0x0001;
const TASK_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(target_os = "macos")]
#[link(name = "Security", kind = "framework")]
unsafe extern "C" {
    fn SessionGetInfo(
        session: u32,
        session_id: *mut u32,
        attributes: *mut u32,
    ) -> i32;
}

pub struct AppleBackend {
    session: Retained<NSURLSession>,
    _delegate_protocol_object: Retained<ProtocolObject<dyn NSURLSessionDelegate>>,
    event_registry: AppleEventRegistry,
    runtime_handle: RuntimeHandle,
    pending_tasks: TokioOnceCell<Mutex<Vec<Retained<NSURLSessionDownloadTask>>>>,
    pending_task_error: Mutex<Option<(Instant, String)>>,
}

unsafe impl Send for AppleBackend {}
unsafe impl Sync for AppleBackend {}

impl AppleBackend {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        let event_registry = AppleEventRegistry::default();
        let delegate_protocol_object = ProtocolObject::<dyn NSURLSessionDelegate>::from_retained(
            AppleSessionDelegate::new(Arc::clone(&event_registry)),
        );
        let bundle_id = Self::background_bundle_identifier();
        let configuration = if bundle_id.is_empty() {
            NSURLSessionConfiguration::ephemeralSessionConfiguration()
        } else {
            let session_id = NSString::from_str(&format!("{bundle_id}.trymirai.download-manager"));
            let configuration = NSURLSessionConfiguration::backgroundSessionConfigurationWithIdentifier(&session_id);
            configuration.setSessionSendsLaunchEvents(true);
            configuration.setDiscretionary(false);
            configuration.setWaitsForConnectivity(true);
            configuration
        };
        let session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &configuration,
                Some(&delegate_protocol_object),
                None,
            )
        };
        Self {
            session,
            _delegate_protocol_object: delegate_protocol_object,
            event_registry,
            runtime_handle,
            pending_tasks: TokioOnceCell::new(),
            pending_task_error: Mutex::new(None),
        }
    }

    pub fn background_bundle_identifier() -> String {
        #[cfg(target_os = "macos")]
        {
            let mut attributes = 0;
            let status = unsafe { SessionGetInfo(CALLER_SECURITY_SESSION, std::ptr::null_mut(), &mut attributes) };
            if status != 0 || attributes & SESSION_IS_ROOT != 0 {
                return String::new();
            }
        }
        NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string()
    }

    async fn pending_tasks(&self) -> Result<&Mutex<Vec<Retained<NSURLSessionDownloadTask>>>, BackendError> {
        self.pending_tasks
            .get_or_try_init(|| async {
                // A failed lookup is shared briefly by concurrent initializers.
                // Otherwise each queued file waits through its own timeout.
                if let Some((when, error)) =
                    self.pending_task_error.lock().unwrap_or_else(PoisonError::into_inner).as_ref()
                    && when.elapsed() < TASK_DISCOVERY_TIMEOUT
                {
                    return Err(AppleBackendError::TaskDiscovery(error.clone()).into());
                }
                let (tasks_sender, tasks_receiver) = tokio_oneshot_channel();
                {
                    let tasks_sender = Mutex::new(Some(tasks_sender));
                    let handler = RcBlock::new(
                        move |_data_tasks: NonNull<NSArray<NSURLSessionDataTask>>,
                              _upload_tasks: NonNull<NSArray<NSURLSessionUploadTask>>,
                              download_tasks: NonNull<NSArray<NSURLSessionDownloadTask>>| {
                            if let Some(sender) = tasks_sender.lock().unwrap_or_else(PoisonError::into_inner).take() {
                                let _ = sender.send(unsafe { download_tasks.as_ref() }.to_vec());
                            }
                        },
                    );
                    unsafe {
                        self.session.getTasksWithCompletionHandler(&handler);
                    }
                }
                let result = tokio::time::timeout(TASK_DISCOVERY_TIMEOUT, tasks_receiver)
                    .await
                    .map_err(|_| "URLSession did not reply within 5 seconds".to_string())
                    .and_then(|tasks| tasks.map_err(|error| error.to_string()));
                let tasks = match result {
                    Ok(tasks) => tasks,
                    Err(message) => {
                        *self.pending_task_error.lock().unwrap_or_else(PoisonError::into_inner) =
                            Some((Instant::now(), message.clone()));
                        // Leave native tasks and their destination markers alone.
                        return Err(AppleBackendError::TaskDiscovery(message).into());
                    },
                };
                Ok(Mutex::new(tasks))
            })
            .await
    }

    fn activate(
        &self,
        task: Retained<NSURLSessionDownloadTask>,
        config: &DownloadConfig,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Box<dyn ActiveTask> {
        if let Ok(json) = serde_json::to_string(&AppleTaskDescription::from(config)) {
            task.setTaskDescription(Some(&NSString::from_str(&json)));
        }
        self.event_registry.lock().unwrap_or_else(PoisonError::into_inner).insert(
            task.taskIdentifier(),
            AppleEventSink {
                generation,
                destination: config.destination.clone(),
                expected_bytes: config.expected_bytes,
                events,
                runtime_handle: self.runtime_handle.clone(),
            },
        );
        task.resume();
        Box::new(AppleActiveTask::new(task, Arc::clone(&self.event_registry), config.bearer_token.is_some()))
    }

    fn is_live(task: &NSURLSessionDownloadTask) -> bool {
        matches!(task.state(), NSURLSessionTaskState::Running | NSURLSessionTaskState::Suspended)
    }
}

#[async_trait::async_trait]
impl Backend for AppleBackend {
    fn name(&self) -> &'static str {
        "apple"
    }

    fn resume_artifact_extension(&self) -> &'static str {
        "resume_data"
    }

    async fn start(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Box<dyn ActiveTask>, BackendError> {
        let resume_data = fs::asyn::read(&config.resume_artifact_path).await.unwrap_or_default();
        let task = if resume_data.is_empty() {
            let url = NSURL::URLWithString(&NSString::from_str(&config.source_url))
                .ok_or_else(|| AppleBackendError::InvalidUrl(config.source_url.clone()))?;
            let request = NSMutableURLRequest::requestWithURL(&url);
            request.setValue_forHTTPHeaderField(
                Some(&NSString::from_str("identity")),
                &NSString::from_str("Accept-Encoding"),
            );
            if let Some(token) = &config.bearer_token {
                request.setValue_forHTTPHeaderField(
                    Some(&NSString::from_str(&token.header_value())),
                    &NSString::from_str("Authorization"),
                );
            }
            self.session.downloadTaskWithRequest(&request)
        } else {
            self.session.downloadTaskWithResumeData(&NSData::with_bytes(&resume_data))
        };
        Ok(self.activate(task, &config, generation, events))
    }

    async fn read_resume_progress(
        &self,
        resume_artifact_path: &Path,
    ) -> u64 {
        ResumeData::new(fs::asyn::read(resume_artifact_path).await.unwrap_or_default()).bytes_received().unwrap_or(0)
    }

    async fn has_pending_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<bool, BackendError> {
        let pending_tasks = self.pending_tasks().await?.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(pending_tasks.iter().any(|task| {
            Self::is_live(task)
                && AppleTaskDescription::of(task)
                    .is_some_and(|description| description.download_id == config.download_id)
        }))
    }

    async fn attach_pending_task(
        &self,
        config: Arc<DownloadConfig>,
        generation: DownloadGeneration,
        events: BackendEventSender,
    ) -> Result<Option<Box<dyn ActiveTask>>, BackendError> {
        let candidates = {
            let mut pending_tasks = self.pending_tasks().await?.lock().unwrap_or_else(PoisonError::into_inner);
            let (candidates, rest): (Vec<_>, Vec<_>) =
                std::mem::take(&mut *pending_tasks).into_iter().partition(|task| {
                    AppleTaskDescription::of(task)
                        .is_some_and(|description| description.download_id == config.download_id)
                });
            *pending_tasks = rest;
            candidates
        };
        let mut attached = None;
        for task in candidates {
            if attached.is_none()
                && Self::is_live(&task)
                && AppleTaskDescription::of(&task).is_some_and(|description| {
                    description.download_id == config.download_id && description.source_url == config.source_url
                })
            {
                attached = Some(self.activate(task, &config, generation, events.clone()));
            } else {
                task.cancel();
            }
        }
        Ok(attached)
    }
}
