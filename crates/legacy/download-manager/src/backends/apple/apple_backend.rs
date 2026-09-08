use std::{
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex, PoisonError},
};

use block2::RcBlock;
use kiban::{fs, rt::RuntimeHandle};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{
    NSArray, NSBundle, NSData, NSString, NSURL, NSURLSession, NSURLSessionConfiguration, NSURLSessionDataTask,
    NSURLSessionDelegate, NSURLSessionDownloadTask, NSURLSessionTaskState, NSURLSessionUploadTask,
};
use tokio::sync::{OnceCell as TokioOnceCell, oneshot::channel as tokio_oneshot_channel};

use crate::{
    DownloadError,
    backends::{
        ActiveTask, Backend, BackendEventSender, DownloadGeneration,
        apple::{
            AppleActiveTask, AppleEventRegistry, AppleEventSink, AppleSessionDelegate, AppleTaskDescription, ResumeData,
        },
    },
    file_download::DownloadConfig,
};

pub struct AppleBackend {
    session: Retained<NSURLSession>,
    _delegate_protocol_object: Retained<ProtocolObject<dyn NSURLSessionDelegate>>,
    event_registry: AppleEventRegistry,
    runtime_handle: RuntimeHandle,
    pending_tasks: TokioOnceCell<Mutex<Vec<Retained<NSURLSessionDownloadTask>>>>,
}

unsafe impl Send for AppleBackend {}
unsafe impl Sync for AppleBackend {}

impl AppleBackend {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        let event_registry = AppleEventRegistry::default();
        let delegate_protocol_object = ProtocolObject::<dyn NSURLSessionDelegate>::from_retained(
            AppleSessionDelegate::new(Arc::clone(&event_registry)),
        );
        let session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &Self::session_configuration(),
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
        }
    }

    pub fn bundle_identifier() -> String {
        NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string()
    }

    fn session_configuration() -> Retained<NSURLSessionConfiguration> {
        let bundle_id = Self::bundle_identifier();
        if bundle_id.is_empty() {
            return NSURLSessionConfiguration::ephemeralSessionConfiguration();
        }
        let session_id = NSString::from_str(&format!("{bundle_id}.trymirai.download-manager"));
        let configuration = NSURLSessionConfiguration::backgroundSessionConfigurationWithIdentifier(&session_id);
        configuration.setSessionSendsLaunchEvents(true);
        configuration.setDiscretionary(false);
        configuration.setWaitsForConnectivity(true);
        configuration
    }

    async fn pending_tasks(&self) -> Result<&Mutex<Vec<Retained<NSURLSessionDownloadTask>>>, DownloadError> {
        self.pending_tasks
            .get_or_try_init(|| async {
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
                let tasks = tasks_receiver.await.map_err(|error| {
                    DownloadError::Backend(format!("URLSession task enumeration callback dropped: {error}"))
                })?;
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
        AppleTaskDescription::from(config).attach_to(&task);
        self.event_registry.lock().unwrap_or_else(PoisonError::into_inner).insert(
            task.taskIdentifier(),
            AppleEventSink {
                generation,
                destination: config.destination.clone(),
                events,
                runtime_handle: self.runtime_handle.clone(),
            },
        );
        task.resume();
        Box::new(AppleActiveTask::new(task, Arc::clone(&self.event_registry)))
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
    ) -> Result<Box<dyn ActiveTask>, DownloadError> {
        let resume_data = fs::asyn::read(&config.resume_artifact_path).await.unwrap_or_default();
        let task = if resume_data.is_empty() {
            let url = NSURL::URLWithString(&NSString::from_str(&config.source_url))
                .ok_or_else(|| DownloadError::Backend(format!("invalid url: {}", config.source_url)))?;
            self.session.downloadTaskWithURL(&url)
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
    ) -> Result<bool, DownloadError> {
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
    ) -> Result<Option<Box<dyn ActiveTask>>, DownloadError> {
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
                && AppleTaskDescription::of(&task).is_some_and(|description| description.matches(&config))
            {
                attached = Some(self.activate(task, &config, generation, events.clone()));
            } else {
                task.cancel();
            }
        }
        Ok(attached)
    }
}
