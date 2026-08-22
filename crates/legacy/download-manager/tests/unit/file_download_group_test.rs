#![allow(deprecated)]

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use tokio::sync::{
    Notify,
    broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel},
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
};

use crate::{
    DownloadError, DownloadEvent, DownloadEventSender, DownloadId, FileCheck, FileDownloadFailure, FileDownloadGroup,
    FileDownloadGroupError, FileDownloadGroupPhase, FileDownloadGroupSpec, FileDownloadManager,
    FileDownloadManagerType, FileDownloadPhase, FileDownloadRequest, FileDownloadSnapshot, FileDownloadState,
    FileDownloadTask, HttpDownloadRequest, RelativeFilePath, SharedDownloadEventSender,
    file_download_group::{
        GROUP_ROOTS, GroupChild, GroupMember, destination_root_is_mount_point, group_artifact_root_for_location,
        reduce_group_state, root_registry_key,
    },
};

fn failure(
    path: &str,
    error: DownloadError,
) -> FileDownloadFailure {
    FileDownloadFailure::new(RelativeFilePath::try_from(path).unwrap(), error)
}

fn snapshots(states: Vec<FileDownloadState>) -> Vec<FileDownloadSnapshot> {
    states.into_iter().map(|state| FileDownloadSnapshot::new(state, None)).collect()
}

#[test]
fn failures_remain_visible_while_another_file_downloads() {
    let members = test_members([("failed.bin", Some(10)), ("active.bin", Some(20))]);
    let states = vec![FileDownloadState::error("failed".to_string()), FileDownloadState::downloading(5, 20)];

    let state = reduce_group_state(&members, &snapshots(states), &[]);

    assert_eq!(state.phase, FileDownloadGroupPhase::Downloading);
    assert_eq!(state.failures.len(), 1);
    assert_eq!(state.total_bytes, Some(30));
}

#[test]
fn ordinary_failure_wins_when_no_file_is_active() {
    let members = test_members([("locked.bin", Some(10)), ("failed.bin", Some(20))]);
    let states =
        vec![FileDownloadState::locked_by_other("other".to_string()), FileDownloadState::error("failed".to_string())];

    let state = reduce_group_state(&members, &snapshots(states), &[]);

    assert_eq!(state.phase, FileDownloadGroupPhase::Error);
    assert_eq!(
        state.failures.iter().map(|failure| failure.relative_path.to_string()).collect::<Vec<_>>(),
        vec!["failed.bin", "locked.bin"]
    );
}

#[test]
fn lock_only_failures_produce_locked_phase() {
    let members = test_members([("locked.bin", Some(10))]);
    let states = vec![FileDownloadState::locked_by_other("other".to_string())];

    let state = reduce_group_state(&members, &snapshots(states), &[]);

    assert_eq!(state.phase, FileDownloadGroupPhase::Locked);
}

#[test]
fn operation_failures_are_sorted_and_counted() {
    let members = test_members([("z.bin", None), ("a.bin", None)]);
    let states = vec![FileDownloadState::not_downloaded(0), FileDownloadState::not_downloaded(0)];
    let failures = vec![
        failure("z.bin", DownloadError::Backend("z".to_string())),
        failure("a.bin", DownloadError::Backend("a".to_string())),
    ];

    let state = reduce_group_state(&members, &snapshots(states), &failures);

    assert_eq!(state.phase, FileDownloadGroupPhase::Error);
    assert_eq!(
        state.failures.iter().map(|failure| failure.relative_path.to_string()).collect::<Vec<_>>(),
        vec!["a.bin", "z.bin"]
    );
    assert_eq!(state.total_bytes, None);
}

#[test]
fn downloaded_progress_does_not_make_an_unknown_group_total_known() {
    let members = test_members([("stream.bin", None)]);
    let member_snapshots =
        vec![FileDownloadSnapshot::with_total_bytes(FileDownloadState::downloading(7, 0), None, None)];

    let state = reduce_group_state(&members, &member_snapshots, &[]);

    assert_eq!(state.downloaded_bytes, 7);
    assert_eq!(state.total_bytes, None);
}

struct MockTask {
    download_id: DownloadId,
    source_url: String,
    destination: PathBuf,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    state_sender: TokioWatchSender<FileDownloadState>,
    snapshot_sender: TokioWatchSender<FileDownloadSnapshot>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    next_download_error: Mutex<Option<DownloadError>>,
    next_cancel_error: Mutex<Option<DownloadError>>,
    download_calls: AtomicUsize,
    pause_calls: AtomicUsize,
    cancel_only_calls: AtomicUsize,
    cancel_calls: AtomicUsize,
}

impl MockTask {
    fn new(
        path: &str,
        state: FileDownloadState,
    ) -> Self {
        let destination = PathBuf::from("/mock").join(path);
        let (state_sender, _) = tokio_watch_channel(state.clone());
        let (snapshot_sender, _) = tokio_watch_channel(FileDownloadSnapshot::new(state, None));
        let (progress_sender, _) = tokio_broadcast_channel(16);
        Self {
            download_id: crate::compute_download_id(&destination),
            source_url: format!("https://example.com/{path}"),
            destination,
            file_check: FileCheck::None,
            expected_bytes: Some(10),
            state_sender,
            snapshot_sender,
            progress_sender,
            next_download_error: Mutex::new(None),
            next_cancel_error: Mutex::new(None),
            download_calls: AtomicUsize::new(0),
            pause_calls: AtomicUsize::new(0),
            cancel_only_calls: AtomicUsize::new(0),
            cancel_calls: AtomicUsize::new(0),
        }
    }

    fn fail_next_download(
        &self,
        error: DownloadError,
    ) {
        *self.next_download_error.lock().unwrap() = Some(error);
    }

    fn fail_next_cancel(
        &self,
        error: DownloadError,
    ) {
        *self.next_cancel_error.lock().unwrap() = Some(error);
    }

    fn publish(
        &self,
        state: FileDownloadState,
    ) {
        self.state_sender.send_replace(state.clone());
        self.snapshot_sender.send_replace(FileDownloadSnapshot::new(state.clone(), None));
        let _ = self.progress_sender.send(state);
    }

    fn publish_failure(
        &self,
        state: FileDownloadState,
        error: DownloadError,
    ) {
        self.state_sender.send_replace(state.clone());
        self.snapshot_sender.send_replace(FileDownloadSnapshot::new(state.clone(), Some(error)));
        let _ = self.progress_sender.send(state);
    }

    fn complete(&self) {
        self.publish(FileDownloadState::downloaded(self.expected_bytes.unwrap_or(0)));
    }
}

impl std::fmt::Debug for MockTask {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("MockTask").field("destination", &self.destination).finish()
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl FileDownloadTask for MockTask {
    fn download_id(&self) -> DownloadId {
        self.download_id
    }

    fn source_url(&self) -> &str {
        &self.source_url
    }

    fn destination(&self) -> &Path {
        &self.destination
    }

    fn file_check(&self) -> &FileCheck {
        &self.file_check
    }

    fn expected_bytes(&self) -> Option<u64> {
        self.expected_bytes
    }

    async fn download(&self) -> Result<(), DownloadError> {
        self.download_calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.state_sender.borrow().phase, FileDownloadPhase::Downloaded | FileDownloadPhase::Downloading) {
            return Ok(());
        }
        if let Some(error) = self.next_download_error.lock().unwrap().take() {
            let state = match &error {
                DownloadError::LockedByOther(owner) => {
                    FileDownloadState::locked_by_other_with_progress(0, self.expected_bytes.unwrap_or(0), owner.clone())
                },
                _ => FileDownloadState::error_with_progress(0, self.expected_bytes.unwrap_or(0), error.to_string()),
            };
            self.publish_failure(state, error.clone());
            return Err(error);
        }
        self.publish(FileDownloadState::downloading(0, self.expected_bytes.unwrap_or(0)));
        Ok(())
    }

    async fn pause(&self) -> Result<(), DownloadError> {
        self.pause_calls.fetch_add(1, Ordering::SeqCst);
        let state = self.state_sender.borrow().clone();
        if !matches!(state.phase, FileDownloadPhase::Downloading) {
            return Err(DownloadError::InvalidStateTransition);
        }
        self.publish(FileDownloadState::paused(state.downloaded_bytes, state.total_bytes));
        Ok(())
    }

    async fn cancel(&self) -> Result<(), DownloadError> {
        self.cancel_only_calls.fetch_add(1, Ordering::SeqCst);
        self.publish(FileDownloadState::not_downloaded(self.expected_bytes.unwrap_or(0)));
        Ok(())
    }

    async fn cancel_and_delete(&self) -> Result<(), DownloadError> {
        self.cancel_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(error) = self.next_cancel_error.lock().unwrap().take() {
            return Err(error);
        }
        self.publish(FileDownloadState::not_downloaded(self.expected_bytes.unwrap_or(0)));
        Ok(())
    }

    async fn state(&self) -> FileDownloadState {
        self.state_sender.borrow().clone()
    }

    fn state_receiver(&self) -> TokioWatchReceiver<FileDownloadState> {
        self.state_sender.subscribe()
    }

    fn snapshot_receiver(&self) -> TokioWatchReceiver<FileDownloadSnapshot> {
        self.snapshot_sender.subscribe()
    }

    fn has_atomic_snapshot_watch(&self) -> bool {
        true
    }

    async fn progress(&self) -> Result<tokio_stream::wrappers::BroadcastStream<FileDownloadState>, DownloadError> {
        Ok(tokio_stream::wrappers::BroadcastStream::new(self.progress_sender.subscribe()))
    }

    async fn start_listening(
        &self,
        _: DownloadEventSender,
    ) {
    }

    async fn stop_listening(&self) {}

    async fn wait(&self) {}

    fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
        self.progress_sender.clone()
    }
}

struct MockManager {
    task: Arc<MockTask>,
    global_sender: SharedDownloadEventSender,
    task_requests: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
    open_existing: bool,
    materialization_gate: Option<MaterializationGate>,
}

#[derive(Clone)]
struct MaterializationGate {
    started: Arc<Notify>,
    continue_download: Arc<Notify>,
}

impl Drop for MockManager {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl FileDownloadManager for MockManager {
    fn manager_id(&self) -> &str {
        "group-lifecycle-test"
    }

    fn subscribe_to_all_downloads(&self) -> tokio_stream::wrappers::BroadcastStream<DownloadEvent> {
        tokio_stream::wrappers::BroadcastStream::new(self.global_sender.subscribe())
    }

    fn global_broadcast_sender(&self) -> SharedDownloadEventSender {
        Arc::clone(&self.global_sender)
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        Ok(vec![self.task.clone()])
    }

    async fn remove_file_task(
        &self,
        _: DownloadId,
    ) -> Result<(), DownloadError> {
        Ok(())
    }

    async fn http_file_download_task(
        &self,
        _: HttpDownloadRequest,
        _: &Path,
        _: FileCheck,
        _: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        self.task_requests.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.materialization_gate {
            gate.started.notify_one();
            gate.continue_download.notified().await;
        }
        Ok(self.task.clone())
    }

    async fn open_existing_http_file_download_task_with_artifact_root(
        &self,
        request: HttpDownloadRequest,
        destination_path: &Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        artifact_root: &Path,
    ) -> Result<Option<Arc<dyn FileDownloadTask>>, DownloadError> {
        if !self.open_existing {
            return Ok(None);
        }
        self.http_file_download_task_with_artifact_root(
            request,
            destination_path,
            file_check,
            expected_bytes,
            artifact_root,
        )
        .await
        .map(Some)
    }
}

fn mock_manager(
    task: Arc<MockTask>,
    task_requests: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
) -> Arc<dyn FileDownloadManager> {
    mock_manager_with_existing(task, task_requests, drop_count, false)
}

fn mock_manager_with_existing(
    task: Arc<MockTask>,
    task_requests: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
    open_existing: bool,
) -> Arc<dyn FileDownloadManager> {
    let (global_sender, _) = tokio_broadcast_channel(16);
    Arc::new(MockManager {
        task,
        global_sender: Arc::new(global_sender),
        task_requests,
        drop_count,
        open_existing,
        materialization_gate: None,
    })
}

fn blocking_mock_manager(
    task: Arc<MockTask>,
    task_requests: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
    gate: MaterializationGate,
) -> Arc<dyn FileDownloadManager> {
    let (global_sender, _) = tokio_broadcast_channel(16);
    Arc::new(MockManager {
        task,
        global_sender: Arc::new(global_sender),
        task_requests,
        drop_count,
        open_existing: false,
        materialization_gate: Some(gate),
    })
}

fn blocking_existing_mock_manager(
    task: Arc<MockTask>,
    task_requests: Arc<AtomicUsize>,
    drop_count: Arc<AtomicUsize>,
    gate: MaterializationGate,
) -> Arc<dyn FileDownloadManager> {
    let (global_sender, _) = tokio_broadcast_channel(16);
    Arc::new(MockManager {
        task,
        global_sender: Arc::new(global_sender),
        task_requests,
        drop_count,
        open_existing: true,
        materialization_gate: Some(gate),
    })
}

fn test_members<const N: usize>(files: [(&str, Option<u64>); N]) -> Vec<GroupMember> {
    files
        .into_iter()
        .map(|(path, expected_bytes)| {
            let task: Arc<dyn FileDownloadTask> = Arc::new(MockTask::new(path, FileDownloadState::not_downloaded(0)));
            GroupMember::new(
                RelativeFilePath::try_from(path).unwrap(),
                expected_bytes,
                task.http_request(),
                task.destination().to_path_buf(),
                task.file_check().clone(),
                PathBuf::from("/mock-artifacts").join(path),
                Some(GroupChild::from_atomic_watch(task)),
            )
        })
        .collect()
}

async fn mock_group(tasks: &[Arc<MockTask>]) -> FileDownloadGroup {
    let requests = tasks.iter().map(|task| {
        let relative_path = task.destination.strip_prefix("/mock").unwrap();
        FileDownloadRequest::new(
            task.source_url.clone(),
            RelativeFilePath::try_from(relative_path).unwrap(),
            FileCheck::None,
            task.expected_bytes,
        )
    });
    let spec = Arc::new(FileDownloadGroupSpec::new("/mock", requests).unwrap());
    let members = tasks
        .iter()
        .map(|task| {
            GroupMember::new(
                RelativeFilePath::try_from(task.destination.strip_prefix("/mock").unwrap()).unwrap(),
                task.expected_bytes,
                task.http_request(),
                task.destination.clone(),
                task.file_check.clone(),
                PathBuf::from("/mock-artifacts").join(task.download_id.to_string()),
                Some(GroupChild::from_atomic_watch(task.clone())),
            )
        })
        .collect::<Vec<_>>()
        .into();
    let manager = mock_manager(
        Arc::clone(tasks.first().expect("mock groups contain at least one task")),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let (actor_count, _) = tokio_watch_channel(0);
    FileDownloadGroup::spawn(Arc::new(super::FileDownloadGroupOwner {
        runtime_handle: kiban::rt::RuntimeHandle::current(),
        manager,
        spec,
        artifact_root: PathBuf::from("/mock-artifacts"),
        members,
        actor_count,
        release_watcher_running: AtomicBool::new(false),
    }))
}

#[test]
fn dropping_a_group_outside_its_runtime_context_does_not_panic() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let group = runtime.block_on(async {
        let task = Arc::new(MockTask::new("outside-runtime.bin", FileDownloadState::not_downloaded(1)));
        mock_group(&[task]).await
    });

    drop(group);
    runtime.block_on(tokio::task::yield_now());
}

#[tokio::test]
async fn download_attempt_waits_for_its_started_files() {
    let first = Arc::new(MockTask::new("first.bin", FileDownloadState::error("old failure".to_string())));
    let second = Arc::new(MockTask::new("second.bin", FileDownloadState::not_downloaded(10)));
    let group = mock_group(&[Arc::clone(&first), Arc::clone(&second)]).await;

    let attempt = group.download().await.unwrap();
    let mut wait = Box::pin(attempt.wait());
    assert!(tokio::time::timeout(Duration::from_millis(20), &mut wait).await.is_err());
    first.complete();
    assert!(tokio::time::timeout(Duration::from_millis(20), &mut wait).await.is_err());
    second.complete();

    let state = tokio::time::timeout(Duration::from_secs(1), wait).await.unwrap().unwrap();
    assert_eq!(state.phase, FileDownloadGroupPhase::Downloaded);
    assert_eq!(first.download_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second.download_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn repeated_download_joins_active_attempt_without_restarting_members() {
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let group = mock_group(&[Arc::clone(&task)]).await;

    let first = group.download().await.unwrap();
    let second = group.download().await.unwrap();

    assert_eq!(task.download_calls.load(Ordering::SeqCst), 1);
    task.complete();
    assert_eq!(first.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
    assert_eq!(second.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
}

#[tokio::test]
async fn completed_attempt_wait_cannot_observe_a_later_retry() {
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    task.fail_next_download(DownloadError::Backend("first attempt failed".to_string()));
    let group = mock_group(&[Arc::clone(&task)]).await;

    let first_attempt = group.download().await.unwrap();
    let second_attempt = group.download().await.unwrap();
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Downloading);

    let first_terminal = first_attempt.wait().await.unwrap();
    assert_eq!(first_terminal.phase, FileDownloadGroupPhase::Error);
    assert_eq!(first_terminal.failures[0].error, DownloadError::Backend("first attempt failed".to_string()));

    task.complete();
    assert_eq!(second_attempt.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
}

#[tokio::test]
async fn failed_start_is_visible_while_sibling_finishes() {
    let failed = Arc::new(MockTask::new("failed.bin", FileDownloadState::not_downloaded(10)));
    failed.fail_next_download(DownloadError::Backend("start failed".to_string()));
    let active = Arc::new(MockTask::new("active.bin", FileDownloadState::not_downloaded(10)));
    let group = mock_group(&[Arc::clone(&failed), Arc::clone(&active)]).await;

    let attempt = group.download().await.unwrap();
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Downloading);
    assert_eq!(group.state().failures.len(), 1);
    active.complete();

    let terminal = attempt.wait().await.unwrap();
    assert_eq!(terminal.phase, FileDownloadGroupPhase::Error);
    assert_eq!(terminal.failures[0].relative_path.to_string(), "failed.bin");
}

#[tokio::test]
async fn failed_start_remains_visible_after_pause_and_reopen() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    task.fail_next_download(DownloadError::Backend("start failed".to_string()));
    let manager = mock_manager_with_existing(
        Arc::clone(&task),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        true,
    );
    let group = FileDownloadGroup::open(Arc::clone(&manager), spec.clone()).await.unwrap();

    let terminal = group.download().await.unwrap().wait().await.unwrap();
    assert_eq!(terminal.phase, FileDownloadGroupPhase::Error);
    group.pause().await.unwrap();
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Error);
    assert_eq!(group.state().failures[0].error, DownloadError::Backend("start failed".to_string()));

    drop(group);
    let reopened = FileDownloadGroup::open(manager, spec).await.unwrap();

    assert_eq!(reopened.state().phase, FileDownloadGroupPhase::Error);
    assert_eq!(reopened.state().failures[0].error, DownloadError::Backend("start failed".to_string()));
}

#[tokio::test]
async fn pause_only_commands_downloading_members() {
    let active = Arc::new(MockTask::new("active.bin", FileDownloadState::downloading(4, 10)));
    let complete = Arc::new(MockTask::new("complete.bin", FileDownloadState::downloaded(10)));
    let group = mock_group(&[Arc::clone(&active), Arc::clone(&complete)]).await;

    group.pause().await.unwrap();

    assert_eq!(active.pause_calls.load(Ordering::SeqCst), 1);
    assert_eq!(complete.pause_calls.load(Ordering::SeqCst), 0);
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Paused);
}

#[tokio::test]
async fn cancel_commands_every_member_and_publishes_not_downloaded() {
    let first = Arc::new(MockTask::new("first.bin", FileDownloadState::downloaded(10)));
    let second = Arc::new(MockTask::new("second.bin", FileDownloadState::paused(4, 10)));
    let group = mock_group(&[Arc::clone(&first), Arc::clone(&second)]).await;

    group.cancel().await.unwrap();

    assert_eq!(first.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
}

#[tokio::test]
async fn cancel_continues_after_one_member_fails() {
    let first = Arc::new(MockTask::new("first.bin", FileDownloadState::paused(4, 10)));
    let second = Arc::new(MockTask::new("second.bin", FileDownloadState::downloaded(10)));
    first.fail_next_cancel(DownloadError::Io("cleanup failed".to_string()));
    let group = mock_group(&[Arc::clone(&first), Arc::clone(&second)]).await;

    let error = group.cancel().await.unwrap_err();

    assert_eq!(first.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(error.failures().len(), 1);
    assert_eq!(error.failures()[0].relative_path.to_string(), "first.bin");
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Error);
}

#[tokio::test]
async fn pristine_missing_child_is_not_created_until_download() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0))),
        spec,
    )
    .await
    .unwrap();

    assert_eq!(task_requests.load(Ordering::SeqCst), 0);
    assert!(group.inner.owner.members[0].child().is_none());

    let attempt = group.download().await.unwrap();
    assert_eq!(task_requests.load(Ordering::SeqCst), 1);
    assert!(group.inner.owner.members[0].child().is_some());
    task.complete();
    assert_eq!(attempt.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
}

#[tokio::test]
async fn open_attaches_child_reported_by_manager_recovery_probe() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::paused(4, 10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager_with_existing(task, Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0)), true),
        spec,
    )
    .await
    .unwrap();

    assert_eq!(task_requests.load(Ordering::SeqCst), 1);
    assert!(group.inner.owner.members[0].child().is_some());
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Paused);
}

#[tokio::test]
async fn download_materializes_every_remaining_child() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("first.bin"), request_for_path("second.bin")])
        .unwrap();
    let task = Arc::new(MockTask::new("shared.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0))),
        spec,
    )
    .await
    .unwrap();
    assert!(group.inner.owner.members.iter().all(|member| member.child().is_none()));

    let attempt = group.download().await.unwrap();

    assert_eq!(task_requests.load(Ordering::SeqCst), 2);
    assert!(group.inner.owner.members.iter().all(|member| member.child().is_some()));
    assert_eq!(task.download_calls.load(Ordering::SeqCst), 2);
    task.complete();
    assert_eq!(attempt.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
}

#[cfg(unix)]
#[tokio::test]
async fn group_open_rejects_existing_symlink_in_destination_path() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    symlink(outside.path(), root.path().join("linked")).unwrap();
    let spec = FileDownloadGroupSpec::new(
        root.path(),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("linked/model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );

    let error = FileDownloadGroup::open(manager, spec).await.unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "linked/model.bin");
}

#[cfg(unix)]
#[tokio::test]
async fn group_open_rejects_symlink_destination_root() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    symlink(outside.path(), parent.path().join("linked-root")).unwrap();
    let spec = FileDownloadGroupSpec::new(
        parent.path().join("linked-root"),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );

    let error = FileDownloadGroup::open(manager, spec).await.unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "model.bin");
}

#[cfg(unix)]
#[tokio::test]
async fn group_open_rejects_missing_root_below_symlink() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let linked_parent = parent.path().join("linked-parent");
    symlink(outside.path(), &linked_parent).unwrap();
    let spec = FileDownloadGroupSpec::new(
        linked_parent.join("missing-root"),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );

    let error = FileDownloadGroup::open(manager, spec).await.unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "model.bin");
    assert!(!outside.path().join("missing-root").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn group_open_rejects_existing_root_below_symlink() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let existing_root = outside.path().join("existing-root");
    std::fs::create_dir(&existing_root).unwrap();
    let linked_parent = parent.path().join("linked-parent");
    symlink(outside.path(), &linked_parent).unwrap();
    let spec = FileDownloadGroupSpec::new(
        linked_parent.join("existing-root"),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );

    let error = FileDownloadGroup::open(manager, spec).await.unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "model.bin");
}

#[cfg(unix)]
#[tokio::test]
async fn download_rejects_destination_symlink_inserted_after_open() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("nested/model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0))),
        spec,
    )
    .await
    .unwrap();
    symlink(outside.path(), root.path().join("nested")).unwrap();

    let terminal = group.download().await.unwrap().wait().await.unwrap();

    assert_eq!(terminal.phase, FileDownloadGroupPhase::Error);
    assert_eq!(task_requests.load(Ordering::SeqCst), 0);
    assert!(!outside.path().join("model.bin").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_quiesces_active_child_before_rejecting_late_destination_symlink() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let outside_file = outside.path().join("model.bin");
    tokio::fs::write(&outside_file, b"keep").await.unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("nested/model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let group = FileDownloadGroup::open(
        mock_manager_with_existing(
            Arc::clone(&task),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            true,
        ),
        spec,
    )
    .await
    .unwrap();
    let _attempt = group.download().await.unwrap();
    symlink(outside.path(), root.path().join("nested")).unwrap();

    let error = group.cancel().await.unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "nested/model.bin");
    assert_eq!(task.cancel_only_calls.load(Ordering::SeqCst), 0);
    assert_eq!(task.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(tokio::fs::read(&outside_file).await.unwrap(), b"keep");
}

#[cfg(unix)]
#[tokio::test]
async fn open_rejects_symlinked_group_artifact_root() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let destination_root = tokio::fs::canonicalize(root.path()).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [request_for_path("model.bin")]).unwrap();
    let artifact_root = group_artifact_root_for_location(&destination_root, spec.files(), false);
    tokio::fs::create_dir_all(artifact_root.parent().unwrap()).await.unwrap();
    symlink(outside.path(), &artifact_root).unwrap();
    tokio::fs::write(outside.path().join("download.part"), b"keep part").await.unwrap();
    tokio::fs::write(outside.path().join("integrity.json"), b"keep receipt").await.unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));

    let error =
        FileDownloadGroup::open(mock_manager(task, Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0))), spec)
            .await
            .unwrap_err();

    assert_eq!(error.failures()[0].relative_path.to_string(), "model.bin");
    assert_eq!(task_requests.load(Ordering::SeqCst), 0);
    assert_eq!(tokio::fs::read(outside.path().join("download.part")).await.unwrap(), b"keep part");
    assert_eq!(tokio::fs::read(outside.path().join("integrity.json")).await.unwrap(), b"keep receipt");
}

#[cfg(unix)]
#[tokio::test]
async fn download_and_cancel_reject_artifact_symlink_inserted_after_open() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::clone(&task_requests), Arc::new(AtomicUsize::new(0))),
        spec,
    )
    .await
    .unwrap();
    let artifact_root = group.inner.owner.artifact_root.clone();
    tokio::fs::create_dir_all(artifact_root.parent().unwrap()).await.unwrap();
    symlink(outside.path(), &artifact_root).unwrap();
    let part = outside.path().join("download.part");
    let receipt = outside.path().join("integrity.json");
    tokio::fs::write(&part, b"keep part").await.unwrap();
    tokio::fs::write(&receipt, b"keep receipt").await.unwrap();

    let terminal = group.download().await.unwrap().wait().await.unwrap();
    let cancel_error = group.cancel().await.unwrap_err();

    assert_eq!(terminal.phase, FileDownloadGroupPhase::Error);
    assert_eq!(cancel_error.failures()[0].relative_path.to_string(), "model.bin");
    assert_eq!(task_requests.load(Ordering::SeqCst), 1);
    assert_eq!(task.cancel_calls.load(Ordering::SeqCst), 1);
    assert_eq!(tokio::fs::read(part).await.unwrap(), b"keep part");
    assert_eq!(tokio::fs::read(receipt).await.unwrap(), b"keep receipt");
}

#[tokio::test]
async fn open_reuses_matching_live_group_and_rejects_conflicting_spec() {
    let root = tempfile::tempdir().unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let request = |url: &str| {
        FileDownloadRequest::new(url, RelativeFilePath::try_from("model.bin").unwrap(), FileCheck::None, None)
    };
    let spec = FileDownloadGroupSpec::new(root.path(), [request("https://example.com/first")]).unwrap();

    let first = FileDownloadGroup::open(Arc::clone(&manager), spec.clone()).await.unwrap();
    let reused = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    assert!(Arc::ptr_eq(&first.inner, &reused.inner));

    let conflict_spec = FileDownloadGroupSpec::new(root.path(), [request("https://example.com/second")]).unwrap();
    let conflict = FileDownloadGroup::open(manager, conflict_spec).await.unwrap_err();
    assert!(matches!(conflict, FileDownloadGroupError::RootConflict { .. }));
}

#[tokio::test]
async fn open_reuses_matching_group_across_manager_instances() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(
        root.path(),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let manager = || async {
        let manager: Arc<dyn FileDownloadManager> = Arc::from(
            <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
                .await
                .unwrap(),
        );
        manager
    };

    let first = FileDownloadGroup::open(manager().await, spec.clone()).await.unwrap();
    let second = FileDownloadGroup::open(manager().await, spec).await.unwrap();

    assert!(Arc::ptr_eq(&first.inner, &second.inner));
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
#[tokio::test]
async fn roots_that_only_differ_by_case_share_one_ownership_key() {
    let parent = tempfile::tempdir().unwrap();
    let upper_root = parent.path().join("Foo");
    let lower_root = parent.path().join("foo");
    let first_spec = FileDownloadGroupSpec::new(
        &upper_root,
        [FileDownloadRequest::new(
            "https://example.com/first",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let second_spec = FileDownloadGroupSpec::new(
        &lower_root,
        [FileDownloadRequest::new(
            "https://example.com/second",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            None,
        )],
    )
    .unwrap();
    let first_task = Arc::new(MockTask::new("first.bin", FileDownloadState::not_downloaded(10)));
    let first = FileDownloadGroup::open(
        mock_manager(first_task, Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))),
        first_spec,
    )
    .await
    .unwrap();
    let second_task = Arc::new(MockTask::new("second.bin", FileDownloadState::not_downloaded(10)));

    let conflict = FileDownloadGroup::open(
        mock_manager(second_task, Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))),
        second_spec,
    )
    .await
    .unwrap_err();

    assert!(matches!(conflict, FileDownloadGroupError::RootConflict { .. }));
    drop(first);
}

#[tokio::test]
async fn dropping_and_reopening_an_active_group_reuses_its_live_children() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let first_task_requests = Arc::new(AtomicUsize::new(0));
    let first_manager_drops = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::clone(&first_task_requests), Arc::clone(&first_manager_drops)),
        spec.clone(),
    )
    .await
    .unwrap();
    let _abandoned_attempt = group.download().await.unwrap();

    drop(group);
    tokio::task::yield_now().await;

    assert_eq!(task.pause_calls.load(Ordering::SeqCst), 0);
    assert_eq!(first_manager_drops.load(Ordering::SeqCst), 0);

    let replacement_task = Arc::new(MockTask::new("replacement.bin", FileDownloadState::not_downloaded(10)));
    let replacement_task_requests = Arc::new(AtomicUsize::new(0));
    let replacement_manager_drops = Arc::new(AtomicUsize::new(0));
    let reopened = FileDownloadGroup::open(
        mock_manager(replacement_task, Arc::clone(&replacement_task_requests), Arc::clone(&replacement_manager_drops)),
        spec,
    )
    .await
    .unwrap();

    assert_eq!(reopened.state().phase, FileDownloadGroupPhase::Downloading);
    assert_eq!(replacement_task_requests.load(Ordering::SeqCst), 0);
    assert_eq!(replacement_manager_drops.load(Ordering::SeqCst), 1);
    let resumed_attempt = reopened.download().await.unwrap();
    assert_eq!(task.download_calls.load(Ordering::SeqCst), 1);

    task.complete();
    assert_eq!(resumed_attempt.wait().await.unwrap().phase, FileDownloadGroupPhase::Downloaded);
    drop(reopened);

    tokio::time::timeout(Duration::from_secs(1), async {
        while first_manager_drops.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(first_manager_drops.load(Ordering::SeqCst), 1);
    assert_eq!(first_task_requests.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn blocked_root_reconciliation_does_not_block_an_unrelated_root() {
    let parent = tempfile::tempdir().unwrap();
    let blocked_root = parent.path().join("blocked");
    let unrelated_root = parent.path().join("unrelated");
    let blocked_spec = FileDownloadGroupSpec::new(&blocked_root, [request_for_path("model.bin")]).unwrap();
    let unrelated_spec = FileDownloadGroupSpec::new(&unrelated_root, [request_for_path("model.bin")]).unwrap();
    let gate = MaterializationGate {
        started: Arc::new(Notify::new()),
        continue_download: Arc::new(Notify::new()),
    };
    let blocked_open = tokio::spawn(FileDownloadGroup::open(
        blocking_existing_mock_manager(
            Arc::new(MockTask::new("blocked.bin", FileDownloadState::not_downloaded(10))),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            gate.clone(),
        ),
        blocked_spec,
    ));
    gate.started.notified().await;
    assert!(!blocked_open.is_finished());

    let unrelated_open = tokio::time::timeout(
        Duration::from_secs(1),
        FileDownloadGroup::open(
            mock_manager(
                Arc::new(MockTask::new("unrelated.bin", FileDownloadState::not_downloaded(10))),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            ),
            unrelated_spec,
        ),
    )
    .await;
    gate.continue_download.notify_one();

    let blocked_group = tokio::time::timeout(Duration::from_secs(1), blocked_open).await.unwrap().unwrap().unwrap();
    let unrelated_group = unrelated_open.expect("an unrelated root must not wait for blocked reconciliation").unwrap();
    drop(blocked_group);
    drop(unrelated_group);
}

#[tokio::test]
async fn dropping_group_while_download_command_materializes_keeps_owner_until_transfer_settles() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let task_requests = Arc::new(AtomicUsize::new(0));
    let manager_drops = Arc::new(AtomicUsize::new(0));
    let gate = MaterializationGate {
        started: Arc::new(Notify::new()),
        continue_download: Arc::new(Notify::new()),
    };
    let group = FileDownloadGroup::open(
        blocking_mock_manager(Arc::clone(&task), Arc::clone(&task_requests), Arc::clone(&manager_drops), gate.clone()),
        spec.clone(),
    )
    .await
    .unwrap();
    let downloading_group = group.clone();
    let download_call = tokio::spawn(async move { downloading_group.download().await });
    gate.started.notified().await;

    download_call.abort();
    let _ = download_call.await;
    drop(group);
    assert_eq!(manager_drops.load(Ordering::SeqCst), 0);

    let replacement_task = Arc::new(MockTask::new("replacement.bin", FileDownloadState::not_downloaded(10)));
    let replacement_task_requests = Arc::new(AtomicUsize::new(0));
    let replacement_manager_drops = Arc::new(AtomicUsize::new(0));
    let reopening = tokio::spawn(FileDownloadGroup::open(
        mock_manager(replacement_task, Arc::clone(&replacement_task_requests), Arc::clone(&replacement_manager_drops)),
        spec,
    ));
    tokio::task::yield_now().await;
    assert!(!reopening.is_finished(), "reopen must wait for the in-flight group actor");

    gate.continue_download.notify_one();
    tokio::time::timeout(Duration::from_secs(1), async {
        while !matches!(task.state_sender.borrow().phase, FileDownloadPhase::Downloading) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let reopened = tokio::time::timeout(Duration::from_secs(1), reopening).await.unwrap().unwrap().unwrap();
    assert_eq!(*reopened.inner.owner.actor_count.borrow(), 1);
    assert_eq!(manager_drops.load(Ordering::SeqCst), 0);
    assert_eq!(task_requests.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_task_requests.load(Ordering::SeqCst), 0);
    assert_eq!(replacement_manager_drops.load(Ordering::SeqCst), 1);

    task.complete();
    drop(reopened);
    tokio::time::timeout(Duration::from_secs(1), async {
        while manager_drops.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(manager_drops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn repeated_reopen_and_drop_uses_one_release_watcher() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let manager_drops = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::new(AtomicUsize::new(0)), Arc::clone(&manager_drops)),
        spec.clone(),
    )
    .await
    .unwrap();
    let _abandoned_attempt = group.download().await.unwrap();
    let owner = Arc::clone(&group.inner.owner);
    drop(group);

    tokio::time::timeout(Duration::from_secs(1), async {
        while *owner.actor_count.borrow() > 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(owner.release_watcher_running.load(Ordering::Acquire));
    let retained_owner_count = Arc::strong_count(&owner);

    for _ in 0..3 {
        let reopened = FileDownloadGroup::open(
            mock_manager(
                Arc::new(MockTask::new("replacement.bin", FileDownloadState::not_downloaded(10))),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            ),
            spec.clone(),
        )
        .await
        .unwrap();
        assert!(Arc::ptr_eq(&reopened.inner.owner, &owner));
        drop(reopened);

        tokio::time::timeout(Duration::from_secs(1), async {
            while *owner.actor_count.borrow() > 0 || Arc::strong_count(&owner) > retained_owner_count {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reopening must not retain another release watcher");
        assert_eq!(Arc::strong_count(&owner), retained_owner_count);
    }

    task.complete();
    let registry_key = root_registry_key(owner.spec.destination_root());
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let root_is_owned =
                GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).contains_key(&registry_key);
            if !root_is_owned {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(owner);
    tokio::time::timeout(Duration::from_secs(1), async {
        while manager_drops.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn active_dropped_group_keeps_overlapping_roots_reserved_until_settled() {
    let parent = tempfile::tempdir().unwrap();
    let destination_root = parent.path().join("model");
    tokio::fs::create_dir(&destination_root).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [request_for_path("model.bin")]).unwrap();
    let task = Arc::new(MockTask::new("model.bin", FileDownloadState::not_downloaded(10)));
    let manager_drops = Arc::new(AtomicUsize::new(0));
    let group = FileDownloadGroup::open(
        mock_manager(Arc::clone(&task), Arc::new(AtomicUsize::new(0)), Arc::clone(&manager_drops)),
        spec,
    )
    .await
    .unwrap();
    let _abandoned_attempt = group.download().await.unwrap();
    drop(group);

    let conflicting_spec =
        FileDownloadGroupSpec::new(destination_root.join("nested"), [request_for_path("other.bin")]).unwrap();
    let conflicting_task = Arc::new(MockTask::new("other.bin", FileDownloadState::not_downloaded(10)));
    let conflicting_task_requests = Arc::new(AtomicUsize::new(0));
    let conflict = FileDownloadGroup::open(
        mock_manager(conflicting_task, Arc::clone(&conflicting_task_requests), Arc::new(AtomicUsize::new(0))),
        conflicting_spec,
    )
    .await
    .unwrap_err();

    assert!(matches!(conflict, FileDownloadGroupError::RootConflict { .. }));
    assert_eq!(conflicting_task_requests.load(Ordering::SeqCst), 0);
    assert_eq!(manager_drops.load(Ordering::SeqCst), 0);

    task.complete();
    tokio::time::timeout(Duration::from_secs(1), async {
        while manager_drops.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(manager_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn artifact_state_is_a_sibling_of_a_regular_destination_root() {
    let destination_root = Path::new("/cache/models/model");

    assert_eq!(
        group_artifact_root_for_location(destination_root, &[], false),
        Path::new("/cache/models")
            .join(".uzu-download-manager")
            .join(crate::compute_download_id(destination_root).to_string())
    );
}

#[test]
fn artifact_state_moves_inside_a_mount_point_root() {
    let destination_root = Path::new("/volumes/models");

    assert_eq!(
        group_artifact_root_for_location(destination_root, &[], true),
        destination_root.join(format!(".uzu-download-manager-{}", crate::compute_download_id(destination_root)))
    );
}

#[test]
fn mount_point_artifact_state_does_not_overlap_declared_files() {
    let destination_root = Path::new("/volumes/models");
    let root_id = crate::compute_download_id(destination_root);
    let colliding_path = format!(".UZU-DOWNLOAD-MANAGER-{root_id}/metadata.json");
    let files = [request_for_path(&colliding_path)];

    assert_eq!(
        group_artifact_root_for_location(destination_root, &files, true),
        destination_root.join(format!(".uzu-download-manager-{root_id}-1"))
    );
}

#[cfg(unix)]
#[tokio::test]
async fn detects_filesystem_root_but_not_regular_directory_as_mount_point() {
    let regular_directory = tempfile::tempdir().unwrap();

    assert!(destination_root_is_mount_point(Path::new("/")).await);
    assert!(!destination_root_is_mount_point(regular_directory.path()).await);
}

#[tokio::test]
async fn destructive_cancel_deletes_declared_file_but_keeps_group_root() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("model.bin");
    tokio::fs::write(&destination, b"model").await.unwrap();
    let spec = FileDownloadGroupSpec::new(
        root.path(),
        [FileDownloadRequest::new(
            "https://example.com/model.bin",
            RelativeFilePath::try_from("model.bin").unwrap(),
            FileCheck::None,
            Some(5),
        )],
    )
    .unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let group = FileDownloadGroup::open(manager, spec).await.unwrap();
    let artifact_root = group.inner.owner.artifact_root.clone();
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Downloaded);

    group.cancel().await.unwrap();

    assert!(!destination.exists());
    assert!(root.path().exists());
    assert!(!artifact_root.exists());
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
}

#[tokio::test]
async fn cancel_preserves_unrelated_group_artifact_contents_without_failing() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let group = FileDownloadGroup::open(manager, spec).await.unwrap();
    tokio::fs::create_dir_all(&group.inner.owner.artifact_root).await.unwrap();
    let unrelated = group.inner.owner.artifact_root.join("unrelated.txt");
    tokio::fs::write(&unrelated, b"keep").await.unwrap();

    group.cancel().await.unwrap();

    assert!(unrelated.exists());
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
}

#[tokio::test]
async fn open_materializes_member_with_existing_manager_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let destination_root = tokio::fs::canonicalize(root.path()).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [request_for_path("model.bin")]).unwrap();
    let destination = destination_root.join("model.bin");
    let artifact_root = group_artifact_root_for_location(&destination_root, spec.files(), false)
        .join(crate::compute_download_id(&destination).to_string());
    tokio::fs::create_dir_all(&artifact_root).await.unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );

    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();

    assert!(group.inner.owner.members[0].child().is_some());
    assert_eq!(manager.get_all_file_tasks().await.unwrap().len(), 1);
}

#[tokio::test]
async fn open_reconciles_cached_task_when_valid_destination_appears() {
    let root = tempfile::tempdir().unwrap();
    let destination_root = tokio::fs::canonicalize(root.path()).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [sized_request_for_path("model.bin", 5)]).unwrap();
    let request = spec.files()[0].clone();
    let destination = destination_root.join("model.bin");
    let artifact_root = group_artifact_root_for_location(&destination_root, spec.files(), false)
        .join(crate::compute_download_id(&destination).to_string());
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let cached = manager
        .http_file_download_task_with_artifact_root(
            request.source,
            &destination,
            request.check,
            request.expected_bytes,
            &artifact_root,
        )
        .await
        .unwrap();
    assert_eq!(cached.state().await.phase, FileDownloadPhase::NotDownloaded);
    tokio::fs::write(&destination, b"model").await.unwrap();

    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    let attached = group.inner.owner.members[0].child().unwrap();

    assert!(!Arc::ptr_eq(&cached, &attached.task));
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Downloaded);
}

#[tokio::test]
async fn open_reconciles_cached_downloaded_task_after_destination_is_removed() {
    let root = tempfile::tempdir().unwrap();
    let destination_root = tokio::fs::canonicalize(root.path()).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [sized_request_for_path("model.bin", 5)]).unwrap();
    let request = spec.files()[0].clone();
    let destination = destination_root.join("model.bin");
    let artifact_root = group_artifact_root_for_location(&destination_root, spec.files(), false)
        .join(crate::compute_download_id(&destination).to_string());
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    tokio::fs::write(&destination, b"model").await.unwrap();
    let cached = manager
        .http_file_download_task_with_artifact_root(
            request.source,
            &destination,
            request.check,
            request.expected_bytes,
            &artifact_root,
        )
        .await
        .unwrap();
    assert_eq!(cached.state().await.phase, FileDownloadPhase::Downloaded);
    tokio::fs::remove_file(&destination).await.unwrap();

    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    let attached = group.inner.owner.members[0].child().unwrap();

    assert!(!Arc::ptr_eq(&cached, &attached.task));
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
}

#[tokio::test]
async fn open_reconciles_cached_downloaded_task_after_destination_is_truncated() {
    let root = tempfile::tempdir().unwrap();
    let destination_root = tokio::fs::canonicalize(root.path()).await.unwrap();
    let spec = FileDownloadGroupSpec::new(&destination_root, [sized_request_for_path("model.bin", 5)]).unwrap();
    let request = spec.files()[0].clone();
    let destination = destination_root.join("model.bin");
    let artifact_root = group_artifact_root_for_location(&destination_root, spec.files(), false)
        .join(crate::compute_download_id(&destination).to_string());
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    tokio::fs::write(&destination, b"model").await.unwrap();
    let cached = manager
        .http_file_download_task_with_artifact_root(
            request.source,
            &destination,
            request.check,
            request.expected_bytes,
            &artifact_root,
        )
        .await
        .unwrap();
    assert_eq!(cached.state().await.phase, FileDownloadPhase::Downloaded);
    tokio::fs::write(&destination, b"bad").await.unwrap();

    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    let attached = group.inner.owner.members[0].child().unwrap();

    assert!(!Arc::ptr_eq(&cached, &attached.task));
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
    assert!(!destination.exists());
}

#[tokio::test]
async fn dropping_settled_group_evicts_task_without_deleting_destination() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("model.bin");
    tokio::fs::write(&destination, b"model").await.unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [sized_request_for_path("model.bin", 5)]).unwrap();
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    assert_eq!(group.state().phase, FileDownloadGroupPhase::Downloaded);
    assert_eq!(manager.get_all_file_tasks().await.unwrap().len(), 1);

    drop(group);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if manager.get_all_file_tasks().await.unwrap().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    assert_eq!(tokio::fs::read(&destination).await.unwrap(), b"model");
}

#[tokio::test]
async fn cancel_materializes_lazy_member_that_gained_a_destination_after_open() {
    let root = tempfile::tempdir().unwrap();
    let spec = FileDownloadGroupSpec::new(root.path(), [request_for_path("model.bin")]).unwrap();
    let destination = root.path().join("model.bin");
    let manager: Arc<dyn FileDownloadManager> = Arc::from(
        <dyn FileDownloadManager>::new(FileDownloadManagerType::Universal, kiban::rt::RuntimeHandle::current())
            .await
            .unwrap(),
    );
    let group = FileDownloadGroup::open(Arc::clone(&manager), spec).await.unwrap();
    assert!(group.inner.owner.members[0].child().is_none());
    tokio::fs::write(&destination, b"late model").await.unwrap();

    group.cancel().await.unwrap();

    assert!(!destination.exists());
    assert!(group.inner.owner.members[0].child().is_some());
    assert_eq!(group.state().phase, FileDownloadGroupPhase::NotDownloaded);
}

fn request_for_path(path: &str) -> FileDownloadRequest {
    FileDownloadRequest::new(
        format!("https://example.com/{path}"),
        RelativeFilePath::try_from(path).unwrap(),
        FileCheck::None,
        None,
    )
}

fn sized_request_for_path(
    path: &str,
    expected_bytes: u64,
) -> FileDownloadRequest {
    FileDownloadRequest::new(
        format!("https://example.com/{path}"),
        RelativeFilePath::try_from(path).unwrap(),
        FileCheck::None,
        Some(expected_bytes),
    )
}
