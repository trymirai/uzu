use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    path::{Component, Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, LazyLock, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use futures_util::{Stream, StreamExt, future::join_all, stream::SelectAll};
use kiban::rt;
use tokio::sync::{
    mpsc::{Receiver as TokioMpscReceiver, Sender as TokioMpscSender, channel as tokio_mpsc_channel},
    oneshot::{Receiver as TokioOneshotReceiver, Sender as TokioOneshotSender, channel as tokio_oneshot_channel},
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
};
use tokio_stream::wrappers::WatchStream;

use crate::{
    DownloadError, FileDownloadFailure, FileDownloadGroupError, FileDownloadGroupOperation, FileDownloadGroupPhase,
    FileDownloadGroupSpec, FileDownloadGroupState, FileDownloadManager, FileDownloadPhase, FileDownloadRequest,
    FileDownloadSnapshot, FileDownloadState, FileDownloadTask, RelativeFilePath, compute_download_id,
    file_download_task::seeded_compatibility_snapshot_receiver,
};

#[derive(Clone)]
pub struct FileDownloadGroup {
    inner: Arc<FileDownloadGroupHandle>,
}

struct FileDownloadGroupHandle {
    owner: Arc<FileDownloadGroupOwner>,
    command_sender: TokioMpscSender<GroupCommand>,
    state_receiver: TokioWatchReceiver<FileDownloadGroupState>,
}

struct FileDownloadGroupOwner {
    runtime_handle: rt::RuntimeHandle,
    manager: Arc<dyn FileDownloadManager>,
    spec: Arc<FileDownloadGroupSpec>,
    artifact_root: PathBuf,
    members: Arc<[GroupMember]>,
    actor_count: TokioWatchSender<usize>,
    release_watcher_running: AtomicBool,
}

enum GroupRootEntry {
    Reserved(Arc<GroupRootReservationState>),
    Live(GroupRootOwner),
}

struct GroupRootOwner {
    owner: Arc<FileDownloadGroupOwner>,
    handle: Weak<FileDownloadGroupHandle>,
}

struct GroupRootReservationState {
    completed: TokioWatchSender<bool>,
}

struct GroupRootReservation {
    registry_key: PathBuf,
    state: Arc<GroupRootReservationState>,
    active: bool,
}

enum GroupRootClaim {
    Existing(Arc<FileDownloadGroupHandle>),
    Reserved(GroupRootReservation),
    Wait(GroupRootWait),
}

enum GroupRootWait {
    Construction(Arc<GroupRootReservationState>),
    Actor(Arc<FileDownloadGroupOwner>),
}

static GROUP_ROOTS: LazyLock<Mutex<HashMap<PathBuf, GroupRootEntry>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct DownloadAttempt {
    completion_receiver: TokioOneshotReceiver<FileDownloadGroupState>,
}

#[derive(Clone)]
struct GroupMember {
    relative_path: RelativeFilePath,
    expected_bytes: Option<u64>,
    source: crate::HttpDownloadRequest,
    destination: PathBuf,
    file_check: crate::FileCheck,
    artifact_root: PathBuf,
    child: Arc<OnceLock<GroupChild>>,
    retained_failure: Arc<Mutex<Option<DownloadError>>>,
}

#[derive(Clone)]
struct GroupChild {
    task: Arc<dyn FileDownloadTask>,
    snapshot_receiver: TokioWatchReceiver<FileDownloadSnapshot>,
}

type GroupMemberSnapshotStream = Pin<Box<dyn Stream<Item = (usize, FileDownloadSnapshot)> + Send>>;

enum GroupCommand {
    Download {
        reply_sender: TokioOneshotSender<DownloadAttempt>,
    },
    Pause {
        reply_sender: TokioOneshotSender<Result<(), FileDownloadGroupError>>,
    },
    Cancel {
        reply_sender: TokioOneshotSender<Result<(), FileDownloadGroupError>>,
    },
}

struct FileDownloadGroupActor {
    owner: Arc<FileDownloadGroupOwner>,
    _lease: GroupActorLease,
    members: Arc<[GroupMember]>,
    member_snapshots: Vec<FileDownloadSnapshot>,
    watched_members: Vec<bool>,
    operation_failures: Vec<FileDownloadFailure>,
    command_receiver: TokioMpscReceiver<GroupCommand>,
    state_sender: TokioWatchSender<FileDownloadGroupState>,
    attempt_waiters: Vec<TokioOneshotSender<FileDownloadGroupState>>,
}

struct GroupActorLease(Arc<FileDownloadGroupOwner>);

impl FileDownloadGroupOwner {
    fn has_downloading_member(&self) -> bool {
        self.members
            .iter()
            .filter_map(GroupMember::child)
            .any(|child| matches!(child.snapshot_receiver.borrow().state.phase, FileDownloadPhase::Downloading))
    }

    fn is_live(&self) -> bool {
        *self.actor_count.borrow() > 0 || self.has_downloading_member()
    }
}

impl GroupActorLease {
    fn new(owner: Arc<FileDownloadGroupOwner>) -> Self {
        owner.actor_count.send_modify(|count| *count += 1);
        Self(owner)
    }
}

impl Drop for GroupActorLease {
    fn drop(&mut self) {
        self.0.actor_count.send_modify(|count| *count = count.saturating_sub(1));
    }
}

impl GroupMember {
    fn new(
        relative_path: RelativeFilePath,
        expected_bytes: Option<u64>,
        source: crate::HttpDownloadRequest,
        destination: PathBuf,
        file_check: crate::FileCheck,
        artifact_root: PathBuf,
        child_task: Option<GroupChild>,
    ) -> Self {
        let child = Arc::new(OnceLock::new());
        if let Some(child_task) = child_task {
            let _ = child.set(child_task);
        }
        Self {
            relative_path,
            expected_bytes,
            source,
            destination,
            file_check,
            artifact_root,
            child,
            retained_failure: Arc::new(Mutex::new(None)),
        }
    }

    fn child(&self) -> Option<GroupChild> {
        self.child.get().cloned()
    }

    fn set_child_if_missing(
        &self,
        child: GroupChild,
    ) -> GroupChild {
        self.child.get_or_init(|| child).clone()
    }

    fn snapshot(&self) -> FileDownloadSnapshot {
        self.child().map_or_else(
            || FileDownloadSnapshot::new(FileDownloadState::not_downloaded(self.expected_bytes.unwrap_or(0)), None),
            |child| child.snapshot_receiver.borrow().clone(),
        )
    }

    fn retained_failure(&self) -> Option<DownloadError> {
        self.retained_failure.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    fn set_retained_failure(
        &self,
        failure: Option<DownloadError>,
    ) {
        *self.retained_failure.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = failure;
    }
}

impl GroupChild {
    async fn new(task: Arc<dyn FileDownloadTask>) -> Self {
        if task.has_atomic_snapshot_watch() {
            return Self::from_atomic_watch(task);
        }
        let snapshot_receiver = seeded_compatibility_snapshot_receiver(Arc::clone(&task)).await;
        Self {
            task,
            snapshot_receiver,
        }
    }

    fn from_atomic_watch(task: Arc<dyn FileDownloadTask>) -> Self {
        let snapshot_receiver = task.snapshot_receiver();
        Self {
            task,
            snapshot_receiver,
        }
    }
}

impl Drop for FileDownloadGroupHandle {
    fn drop(&mut self) {
        schedule_group_root_release(Arc::clone(&self.owner));
    }
}

impl FileDownloadGroup {
    pub async fn open(
        manager: Arc<dyn FileDownloadManager>,
        spec: FileDownloadGroupSpec,
    ) -> Result<Self, FileDownloadGroupError> {
        let normalized_root = validate_existing_symlinks(&spec).await?;
        let spec = Arc::new(spec.with_destination_root(normalized_root.clone()));
        let registry_key = root_registry_key(&normalized_root);
        let reservation = loop {
            match claim_group_root(&registry_key, &normalized_root, &spec)? {
                GroupRootClaim::Existing(inner) => {
                    return Ok(Self {
                        inner,
                    });
                },
                GroupRootClaim::Reserved(reservation) => break reservation,
                GroupRootClaim::Wait(wait) => wait.wait().await,
            }
        };
        let artifact_root = group_artifact_root(&spec).await;
        validate_artifact_roots(&spec, &artifact_root).await?;

        let task_results = join_all(spec.files().iter().map(|file| {
            let manager = Arc::clone(&manager);
            let destination = spec.destination_for(file);
            let artifact_root = artifact_root.join(compute_download_id(&destination).to_string());
            let source = file.source.clone();
            let file_check = file.check.clone();
            let expected_bytes = file.expected_bytes;
            let relative_path = file.relative_path.clone();
            async move {
                let existing_task = manager
                    .open_existing_http_file_download_task_with_artifact_root(
                        source.clone(),
                        &destination,
                        file_check.clone(),
                        expected_bytes,
                        &artifact_root,
                    )
                    .await
                    .map_err(|error| FileDownloadFailure::new(relative_path.clone(), error))?;
                let child_task = match existing_task {
                    Some(task) => Some(GroupChild::new(task).await),
                    None => None,
                };
                Ok(GroupMember::new(
                    relative_path,
                    expected_bytes,
                    source,
                    destination,
                    file_check,
                    artifact_root,
                    child_task,
                ))
            }
        }))
        .await;

        let mut members = Vec::with_capacity(task_results.len());
        let mut failures = Vec::new();
        for result in task_results {
            match result {
                Ok(member) => members.push(member),
                Err(failure) => failures.push(failure),
            }
        }
        if !failures.is_empty() {
            sort_failures(&mut failures);
            return Err(FileDownloadGroupError::file_failures(FileDownloadGroupOperation::Create, failures));
        }

        let (actor_count, _) = tokio_watch_channel(0);
        let owner = Arc::new(FileDownloadGroupOwner {
            runtime_handle: rt::RuntimeHandle::current(),
            manager,
            spec,
            artifact_root,
            members: members.into(),
            actor_count,
            release_watcher_running: AtomicBool::new(false),
        });
        let group = Self::spawn(owner);
        reservation.publish(&group.inner);
        Ok(group)
    }

    fn spawn(owner: Arc<FileDownloadGroupOwner>) -> Self {
        let members = Arc::clone(&owner.members);
        let member_snapshots = current_member_snapshots(&members);
        let watched_members = members.iter().map(|member| member.child().is_some()).collect();
        let initial_state = reduce_group_state(&members, &member_snapshots, &retained_member_failures(&members));
        let (state_sender, state_receiver) = tokio_watch_channel(initial_state);
        let (command_sender, command_receiver) = tokio_mpsc_channel(32);
        let lease = GroupActorLease::new(Arc::clone(&owner));
        owner.runtime_handle.spawn(
            FileDownloadGroupActor {
                owner: Arc::clone(&owner),
                _lease: lease,
                members: Arc::clone(&members),
                member_snapshots,
                watched_members,
                operation_failures: Vec::new(),
                command_receiver,
                state_sender,
                attempt_waiters: Vec::new(),
            }
            .run(),
        );

        Self {
            inner: Arc::new(FileDownloadGroupHandle {
                owner,
                command_sender,
                state_receiver,
            }),
        }
    }

    pub fn spec(&self) -> &FileDownloadGroupSpec {
        &self.inner.owner.spec
    }

    pub fn state(&self) -> FileDownloadGroupState {
        self.inner.state_receiver.borrow().clone()
    }

    #[doc(hidden)]
    #[deprecated(note = "per-file tasks are private implementation details of FileDownloadGroup")]
    pub fn legacy_file_task_by_download_id(
        &self,
        download_id: crate::DownloadId,
    ) -> Option<Arc<dyn FileDownloadTask>> {
        self.inner
            .owner
            .members
            .iter()
            .filter_map(GroupMember::child)
            .find(|child| child.task.download_id() == download_id)
            .map(|child| child.task)
    }

    /// Returns a stream that immediately yields the current state and then its replacements.
    pub fn subscribe(&self) -> WatchStream<FileDownloadGroupState> {
        WatchStream::new(self.inner.state_receiver.clone())
    }

    pub async fn download(&self) -> Result<DownloadAttempt, FileDownloadGroupError> {
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.inner
            .command_sender
            .send(GroupCommand::Download {
                reply_sender,
            })
            .await
            .map_err(|_| FileDownloadGroupError::ActorStopped)?;
        reply_receiver.await.map_err(|_| FileDownloadGroupError::ActorStopped)
    }

    pub async fn pause(&self) -> Result<(), FileDownloadGroupError> {
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.inner
            .command_sender
            .send(GroupCommand::Pause {
                reply_sender,
            })
            .await
            .map_err(|_| FileDownloadGroupError::ActorStopped)?;
        reply_receiver.await.map_err(|_| FileDownloadGroupError::ActorStopped)?
    }

    pub async fn cancel(&self) -> Result<(), FileDownloadGroupError> {
        let (reply_sender, reply_receiver) = tokio_oneshot_channel();
        self.inner
            .command_sender
            .send(GroupCommand::Cancel {
                reply_sender,
            })
            .await
            .map_err(|_| FileDownloadGroupError::ActorStopped)?;
        reply_receiver.await.map_err(|_| FileDownloadGroupError::ActorStopped)?
    }
}

fn root_conflict(destination_root: &Path) -> FileDownloadGroupError {
    FileDownloadGroupError::RootConflict {
        destination_root: destination_root.to_path_buf(),
    }
}

fn claim_group_root(
    registry_key: &Path,
    destination_root: &Path,
    spec: &FileDownloadGroupSpec,
) -> Result<GroupRootClaim, FileDownloadGroupError> {
    let mut roots = GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let overlapping_key = roots.keys().find(|root| roots_overlap(root, registry_key)).cloned();
    let Some(overlapping_key) = overlapping_key else {
        let state = Arc::new(GroupRootReservationState::new());
        roots.insert(registry_key.to_path_buf(), GroupRootEntry::Reserved(Arc::clone(&state)));
        return Ok(GroupRootClaim::Reserved(GroupRootReservation {
            registry_key: registry_key.to_path_buf(),
            state,
            active: true,
        }));
    };

    if overlapping_key != registry_key {
        return Err(root_conflict(destination_root));
    }

    match roots.get_mut(&overlapping_key).expect("the overlapping root came from this registry") {
        GroupRootEntry::Reserved(state) => Ok(GroupRootClaim::Wait(GroupRootWait::Construction(Arc::clone(state)))),
        GroupRootEntry::Live(group) => {
            if group.owner.spec.as_ref() != spec {
                return Err(root_conflict(destination_root));
            }
            if let Some(handle) = group.handle.upgrade() {
                return Ok(GroupRootClaim::Existing(handle));
            }
            if *group.owner.actor_count.borrow() > 0 {
                return Ok(GroupRootClaim::Wait(GroupRootWait::Actor(Arc::clone(&group.owner))));
            }

            let reopened = FileDownloadGroup::spawn(Arc::clone(&group.owner));
            group.handle = Arc::downgrade(&reopened.inner);
            Ok(GroupRootClaim::Existing(reopened.inner))
        },
    }
}

impl GroupRootReservationState {
    fn new() -> Self {
        let (completed, _) = tokio_watch_channel(false);
        Self {
            completed,
        }
    }

    async fn wait(&self) {
        let mut completed = self.completed.subscribe();
        if !*completed.borrow() {
            let _ = completed.changed().await;
        }
    }

    fn complete(&self) {
        self.completed.send_replace(true);
    }
}

impl GroupRootWait {
    async fn wait(self) {
        match self {
            Self::Construction(state) => state.wait().await,
            Self::Actor(owner) => {
                let mut actor_count = owner.actor_count.subscribe();
                while *actor_count.borrow() > 0 {
                    if actor_count.changed().await.is_err() {
                        break;
                    }
                }
            },
        }
    }
}

impl GroupRootReservation {
    fn publish(
        mut self,
        group: &Arc<FileDownloadGroupHandle>,
    ) {
        let mut roots = GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let owns_reservation = matches!(
            roots.get(&self.registry_key),
            Some(GroupRootEntry::Reserved(state)) if Arc::ptr_eq(state, &self.state)
        );
        debug_assert!(owns_reservation, "group root reservation disappeared before publication");
        if owns_reservation {
            roots.insert(
                self.registry_key.clone(),
                GroupRootEntry::Live(GroupRootOwner {
                    owner: Arc::clone(&group.owner),
                    handle: Arc::downgrade(group),
                }),
            );
        }
        self.active = false;
        drop(roots);
        self.state.complete();
    }
}

impl Drop for GroupRootReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let mut roots = GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let owns_reservation = matches!(
            roots.get(&self.registry_key),
            Some(GroupRootEntry::Reserved(state)) if Arc::ptr_eq(state, &self.state)
        );
        if owns_reservation {
            roots.remove(&self.registry_key);
        }
        drop(roots);
        self.state.complete();
    }
}

fn schedule_group_root_release(owner: Arc<FileDownloadGroupOwner>) {
    if owner.release_watcher_running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
        owner.runtime_handle.clone().spawn(release_group_root_when_idle(owner));
    }
}

async fn release_group_root_when_idle(owner: Arc<FileDownloadGroupOwner>) {
    loop {
        if !wait_for_group_owner_idle(&owner).await {
            owner.release_watcher_running.store(false, Ordering::Release);
            return;
        }

        let key = root_registry_key(owner.spec.destination_root());
        let removed = {
            let mut roots = GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let owns_root = matches!(
                roots.get(&key),
                Some(GroupRootEntry::Live(group))
                    if Arc::ptr_eq(&group.owner, &owner)
                        && group.handle.strong_count() == 0
                        && !owner.is_live()
            );
            owns_root.then(|| roots.remove(&key)).flatten().is_some()
        };
        if removed {
            for child in owner.members.iter().filter_map(GroupMember::child) {
                if let Err(error) = owner.manager.release_file_task_if_inactive(child.task).await {
                    tracing::debug!("failed to release retired file task: {error}");
                }
            }
            owner.release_watcher_running.store(false, Ordering::Release);
            return;
        }

        owner.release_watcher_running.store(false, Ordering::Release);
        let handle_is_gone = {
            let roots = GROUP_ROOTS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            matches!(
                roots.get(&key),
                Some(GroupRootEntry::Live(group))
                    if Arc::ptr_eq(&group.owner, &owner) && group.handle.strong_count() == 0
            )
        };
        if !handle_is_gone
            || owner.release_watcher_running.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err()
        {
            return;
        }
    }
}

async fn wait_for_group_owner_idle(owner: &FileDownloadGroupOwner) -> bool {
    let mut actor_count = owner.actor_count.subscribe();
    loop {
        while *actor_count.borrow() > 0 {
            if actor_count.changed().await.is_err() {
                return false;
            }
        }

        let mut updates = SelectAll::new();
        for child in owner.members.iter().filter_map(GroupMember::child) {
            updates.push(WatchStream::from_changes(child.snapshot_receiver));
        }
        if !owner.has_downloading_member() {
            return true;
        }

        tokio::select! {
            changed = actor_count.changed() => {
                if changed.is_err() {
                    return false;
                }
            }
            _ = updates.next() => {}
        }
    }
}

async fn group_artifact_root(spec: &FileDownloadGroupSpec) -> PathBuf {
    group_artifact_root_for_location(
        spec.destination_root(),
        spec.files(),
        destination_root_is_mount_point(spec.destination_root()).await,
    )
}

fn group_artifact_root_for_location(
    destination_root: &Path,
    files: &[FileDownloadRequest],
    root_is_mount_point: bool,
) -> PathBuf {
    let root_id = compute_download_id(destination_root);
    if !root_is_mount_point {
        return destination_root
            .parent()
            .unwrap_or(destination_root)
            .join(".uzu-download-manager")
            .join(root_id.to_string());
    }

    for suffix in 0..=files.len() {
        let directory_name = match suffix {
            0 => format!(".uzu-download-manager-{root_id}"),
            suffix => format!(".uzu-download-manager-{root_id}-{suffix}"),
        };
        if files.iter().all(|file| !uses_top_level_directory(file, &directory_name)) {
            return destination_root.join(directory_name);
        }
    }

    unreachable!("there are more artifact directory candidates than declared file paths")
}

fn uses_top_level_directory(
    file: &FileDownloadRequest,
    directory_name: &str,
) -> bool {
    let Some(Component::Normal(name)) = file.relative_path.as_path().components().next() else {
        return false;
    };
    name.to_str().is_some_and(|name| name.eq_ignore_ascii_case(directory_name))
}

#[cfg(unix)]
async fn destination_root_is_mount_point(destination_root: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;

    let Some(parent) = destination_root.parent() else {
        return true;
    };
    let Ok(root_metadata) = tokio::fs::metadata(destination_root).await else {
        return false;
    };
    match tokio::fs::metadata(parent).await {
        Ok(parent_metadata) => root_metadata.dev() != parent_metadata.dev(),
        Err(_) => true,
    }
}

#[cfg(windows)]
async fn destination_root_is_mount_point(destination_root: &Path) -> bool {
    let Some(parent) = destination_root.parent() else {
        return true;
    };
    let Ok(canonical_root) = tokio::fs::canonicalize(destination_root).await else {
        return false;
    };
    match tokio::fs::canonicalize(parent).await {
        Ok(canonical_parent) => !root_registry_key(&canonical_root).starts_with(root_registry_key(&canonical_parent)),
        Err(_) => true,
    }
}

#[cfg(not(any(unix, windows)))]
async fn destination_root_is_mount_point(destination_root: &Path) -> bool {
    destination_root.parent().is_none()
}

fn roots_overlap(
    left: &Path,
    right: &Path,
) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn root_registry_key(path: &Path) -> PathBuf {
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        PathBuf::from(crate::file_download_request::portable_path_key(path))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        path.to_path_buf()
    }
}

impl fmt::Debug for FileDownloadGroup {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter
            .debug_struct("FileDownloadGroup")
            .field("destination_root", &self.inner.owner.spec.destination_root())
            .field("file_count", &self.inner.owner.members.len())
            .finish()
    }
}

impl DownloadAttempt {
    pub async fn wait(self) -> Result<FileDownloadGroupState, FileDownloadGroupError> {
        self.completion_receiver.await.map_err(|_| FileDownloadGroupError::ActorStopped)
    }
}

impl fmt::Debug for DownloadAttempt {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter.debug_struct("DownloadAttempt").finish_non_exhaustive()
    }
}

impl FileDownloadGroupActor {
    async fn run(mut self) {
        let mut member_updates = SelectAll::new();
        for (index, member) in self.members.iter().enumerate() {
            if let Some(child) = member.child() {
                member_updates.push(member_snapshot_stream(index, child.snapshot_receiver));
            }
        }

        loop {
            tokio::select! {
                command = self.command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.handle_command(command, &mut member_updates).await;
                }
                update = member_updates.next(), if !member_updates.is_empty() => {
                    if let Some((index, snapshot)) = update {
                        self.member_snapshots[index] = snapshot;
                        self.publish_state();
                    }
                }
            }
        }
    }

    async fn handle_command(
        &mut self,
        command: GroupCommand,
        member_updates: &mut SelectAll<GroupMemberSnapshotStream>,
    ) {
        match command {
            GroupCommand::Download {
                reply_sender,
            } => self.start_download_attempt(reply_sender, member_updates).await,
            GroupCommand::Pause {
                reply_sender,
            } => {
                let result = self.pause_downloading_members().await;
                let _ = reply_sender.send(result);
            },
            GroupCommand::Cancel {
                reply_sender,
            } => {
                let result = self.cancel_and_delete_members(member_updates).await;
                let _ = reply_sender.send(result);
            },
        }
    }

    async fn start_download_attempt(
        &mut self,
        reply_sender: TokioOneshotSender<DownloadAttempt>,
        member_updates: &mut SelectAll<GroupMemberSnapshotStream>,
    ) {
        self.refresh_member_snapshots();
        self.publish_state();

        if !self.attempt_waiters.is_empty() {
            let (completion_sender, completion_receiver) = tokio_oneshot_channel();
            self.attempt_waiters.push(completion_sender);
            let _ = reply_sender.send(DownloadAttempt {
                completion_receiver,
            });
            return;
        }

        let (completion_sender, completion_receiver) = tokio_oneshot_channel();
        self.attempt_waiters.push(completion_sender);
        self.operation_failures.clear();

        let selected_indices: Vec<_> = self
            .member_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                !matches!(snapshot.state.phase, FileDownloadPhase::Downloaded | FileDownloadPhase::Downloading)
            })
            .map(|(index, _)| index)
            .collect();
        let (materialized, mut failures) = self.materialize_members(&selected_indices, member_updates).await;
        let selected: Vec<_> = materialized
            .into_iter()
            .filter(|(index, _)| {
                let should_download = !matches!(
                    self.member_snapshots[*index].state.phase,
                    FileDownloadPhase::Downloaded | FileDownloadPhase::Downloading
                );
                if !should_download {
                    self.members[*index].set_retained_failure(None);
                }
                should_download
            })
            .collect();
        let results = join_all(selected.iter().map(|(_, child)| child.task.download())).await;
        failures.extend(selected_member_failures(
            &self.members,
            &selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            results,
        ));
        sort_failures(&mut failures);
        self.operation_failures = failures;
        self.refresh_member_snapshots();
        self.publish_state();

        let _ = reply_sender.send(DownloadAttempt {
            completion_receiver,
        });
    }

    async fn pause_downloading_members(&mut self) -> Result<(), FileDownloadGroupError> {
        self.operation_failures.clear();
        self.refresh_member_snapshots();

        let selected_indices: Vec<_> = self
            .member_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| matches!(snapshot.state.phase, FileDownloadPhase::Downloading))
            .map(|(index, _)| index)
            .collect();
        let selected: Vec<_> = selected_indices
            .iter()
            .filter_map(|index| self.members[*index].child().map(|child| (*index, child)))
            .collect();
        let results = join_all(selected.iter().map(|(_, child)| child.task.pause())).await;
        self.refresh_member_snapshots();
        self.operation_failures = selected_member_failures(
            &self.members,
            &selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            results,
        )
        .into_iter()
        .filter(|failure| {
            if !matches!(failure.error, DownloadError::InvalidStateTransition) {
                return true;
            }
            self.members
                .iter()
                .zip(&self.member_snapshots)
                .find(|(member, _)| member.relative_path == failure.relative_path)
                .is_some_and(|(_, snapshot)| matches!(snapshot.state.phase, FileDownloadPhase::Downloading))
        })
        .collect();
        self.publish_state();

        operation_result(FileDownloadGroupOperation::Pause, &self.operation_failures)
    }

    async fn cancel_and_delete_members(
        &mut self,
        member_updates: &mut SelectAll<GroupMemberSnapshotStream>,
    ) -> Result<(), FileDownloadGroupError> {
        self.operation_failures.clear();
        let mut indices = (0..self.members.len()).collect::<Vec<_>>();
        indices.sort_by(|left, right| self.members[*left].relative_path.cmp(&self.members[*right].relative_path));

        let mut failures = BTreeMap::new();
        if let Err(error) = validate_owned_paths(&self.owner).await {
            for failure in error.failures() {
                failures.insert(failure.relative_path.clone(), failure.error.clone());
            }
        }

        let manager = Arc::clone(&self.owner.manager);
        for index in indices {
            let child = match materialize_member(self.members[index].clone(), Arc::clone(&manager)).await {
                Ok(child) => {
                    self.member_snapshots[index] = child.snapshot_receiver.borrow().clone();
                    self.members[index].set_retained_failure(None);
                    if !self.watched_members[index] {
                        member_updates.push(member_snapshot_stream(index, child.snapshot_receiver.clone()));
                        self.watched_members[index] = true;
                    }
                    child
                },
                Err(error) => {
                    self.members[index].set_retained_failure(Some(error.clone()));
                    failures.insert(self.members[index].relative_path.clone(), error);
                    continue;
                },
            };
            match child.task.cancel_and_delete().await {
                Ok(()) => self.members[index].set_retained_failure(None),
                Err(error) => {
                    failures.insert(self.members[index].relative_path.clone(), error);
                },
            }
        }
        if failures.is_empty() {
            match tokio::fs::remove_dir(&self.owner.artifact_root).await {
                Ok(()) => {},
                Err(error)
                    if matches!(error.kind(), std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty) => {
                },
                Err(error) => {
                    failures.insert(self.members[0].relative_path.clone(), DownloadError::from(error));
                },
            }
        }
        self.operation_failures =
            failures.into_iter().map(|(relative_path, error)| FileDownloadFailure::new(relative_path, error)).collect();
        self.refresh_member_snapshots();
        self.publish_state();

        operation_result(FileDownloadGroupOperation::Cancel, &self.operation_failures)
    }

    async fn materialize_members(
        &mut self,
        indices: &[usize],
        member_updates: &mut SelectAll<GroupMemberSnapshotStream>,
    ) -> (Vec<(usize, GroupChild)>, Vec<FileDownloadFailure>) {
        if let Err(error) = validate_owned_paths(&self.owner).await {
            let failures = error.failures().to_vec();
            for failure in &failures {
                if let Some(member) = self.members.iter().find(|member| member.relative_path == failure.relative_path) {
                    member.set_retained_failure(Some(failure.error.clone()));
                }
            }
            return (Vec::new(), failures);
        }

        let manager = Arc::clone(&self.owner.manager);
        let results = join_all(indices.iter().map(|index| {
            let member = self.members[*index].clone();
            let manager = Arc::clone(&manager);
            async move { materialize_member(member, manager).await }
        }))
        .await;

        let mut materialized = Vec::with_capacity(results.len());
        let mut failures = Vec::new();
        for (index, result) in indices.iter().copied().zip(results) {
            match result {
                Ok(child) => {
                    self.member_snapshots[index] = child.snapshot_receiver.borrow().clone();
                    self.members[index].set_retained_failure(None);
                    if !self.watched_members[index] {
                        member_updates.push(member_snapshot_stream(index, child.snapshot_receiver.clone()));
                        self.watched_members[index] = true;
                    }
                    materialized.push((index, child));
                },
                Err(error) => {
                    self.members[index].set_retained_failure(Some(error.clone()));
                    failures.push(FileDownloadFailure::new(self.members[index].relative_path.clone(), error));
                },
            }
        }
        (materialized, failures)
    }

    fn refresh_member_snapshots(&mut self) {
        self.member_snapshots = current_member_snapshots(&self.members);
    }

    fn publish_state(&mut self) {
        let mut failures = retained_member_failures(&self.members);
        failures.extend(self.operation_failures.iter().cloned());
        let state = reduce_group_state(&self.members, &self.member_snapshots, &failures);
        if *self.state_sender.borrow() != state {
            self.state_sender.send_replace(state.clone());
        }

        if !matches!(state.phase, FileDownloadGroupPhase::Downloading) {
            for waiter in self.attempt_waiters.drain(..) {
                let _ = waiter.send(state.clone());
            }
        }
    }
}

fn member_snapshot_stream(
    index: usize,
    snapshot_receiver: TokioWatchReceiver<FileDownloadSnapshot>,
) -> GroupMemberSnapshotStream {
    Box::pin(WatchStream::from_changes(snapshot_receiver).map(move |snapshot| (index, snapshot)))
}

async fn materialize_member(
    member: GroupMember,
    manager: Arc<dyn FileDownloadManager>,
) -> Result<GroupChild, DownloadError> {
    if let Some(child) = member.child() {
        return Ok(child);
    }
    let task = manager
        .http_file_download_task_with_artifact_root(
            member.source.clone(),
            &member.destination,
            member.file_check.clone(),
            member.expected_bytes,
            &member.artifact_root,
        )
        .await?;
    Ok(member.set_child_if_missing(GroupChild::new(task).await))
}

fn current_member_snapshots(members: &[GroupMember]) -> Vec<FileDownloadSnapshot> {
    members.iter().map(GroupMember::snapshot).collect()
}

fn retained_member_failures(members: &[GroupMember]) -> Vec<FileDownloadFailure> {
    members
        .iter()
        .filter_map(|member| {
            member.retained_failure().map(|error| FileDownloadFailure::new(member.relative_path.clone(), error))
        })
        .collect()
}

fn selected_member_failures(
    members: &[GroupMember],
    indices: &[usize],
    results: Vec<Result<(), DownloadError>>,
) -> Vec<FileDownloadFailure> {
    let mut failures = indices
        .iter()
        .zip(results)
        .filter_map(|(index, result)| {
            result.err().map(|error| FileDownloadFailure::new(members[*index].relative_path.clone(), error))
        })
        .collect::<Vec<_>>();
    sort_failures(&mut failures);
    failures
}

fn operation_result(
    operation: FileDownloadGroupOperation,
    failures: &[FileDownloadFailure],
) -> Result<(), FileDownloadGroupError> {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(FileDownloadGroupError::file_failures(operation, failures.to_vec()))
    }
}

fn sort_failures(failures: &mut [FileDownloadFailure]) {
    failures.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
}

fn reduce_group_state(
    members: &[GroupMember],
    member_snapshots: &[FileDownloadSnapshot],
    operation_failures: &[FileDownloadFailure],
) -> FileDownloadGroupState {
    debug_assert_eq!(members.len(), member_snapshots.len());

    let mut downloaded_bytes = 0_u64;
    let mut total_bytes = Some(0_u64);
    let mut completed_files = 0_usize;
    let mut any_downloading = false;
    let mut any_paused = false;
    let mut failures = BTreeMap::<RelativeFilePath, DownloadError>::new();

    for (member, snapshot) in members.iter().zip(member_snapshots) {
        let state = &snapshot.state;
        downloaded_bytes = downloaded_bytes.saturating_add(state.downloaded_bytes);
        let member_total = member.expected_bytes.or(snapshot.total_bytes);
        total_bytes = match (total_bytes, member_total) {
            (Some(total), Some(member_total)) => total.checked_add(member_total),
            _ => None,
        };

        match &state.phase {
            FileDownloadPhase::NotDownloaded => {},
            FileDownloadPhase::Downloading => any_downloading = true,
            FileDownloadPhase::Paused => any_paused = true,
            FileDownloadPhase::Downloaded => completed_files = completed_files.saturating_add(1),
            FileDownloadPhase::LockedByOther(owner) => {
                failures.insert(member.relative_path.clone(), DownloadError::LockedByOther(owner.clone()));
            },
            FileDownloadPhase::Error(message) => {
                failures.insert(
                    member.relative_path.clone(),
                    snapshot.failure.clone().unwrap_or_else(|| DownloadError::Backend(message.clone())),
                );
            },
        }
    }

    for failure in operation_failures {
        failures.insert(failure.relative_path.clone(), failure.error.clone());
    }

    let failures: Arc<[FileDownloadFailure]> = failures
        .into_iter()
        .map(|(relative_path, error)| FileDownloadFailure::new(relative_path, error))
        .collect::<Vec<_>>()
        .into();
    let total_files = members.len();
    let phase = if any_downloading {
        FileDownloadGroupPhase::Downloading
    } else if !failures.is_empty() {
        if failures.iter().all(|failure| matches!(failure.error, DownloadError::LockedByOther(_))) {
            FileDownloadGroupPhase::Locked
        } else {
            FileDownloadGroupPhase::Error
        }
    } else if completed_files == total_files {
        FileDownloadGroupPhase::Downloaded
    } else if any_paused || completed_files > 0 {
        FileDownloadGroupPhase::Paused
    } else {
        FileDownloadGroupPhase::NotDownloaded
    };

    FileDownloadGroupState {
        phase,
        downloaded_bytes,
        total_bytes,
        completed_files,
        total_files,
        failures,
    }
}

async fn validate_owned_paths(owner: &FileDownloadGroupOwner) -> Result<(), FileDownloadGroupError> {
    let mut failures = BTreeMap::new();
    if let Err(error) = validate_existing_symlinks(&owner.spec).await {
        for failure in error.failures() {
            failures.insert(failure.relative_path.clone(), failure.error.clone());
        }
    }
    if let Err(error) = validate_artifact_roots(&owner.spec, &owner.artifact_root).await {
        for failure in error.failures() {
            failures.insert(failure.relative_path.clone(), failure.error.clone());
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(FileDownloadGroupError::file_failures(
            FileDownloadGroupOperation::Create,
            failures.into_iter().map(|(relative_path, error)| FileDownloadFailure::new(relative_path, error)).collect(),
        ))
    }
}

async fn validate_artifact_roots(
    spec: &FileDownloadGroupSpec,
    artifact_root: &Path,
) -> Result<(), FileDownloadGroupError> {
    #[cfg(target_family = "wasm")]
    {
        let _ = (spec, artifact_root);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let mut failures = Vec::new();
        for file in spec.files() {
            let member_root = artifact_root.join(compute_download_id(&spec.destination_for(file)).to_string());
            match first_invalid_owned_directory(&member_root).await {
                Ok(Some((path, reason))) => failures.push(FileDownloadFailure::new(
                    file.relative_path.clone(),
                    DownloadError::Io(format!("download artifact path {reason}: {}", path.display())),
                )),
                Ok(None) => {},
                Err(error) => {
                    failures.push(FileDownloadFailure::new(file.relative_path.clone(), DownloadError::from(error)));
                },
            }
        }

        if failures.is_empty() {
            Ok(())
        } else {
            sort_failures(&mut failures);
            Err(FileDownloadGroupError::file_failures(FileDownloadGroupOperation::Create, failures))
        }
    }
}

#[cfg(not(target_family = "wasm"))]
async fn first_invalid_owned_directory(path: &Path) -> Result<Option<(PathBuf, &'static str)>, std::io::Error> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Ok(Some((current, "contains a symlink")));
            },
            Ok(metadata) if !metadata.is_dir() => return Ok(Some((current, "contains a non-directory ancestor"))),
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

async fn validate_existing_symlinks(spec: &FileDownloadGroupSpec) -> Result<PathBuf, FileDownloadGroupError> {
    #[cfg(target_family = "wasm")]
    {
        let _ = spec;
        Ok(spec.destination_root().to_path_buf())
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let mut normalized_root = normalize_absolute_path(spec.destination_root()).map_err(|error| {
            FileDownloadGroupError::file_failures(
                FileDownloadGroupOperation::Create,
                vec![FileDownloadFailure::new(spec.files()[0].relative_path.clone(), DownloadError::from(error))],
            )
        })?;
        let mut failures = Vec::new();
        match first_disallowed_symlink_ancestor(&normalized_root).await {
            Ok(Some(symlink)) => failures.push(FileDownloadFailure::new(
                spec.files()[0].relative_path.clone(),
                DownloadError::Io(format!("destination root has a symlinked ancestor: {}", symlink.display())),
            )),
            Ok(None) => {},
            Err(error) => failures
                .push(FileDownloadFailure::new(spec.files()[0].relative_path.clone(), DownloadError::from(error))),
        }
        match tokio::fs::symlink_metadata(spec.destination_root()).await {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                failures.push(FileDownloadFailure::new(
                    spec.files()[0].relative_path.clone(),
                    DownloadError::Io(format!("destination root is a symlink: {}", spec.destination_root().display())),
                ));
            },
            Ok(_) => {
                if let Ok(canonical_root) = tokio::fs::canonicalize(spec.destination_root()).await {
                    normalized_root = canonical_root;
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut missing_suffix = Vec::new();
                let mut ancestor = spec.destination_root();
                while let Some(parent) = ancestor.parent() {
                    if let Some(name) = ancestor.file_name() {
                        missing_suffix.push(name.to_owned());
                    }
                    match tokio::fs::symlink_metadata(parent).await {
                        Ok(_) => {
                            if let Ok(mut canonical_ancestor) = tokio::fs::canonicalize(parent).await {
                                for component in missing_suffix.iter().rev() {
                                    canonical_ancestor.push(component);
                                }
                                normalized_root = canonical_ancestor;
                            }
                            break;
                        },
                        Err(parent_error) if parent_error.kind() == std::io::ErrorKind::NotFound => {
                            ancestor = parent;
                        },
                        Err(parent_error) => {
                            failures.push(FileDownloadFailure::new(
                                spec.files()[0].relative_path.clone(),
                                DownloadError::from(parent_error),
                            ));
                            break;
                        },
                    }
                }
            },
            Err(error) => {
                failures
                    .push(FileDownloadFailure::new(spec.files()[0].relative_path.clone(), DownloadError::from(error)));
            },
        }
        for file in spec.files() {
            let mut current = spec.destination_root().to_path_buf();
            let mut paths = Vec::new();
            for component in file.relative_path.as_path().components() {
                current.push(component);
                paths.push(current.clone());
            }

            for path in paths {
                match tokio::fs::symlink_metadata(&path).await {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        failures.push(FileDownloadFailure::new(
                            file.relative_path.clone(),
                            DownloadError::Io(format!("destination path contains symlink: {}", path.display())),
                        ));
                        break;
                    },
                    Ok(_) => {},
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                    Err(error) => {
                        failures.push(FileDownloadFailure::new(file.relative_path.clone(), DownloadError::from(error)));
                        break;
                    },
                }
            }
        }

        if failures.is_empty() {
            Ok(normalized_root)
        } else {
            sort_failures(&mut failures);
            Err(FileDownloadGroupError::file_failures(FileDownloadGroupOperation::Create, failures))
        }
    }
}

async fn first_disallowed_symlink_ancestor(path: &Path) -> Result<Option<PathBuf>, std::io::Error> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        match tokio::fs::symlink_metadata(&current).await {
            Ok(metadata) if metadata.file_type().is_symlink() && !is_platform_path_alias(&current) => {
                return Ok(Some(current));
            },
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error),
        }
    }
    Ok(None)
}

fn is_platform_path_alias(path: &Path) -> bool {
    #[cfg(target_os = "macos")]
    {
        matches!(path.to_str(), Some("/var" | "/tmp" | "/etc"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        false
    }
}

fn normalize_absolute_path(path: &Path) -> Result<PathBuf, std::io::Error> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => normalized.push(component),
            Component::CurDir => {},
            Component::ParentDir => {
                normalized.pop();
            },
        }
    }
    Ok(normalized)
}

#[cfg(test)]
#[path = "../tests/unit/file_download_group_test.rs"]
mod tests;
