use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use rocket::response::stream::Event;
use tokio::sync::Mutex;
use uzu::{
    engine::Engine,
    session::chat::ChatSession,
    types::{
        model::{Model, ModelAccessibility, ModelSource},
        session::chat::ChatConfig,
    },
};

use crate::common::thinking::ThinkingSupport;

/// A configured model plus its runtime metadata.
#[derive(Clone)]
pub struct ModelSlot {
    pub model: Model,
    pub thinking: ThinkingSupport,
    pub footprint: u64,
}

impl ModelSlot {
    pub fn new(model: Model) -> Self {
        Self {
            thinking: ThinkingSupport::for_model(&model),
            footprint: footprint_bytes(&model),
            model,
        }
    }

    pub fn identifier(&self) -> &str {
        &self.model.identifier
    }
}

/// Best-effort estimate of a model's in-memory footprint in bytes: the sum of
/// its registry file sizes when known; for a local directory, the sum of the
/// on-disk file sizes; otherwise parameters x bits / 8.
pub fn footprint_bytes(model: &Model) -> u64 {
    match &model.accessibility {
        ModelAccessibility::OnDevice {
            source: ModelSource::Registry {
                files,
                ..
            },
        } if !files.is_empty() => {
            return files.iter().fold(0_u64, |total, file| total.saturating_add(file.size.max(0) as u64));
        },
        ModelAccessibility::OnDevice {
            source: ModelSource::Filesystem {
                path,
            },
        } => {
            return filesystem_bytes(path);
        },
        _ => {},
    }
    let Some(properties) = &model.properties else {
        return 0;
    };
    let parameters = properties.size.max(0) as u64;
    let bits = model.quantization.as_ref().map_or(16, |quantization| quantization.bits_per_weight as u64);
    parameters.saturating_mul(bits).div_ceil(8)
}

/// Sum of the sizes of every file under `path` (the model's on-disk weight
/// files), deduplicated by resolved path so a weight that appears more than
/// once (a symlink next to its target) counts its physical size only once —
/// which is what unified memory actually maps. Symlinks are followed the way
/// the engine does (`File::open` follows links; model layouts keep optional
/// subdirectories like `speculator/` behind links to share them): file links
/// count their target, directory links are descended into, guarded by a set
/// of canonical directory paths so a link cycle can't loop the walk forever.
/// Broken or unreadable entries count as 0; the dynamic free-memory check in
/// `memory_guard` still bounds what actually gets allocated.
fn filesystem_bytes(path: &str) -> u64 {
    let root = Path::new(path);
    if !root.is_dir() {
        return 0;
    }
    let mut total = 0_u64;
    let mut seen_files: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let mut seen_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let root_key = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    if !seen_dirs.insert(root_key) {
        return 0;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let full = entry.path();
            let link_type = match entry.file_type() {
                Ok(link_type) => link_type,
                Err(_) => continue,
            };
            // A symlink's own type is reported; stat through it to see the
            // target (this is what `File::open` does for weights).
            let target = std::fs::metadata(&full).ok();
            if link_type.is_dir() || (link_type.is_symlink() && target.as_ref().is_some_and(|target| target.is_dir())) {
                // Descend (real or symlinked directory); the canonical path
                // makes a link back to an ancestor a no-op.
                let key = std::fs::canonicalize(&full).unwrap_or_else(|_| full.clone());
                if seen_dirs.insert(key) {
                    stack.push(full);
                }
            } else if let Some(target) = target.filter(|target| target.is_file()) {
                // A regular file, or a link to one: count the physical file
                // once, no matter how many links point at it.
                let key = std::fs::canonicalize(&full).unwrap_or_else(|_| full.clone());
                if seen_files.insert(key) {
                    total = total.saturating_add(target.len());
                }
            }
            // Broken or unreadable entries contribute nothing.
        }
    }
    total
}

/// Memory in use right now, as reported by the OS.
fn available_memory() -> u64 {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.available_memory()
}

/// Deterministic OOM guard. `target_footprint` is what the model will
/// allocate; `available_memory` is what the OS reports free *after* any
/// previous model has already been dropped (a swap frees first, then loads,
/// so the two never coexist). Refuses when the load would not fit with a
/// safety margin, or when a single model would dominate the machine.
pub fn memory_guard(
    target: &str,
    target_footprint: u64,
    available_memory: u64,
    total_memory: u64,
) -> Result<(), String> {
    if total_memory == 0 {
        return Ok(());
    }
    // Static backstop: one model may not take a majority of unified memory.
    if target_footprint as f64 > total_memory as f64 * 0.60 {
        return Err(format!(
            "model {target} needs ~{:.1} GB, over 60% of the {:.1} GB unified memory; refusing to load",
            target_footprint as f64 / 1e9,
            total_memory as f64 / 1e9
        ));
    }
    // Dynamic check: will it fit in what is free right now, keeping a 2 GB
    // margin so the OS and other processes retain headroom?
    const MARGIN: u64 = 2 * 1024 * 1024 * 1024;
    if target_footprint.saturating_add(MARGIN) > available_memory {
        return Err(format!(
            "not enough free memory to load {target}: need ~{:.1} GB (+2 GB margin) but only {:.1} GB is available; refusing to avoid OOM",
            target_footprint as f64 / 1e9,
            available_memory as f64 / 1e9
        ));
    }
    Ok(())
}

/// Errors a request can hit while acquiring a session for a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquireError {
    /// The requested model identifier is not a configured server model.
    UnknownModel {
        requested: String,
        available: Vec<String>,
    },
    /// The OOM guard refused the load.
    MemoryGuard(String),
    /// The engine failed to download or load the model.
    Engine(String),
    /// The client went away while the model was still loading.
    ClientDisconnected,
}

impl std::fmt::Display for AcquireError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            AcquireError::UnknownModel {
                requested,
                available,
            } => {
                write!(f, "unknown model `{requested}`; available: {}", available.join(", "))
            },
            AcquireError::MemoryGuard(message) => write!(f, "memory guard: {message}"),
            AcquireError::Engine(message) => write!(f, "engine error: {message}"),
            AcquireError::ClientDisconnected => write!(f, "client disconnected while loading model"),
        }
    }
}

/// A connection that can be polled *while a model is loading* — emitting a
/// keep-alive and reporting when the client disconnects. Lets
/// [`SessionManager::load_into`] race a (potentially multi-second) load
/// against liveness, so a client that goes away releases the session lock
/// early instead of a dead task holding it until the load finishes and
/// stalling every other model.
pub trait LoadLiveness {
    /// Wait for the next keep-alive tick, emit it, and report whether the
    /// connection is still alive. Returns `false` once the client has gone
    /// away (or the interval is exhausted).
    fn tick(&mut self) -> impl std::future::Future<Output = bool>;
}

/// Newline keep-alive over the raw response stream.
pub struct LivenessNewline {
    keepalive: tokio::time::Interval,
    sender: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
}

impl LivenessNewline {
    pub fn new(
        sender: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
        interval: std::time::Duration,
    ) -> Self {
        Self {
            keepalive: tokio::time::interval(interval),
            sender,
        }
    }
}

impl LoadLiveness for LivenessNewline {
    fn tick(&mut self) -> impl std::future::Future<Output = bool> {
        let LivenessNewline {
            keepalive,
            sender,
        } = self;
        async move {
            keepalive.tick().await;
            sender.send(b"\n".to_vec()).is_ok()
        }
    }
}

/// SSE keep-alive over the chat-completions stream: a comment event (valid
/// anywhere in an event stream, ignored by OpenAI-compatible clients) — the
/// `text/event-stream` wire format does not accept the raw leading whitespace
/// that `LivenessNewline` emits.
pub struct LivenessSseComment {
    keepalive: tokio::time::Interval,
    sender: tokio::sync::mpsc::UnboundedSender<Event>,
}

impl LivenessSseComment {
    pub fn new(
        sender: tokio::sync::mpsc::UnboundedSender<Event>,
        interval: std::time::Duration,
    ) -> Self {
        Self {
            keepalive: tokio::time::interval(interval),
            sender,
        }
    }
}

impl LoadLiveness for LivenessSseComment {
    fn tick(&mut self) -> impl std::future::Future<Output = bool> {
        let LivenessSseComment {
            keepalive,
            sender,
        } = self;
        async move {
            keepalive.tick().await;
            sender.send(Event::comment("keepalive")).is_ok()
        }
    }
}

/// Owns the single in-memory `ChatSession` slot and the configured model
/// catalog. Only one model is ever resident: when a different model is
/// requested, the resident one is unloaded first (under the session lock, so
/// never while a turn is in flight) and the new one is loaded in its place.
#[derive(Clone)]
pub struct SessionManager {
    pub session: Arc<Mutex<Option<ChatSession>>>,
    resident: Arc<Mutex<Option<ModelSlot>>>,
    last_used: Arc<std::sync::Mutex<Instant>>,
    engine: Arc<Engine>,
    slots: Vec<ModelSlot>,
    total_memory: u64,
}

impl SessionManager {
    pub fn new(
        engine: Engine,
        slots: Vec<ModelSlot>,
        total_memory: u64,
    ) -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            resident: Arc::new(Mutex::new(None)),
            last_used: Arc::new(std::sync::Mutex::new(Instant::now())),
            engine: Arc::new(engine),
            slots,
            total_memory,
        }
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.slots.iter().map(|slot| slot.identifier().to_string()).collect()
    }

    /// Resolves an identifier (exact or shorthand) to a configured slot.
    pub async fn resolve(
        &self,
        identifier: &str,
    ) -> Result<ModelSlot, AcquireError> {
        if let Some(slot) = self.slots.iter().find(|slot| slot.identifier().eq_ignore_ascii_case(identifier)) {
            return Ok(slot.clone());
        }
        let model = self
            .engine
            .model(identifier.to_string())
            .await
            .map_err(|error| AcquireError::Engine(error.to_string()))?
            .ok_or_else(|| AcquireError::UnknownModel {
                requested: identifier.to_string(),
                available: self.model_ids(),
            })?;
        // Local checkpoints are identified by directory name alone, so
        // `resolve` can hand back a model that shares a name with a
        // configured slot but lives at a different path (or is a registry
        // model that shares a name with a local one). Only the configured
        // source may be served; anything else is an unknown model, never a
        // silent substitution.
        self.slots.iter().find(|slot| matches_sources(&slot.model, &model)).cloned().ok_or_else(|| {
            AcquireError::UnknownModel {
                requested: identifier.to_string(),
                available: self.model_ids(),
            }
        })
    }

    /// Ensures a live session for `slot` exists in `session_slot`, loading
    /// (and, if needed, swapping from another resident model) as required.
    ///
    /// The caller must hold `self.session`'s lock for the whole turn and pass
    /// its slot through here: that is what serializes turns and keeps the
    /// idle-unload watcher from touching a session a turn is using.
    pub async fn load_into(
        &self,
        slot: &ModelSlot,
        session_slot: &mut Option<ChatSession>,
        liveness: &mut impl LoadLiveness,
    ) -> Result<ChatSession, AcquireError> {
        // A turn has started: the idle timer must not fire for this turn's
        // duration (a turn can outlive the idle timeout itself).
        *self.last_used.lock().unwrap() = Instant::now();

        // Fast path: the requested model is already the resident one and a
        // live session is present — nothing to load.
        let current = self.resident.lock().await.clone();
        if current.as_ref().is_some_and(|current| current.model.identifier == slot.identifier())
            && session_slot.is_some()
        {
            return Ok(session_slot.clone().unwrap());
        }

        // A different model is resident: drop it first so its GPU memory is
        // freed before the new model allocates. They never coexist.
        if current.as_ref().is_some_and(|current| current.model.identifier != slot.identifier()) {
            tracing::info!("swapping model: {} -> {}", current.as_ref().unwrap().identifier(), slot.identifier());
            *session_slot = None;
            *self.resident.lock().await = None;
        }

        // Remote models allocate no local checkpoint — the backend only opens
        // an API client — so the local-memory OOM guard does not apply to
        // them and is skipped outright (it would otherwise refuse on the
        // fixed 2 GB margin even though nothing is loaded locally).
        if matches!(slot.model.accessibility, ModelAccessibility::Remote { .. }) {
            return load_model(self, slot, session_slot, liveness).await;
        }

        // OOM guard using live available memory. After a swap the OS may need
        // a moment to reclaim the just-freed pages, so retry briefly. The wait
        // is short (200 ms steps) relative to the keepalive interval, so no
        // liveness race is needed here — the load itself is where the client
        // can disconnect.
        let mut guard_error =
            match memory_guard(slot.identifier(), slot.footprint, available_memory(), self.total_memory) {
                Ok(()) => return load_model(self, slot, session_slot, liveness).await,
                Err(error) => error,
            };
        for _ in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            match memory_guard(slot.identifier(), slot.footprint, available_memory(), self.total_memory) {
                Ok(()) => return load_model(self, slot, session_slot, liveness).await,
                Err(error) => guard_error = error,
            }
        }
        Err(AcquireError::MemoryGuard(guard_error))
    }

    /// Unloads the resident model when the session has been idle for at least
    /// `timeout`.
    ///
    /// The predicate and the clear both run under the session lock, and
    /// `last_used` is re-read immediately before clearing — the two must stay
    /// atomic. A separate check-then-release would let a request queued
    /// behind this check acquire the session between them, finish its turn,
    /// and have a just-used session cleared out from under it (cold-loading
    /// again on the next request). By the time a turn releases the lock,
    /// `TurnUsageGuard` has already stamped `last_used` (reverse drop order:
    /// the guard drops before the session guard), so the re-validation sees a
    /// fresh clock and a long, just-finished turn is never unloaded.
    pub async fn try_unload_idle(
        &self,
        timeout: Duration,
    ) {
        let mut slot = self.session.lock().await;
        if slot.is_some() && self.last_used.lock().unwrap().elapsed() >= timeout {
            *slot = None;
            *self.resident.lock().await = None;
            tracing::info!("model idle for {timeout:?} — unloading to free memory");
        }
    }

    /// Marks the session as used *now*, restarting the idle clock.
    ///
    /// Called by `TurnUsageGuard::drop` at the end of every turn. Without it,
    /// only `load_into` (turn *start*) would advance the clock, and a turn
    /// that runs for at least `timeout` would be unloaded the instant it
    /// released the session lock — discarding the prefix cache and making the
    /// next request cold-load.
    pub fn mark_used(&self) {
        *self.last_used.lock().unwrap() = Instant::now();
    }
}

/// Stamps [`SessionManager::mark_used`] the moment a turn finishes, whatever
/// its outcome (response, error, or client disconnect). Held by a request
/// handler for the whole turn, so idle off-load starts counting from the end
/// of the last turn, not from its start: a long generation that outlives the
/// idle timeout keeps the session warm instead of being dropped mid-recovery
/// of the prefix cache.
pub struct TurnUsageGuard<'a> {
    manager: &'a SessionManager,
}

impl<'a> TurnUsageGuard<'a> {
    pub fn new(manager: &'a SessionManager) -> Self {
        Self {
            manager,
        }
    }
}

impl Drop for TurnUsageGuard<'_> {
    fn drop(&mut self) {
        self.manager.mark_used();
    }
}

/// Whether two models describe the same source. Local (filesystem) models
/// are identified by directory name alone, so two checkpoints merely sharing
/// a name do NOT match, and a local checkpoint never matches a
/// registry/remote model (or vice versa) that happens to share its name —
/// only the on-disk path decides. Non-local models have identifiers that are
/// unique within their registry, so an identifier match is sufficient (this
/// also lets shorthand aliases like `llama:7b` resolve to a configured
/// `meta:llama:7b`).
fn matches_sources(
    a: &Model,
    b: &Model,
) -> bool {
    let is_local = |model: &Model| {
        matches!(
            model.accessibility,
            ModelAccessibility::OnDevice {
                source: ModelSource::Filesystem { .. }
            }
        )
    };
    if is_local(a) || is_local(b) {
        match (&a.accessibility, &b.accessibility) {
            (
                ModelAccessibility::OnDevice {
                    source: ModelSource::Filesystem {
                        path: a,
                    },
                },
                ModelAccessibility::OnDevice {
                    source: ModelSource::Filesystem {
                        path: b,
                    },
                },
            ) => paths_equal(a, b),
            // A local checkpoint and a registry/remote model that share a
            // name are distinct models.
            _ => false,
        }
    } else {
        a.identifier == b.identifier
    }
}

/// Whether two filesystem model paths name the same location, following the
/// platform's case semantics. A `canonicalize` round-trip resolves symlinks
/// (`/tmp` vs `/private/tmp`) and `.`/`..` segments while preserving the
/// on-disk case of the final component — so on a case-sensitive filesystem
/// `/models/Foo` and `/models/foo` are distinct, while on a case-insensitive
/// one (macOS APFS) they canonicalize to the same bytes and compare equal.
fn paths_equal(
    a: &str,
    b: &str,
) -> bool {
    let resolved = |path: &str| std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    resolved(a) == resolved(b)
}

/// Downloads (if missing) and loads `slot` into memory, installing the
/// session into `session_slot`. Called by `load_into` after the OOM guard
/// passes, so callers are guaranteed it fits.
///
/// The (potentially multi-second) download + load is raced against
/// [`LoadLiveness`]: if the client goes away, the in-flight load is dropped
/// (the engine keeps no per-model state, and downloads are resumable) and the
/// session lock is released instead of being held until the load finishes and
/// stalling every other model.
async fn load_model(
    manager: &SessionManager,
    slot: &ModelSlot,
    session_slot: &mut Option<ChatSession>,
    liveness: &mut impl LoadLiveness,
) -> Result<ChatSession, AcquireError> {
    tracing::info!("loading model into memory: {} (~{:.1} GB)", slot.identifier(), slot.footprint as f64 / 1e9);
    // The load future captures only `manager` and `slot` — NOT
    // `session_slot` or `liveness` — so it can be raced against liveness
    // without a double borrow, and dropped on disconnect without leaving the
    // session slot half-written.
    let mut load = Box::pin(async move {
        let downloader =
            manager.engine.download(&slot.model).await.map_err(|error| AcquireError::Engine(error.to_string()))?;
        while downloader.next().await.is_some() {}

        manager
            .engine
            .chat(slot.model.clone(), ChatConfig::default())
            .await
            .map_err(|error| AcquireError::Engine(error.to_string()))
    });
    let session = loop {
        tokio::select! {
            result = &mut load => break result?,
            alive = liveness.tick() => {
                if !alive {
                    return Err(AcquireError::ClientDisconnected);
                }
                // Client is still connected; keep racing until the load
                // finishes or the client goes away.
            },
        }
    };

    *manager.resident.lock().await = Some(slot.clone());
    *manager.last_used.lock().unwrap() = Instant::now();
    *session_slot = Some(session.clone());
    tracing::info!("model loaded: {}", slot.identifier());
    Ok(session)
}

pub struct ServerState {
    pub session_manager: Arc<SessionManager>,
    pub prefix_cache: bool,
}

#[cfg(test)]
mod tests {
    use shoji::types::{basic::Metadata, model::ModelRegistry};

    use super::*;

    #[test]
    fn filesystem_bytes_sums_nested_file_sizes() {
        let root = std::env::temp_dir().join(format!("uzu-fs-bytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("a.bin"), vec![0u8; 1024]).unwrap();
        std::fs::write(root.join("nested/b.bin"), vec![0u8; 2048]).unwrap();
        // A directory entry must not be counted as a file.
        std::fs::create_dir(root.join("nested/empty")).unwrap();

        assert_eq!(filesystem_bytes(root.to_str().unwrap()), 3072);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_bytes_counts_symlinked_weights() {
        let root = std::env::temp_dir().join(format!("uzu-fs-symlink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // A weight symlinked into the model dir must count its target's size.
        std::fs::write(root.join("model.safetensors"), vec![0u8; 4096]).unwrap();
        let weights_dir = root.join("weights");
        std::fs::create_dir_all(&weights_dir).unwrap();
        std::os::unix::fs::symlink(root.join("model.safetensors"), weights_dir.join("w.safetensors")).unwrap();
        // A broken symlink counts as 0, not as a file.
        std::os::unix::fs::symlink(root.join("does-not-exist"), weights_dir.join("broken")).unwrap();

        assert_eq!(filesystem_bytes(root.to_str().unwrap()), 4096);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_bytes_counts_symlinked_speculator_dir() {
        let root = std::env::temp_dir().join(format!("uzu-fs-speculator-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("model.safetensors"), vec![0u8; 1024]).unwrap();
        // The optional speculator keeps its weights behind a symlinked
        // directory; the walk must descend into it.
        let speculator = root.join("real_speculator");
        std::fs::create_dir_all(&speculator).unwrap();
        std::fs::write(speculator.join("model.safetensors"), vec![0u8; 2048]).unwrap();
        std::os::unix::fs::symlink(&speculator, root.join("speculator")).unwrap();

        assert_eq!(filesystem_bytes(root.to_str().unwrap()), 1024 + 2048);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_bytes_terminates_on_link_cycle() {
        let root = std::env::temp_dir().join(format!("uzu-fs-cycle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("w.bin"), vec![0u8; 512]).unwrap();
        // A directory link back to the root must not loop the walk.
        std::os::unix::fs::symlink(&root, root.join("loop")).unwrap();

        assert_eq!(filesystem_bytes(root.to_str().unwrap()), 512);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn filesystem_bytes_missing_path_is_zero() {
        assert_eq!(filesystem_bytes("/definitely/not/a/real/path"), 0);
    }

    /// A minimal `Model` carrying only the fields `matches_sources` reads.
    fn test_model(
        identifier: &str,
        accessibility: ModelAccessibility,
    ) -> Model {
        let registry = ModelRegistry {
            identifier: "mirai".to_string(),
            metadata: Metadata {
                identifier: "mirai".to_string(),
                name: "Mirai".to_string(),
                description: None,
                icons: vec![],
            },
        };
        Model {
            identifier: identifier.to_string(),
            registry,
            backends: vec![],
            family: None,
            properties: None,
            quantization: None,
            specializations: vec![],
            accessibility,
            encoding: None,
        }
    }

    fn fs_model(
        identifier: &str,
        path: &str,
    ) -> Model {
        test_model(
            identifier,
            ModelAccessibility::OnDevice {
                source: ModelSource::Filesystem {
                    path: path.to_string(),
                },
            },
        )
    }

    fn registry_model(identifier: &str) -> Model {
        test_model(
            identifier,
            ModelAccessibility::OnDevice {
                source: ModelSource::Registry {
                    toolchain_version: "test".to_string(),
                    repository: None,
                    source_repository: None,
                    files: vec![],
                },
            },
        )
    }

    fn remote_model(identifier: &str) -> Model {
        test_model(
            identifier,
            ModelAccessibility::Remote {
                repository: None,
            },
        )
    }

    /// A local checkpoint and a registry/remote model sharing its name are
    /// distinct models. Only two filesystem slots pointing at the same on-disk
    /// location ever match — an identifier alone never substitutes a
    /// checkpoint for a registry/remote model that happens to share a name.
    #[test]
    fn matches_sources_rejects_mixed_sources() {
        let fs = fs_model("llama:7b", "/tmp/a");
        let reg = registry_model("llama:7b");
        let rem = remote_model("llama:7b");

        // filesystem vs registry: reject, even when the identifier matches
        assert!(!matches_sources(&fs, &reg));
        assert!(!matches_sources(&reg, &fs));
        // filesystem vs remote: reject, even when the identifier matches
        assert!(!matches_sources(&fs, &rem));
        assert!(!matches_sources(&rem, &fs));
    }

    /// Two registry slots match on identifier alone (the local registry
    /// resolves shorthands like `llama:7b` → `meta:llama:7b`, so an
    /// identifier-only comparison is how shorthand resolution lands on the
    /// configured slot).
    #[test]
    fn matches_sources_registry_by_identifier() {
        assert!(matches_sources(&registry_model("a"), &registry_model("a")));
        assert!(!matches_sources(&registry_model("a"), &registry_model("b")));
    }

    /// Two remote slots match on identifier alone (remote IDs are unique).
    #[test]
    fn matches_sources_remote_by_identifier() {
        assert!(matches_sources(&remote_model("openai:gpt-4"), &remote_model("openai:gpt-4")));
        assert!(!matches_sources(&remote_model("openai:gpt-4"), &remote_model("openai:gpt-5")));
    }

    /// Filesystem slots compare on the on-disk path, not the identifier —
    /// so two checkpoints that merely share a name do NOT match.
    #[test]
    fn matches_sources_filesystem_by_path() {
        assert!(matches_sources(&fs_model("llama", "/tmp/llama-a"), &fs_model("llama", "/tmp/llama-a")));
        assert!(!matches_sources(&fs_model("llama", "/tmp/llama-a"), &fs_model("llama", "/tmp/llama-b")));
    }

    /// Symlinks to the same model directory must compare equal (e.g. `/tmp`
    /// vs `/private/tmp` on macOS).
    #[cfg(unix)]
    #[test]
    fn paths_equal_follows_symlinks() {
        let real = std::env::temp_dir().join(format!("uzu-pe-real-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&real);
        std::fs::create_dir_all(&real).unwrap();
        let link = std::env::temp_dir().join(format!("uzu-pe-link-{}", std::process::id()));
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&real, &link).unwrap();

        assert!(paths_equal(real.to_str().unwrap(), link.to_str().unwrap()));
        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_dir_all(&real);
    }
}
