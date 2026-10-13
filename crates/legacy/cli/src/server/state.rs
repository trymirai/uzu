use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::sync::Mutex;
use uzu::{
    engine::Engine,
    session::chat::ChatSession,
    types::{
        model::{Model, ModelAccessibility, ModelSource},
        session::chat::ChatConfig,
    },
};
use walkdir::WalkDir;

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
/// which is what unified memory actually maps. File symlinks resolve to
/// their target, which may lie outside the directory (the engine opens
/// weights with `File::open`, which follows links; symlinks are a common way
/// to deduplicate weights). Directory symlinks are *not* followed: model
/// layouts are plain trees, and never descending through links keeps the
/// walk bounded even if a model directory contains a cycle. Broken or
/// unreadable entries count as 0; the dynamic free-memory check in
/// `memory_guard` still bounds what actually gets allocated.
fn filesystem_bytes(path: &str) -> u64 {
    let root = Path::new(path);
    if !root.is_dir() {
        return 0;
    }
    let mut seen: std::collections::HashSet<std::path::PathBuf> = std::collections::HashSet::new();
    WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file() || entry.file_type().is_symlink())
        .filter_map(|entry| {
            let resolved = std::fs::canonicalize(entry.path()).ok()?;
            let metadata = std::fs::metadata(resolved.clone()).ok()?;
            metadata.is_file().then_some((resolved, metadata.len()))
        })
        .filter_map(|(resolved, len)| seen.insert(resolved).then_some(len))
        .fold(0_u64, u64::saturating_add)
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
        self.slots.iter().find(|slot| slot.model.identifier == model.identifier).cloned().ok_or_else(|| {
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
    ) -> Result<ChatSession, AcquireError> {
        // A turn has started: the idle timer must not fire for this turn's
        // duration (a turn can outlive the idle timeout itself).
        *self.last_used.lock().unwrap() = Instant::now();

        // Fast path: the requested model is already the resident one and a
        // live session is present — nothing to do.
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

        // OOM guard using live available memory. After a swap the OS may need
        // a moment to reclaim the just-freed pages, so retry briefly.
        let mut guard_error =
            match memory_guard(slot.identifier(), slot.footprint, available_memory(), self.total_memory) {
                Ok(()) => return load_model(self, slot, session_slot).await,
                Err(error) => error,
            };
        for _ in 0..10 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            match memory_guard(slot.identifier(), slot.footprint, available_memory(), self.total_memory) {
                Ok(()) => return load_model(self, slot, session_slot).await,
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

/// Downloads (if missing) and loads `slot` into memory, installing the
/// session into `session_slot`. Called by `load_into` after the OOM guard
/// passes, so callers are guaranteed it fits.
async fn load_model(
    manager: &SessionManager,
    slot: &ModelSlot,
    session_slot: &mut Option<ChatSession>,
) -> Result<ChatSession, AcquireError> {
    tracing::info!("loading model into memory: {} (~{:.1} GB)", slot.identifier(), slot.footprint as f64 / 1e9);
    let downloader =
        manager.engine.download(&slot.model).await.map_err(|error| AcquireError::Engine(error.to_string()))?;
    while let Some(_update) = downloader.next().await {}

    let session = manager
        .engine
        .chat(slot.model.clone(), ChatConfig::default())
        .await
        .map_err(|error| AcquireError::Engine(error.to_string()))?;

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

    #[test]
    fn filesystem_bytes_missing_path_is_zero() {
        assert_eq!(filesystem_bytes("/definitely/not/a/real/path"), 0);
    }
}
