//! The single model slot: at most one model is resident in the server process.
//!
//! The server boots idle (or with `--model` preloaded) and models are swapped
//! through the admin API (`POST /v1/models/load`, `POST /v1/models/unload`).
//! The slot is the single source of truth for what is resident; chat requests
//! take a snapshot and either proceed against the ready model or fail with 409.
//!
//! Memory-safety semantics: `Ready` holds an `Arc<LoadedModel>`. Handlers clone
//! that Arc for the lifetime of their request, so an `unload` (or a swap that
//! drops the slot's Arc) never yanks a model out from under an in-flight
//! generation — GPU memory is reclaimed when the last request finishes. The
//! swap path additionally waits on the text session mutex before loading a new
//! model, so the old model's weights are released before the new ones allocate
//! in the common case.

use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use hanashi::chat::hanashi::config::HanashiConfig;
use tokio::sync::{Mutex, RwLock};
use uzu::session::chat::ChatSession;

use crate::common::{audio::AudioRuntime, thinking::ThinkingSupport};
use crate::server::vlm_session::VlmSessionManager;

/// Everything that is alive for one resident model — the per-model state that
/// used to live directly on `ChatState` (minus the server-level prefix-cache
/// flag and CLI parameters).
pub struct LoadedModel {
    /// Resolved model identifier (directory name for local models); matches
    /// the OpenAI wire `model` field in requests and responses.
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
    pub thinking_support: ThinkingSupport,
    pub vlm_pipeline: Option<Arc<uzu_engine::vision::AnyVlmPipeline>>,
    pub vlm_sessions: Arc<VlmSessionManager>,
    pub audio: Option<Arc<Mutex<AudioRuntime>>>,
    /// The model's hanashi chat config (`encoding.json` or the local registry's
    /// ChatML/qwen3.5 fallback); the VLM reply parser runs the same parsing stack.
    pub reply_encoding: Option<HanashiConfig>,
    /// Session context length in tokens, applied to both the text session and
    /// newly acquired VLM states.
    pub max_context_tokens: u32,
    /// The language model behind the resident session; raw completions
    /// (`/v1/completions`) drive it directly, bypassing the chat session.
    pub lm: std::sync::Arc<uzu_engine::engine::language_model::LanguageModel<uzu_engine::backends::metal::Metal>>,
    /// Wall-clock timestamp of when the model became ready (RFC 3339 via chrono).
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    /// Monotonic timestamp for uptime reporting.
    pub loaded_at_instant: Instant,
}

/// Cheap clone of the slot state for handlers; never holds the slot lock
/// across an await. `Ready` carries the `Arc` so a snapshot taken before an
/// unload keeps the model alive for the rest of the request.
#[derive(Clone)]
pub enum SlotSnapshot {
    Empty,
    Loading {
        model: String,
        started_at: Instant,
    },
    Ready(Arc<LoadedModel>),
    Failed {
        model: String,
        error: String,
        /// Wall-clock timestamp of the failure, for status reporting.
        at: DateTime<Utc>,
    },
}

impl SlotSnapshot {
    /// `"idle" | "loading" | "ready" | "failed"`.
    pub fn status_label(&self) -> &'static str {
        match self {
            SlotSnapshot::Empty => "idle",
            SlotSnapshot::Loading { .. } => "loading",
            SlotSnapshot::Ready(_) => "ready",
            SlotSnapshot::Failed { .. } => "failed",
        }
    }

    /// The model the slot is working with (loading, ready, or last failed), if any.
    pub fn model_name(&self) -> Option<&str> {
        match self {
            SlotSnapshot::Empty => None,
            SlotSnapshot::Loading { model, .. } => Some(model),
            SlotSnapshot::Ready(loaded) => Some(&loaded.model_name),
            SlotSnapshot::Failed { model, .. } => Some(model),
        }
    }
}

pub struct SwapSlot(RwLock<SlotSnapshot>);

impl SwapSlot {
    pub fn new() -> Self {
        Self(RwLock::new(SlotSnapshot::Empty))
    }

    pub async fn snapshot(&self) -> SlotSnapshot {
        self.0.read().await.clone()
    }

    /// Transition into `Loading { model }` and return the previous snapshot.
    /// Swap logic uses the returned `Ready` Arc to drain the old model.
    pub async fn begin_loading(&self, model: String) -> SlotSnapshot {
        let mut guard = self.0.write().await;
        let previous = guard.clone();
        *guard = SlotSnapshot::Loading {
            model,
            started_at: Instant::now(),
        };
        previous
    }

    /// `Loading -> Ready`. Called by the load task when the model is resident.
    pub async fn set_ready(&self, loaded: LoadedModel) {
        let mut guard = self.0.write().await;
        if let SlotSnapshot::Loading {
            model: expected, ..
        } = &*guard
        {
            if *expected != loaded.model_name {
                tracing::warn!(
                    "slot transition Loading('{expected}') -> Ready('{}'): unexpected overwrite",
                    loaded.model_name
                );
            }
        } else {
            tracing::warn!(
                "slot transition -> Ready('{}') from unexpected state {}",
                loaded.model_name,
                guard.status_label()
            );
        }
        *guard = SlotSnapshot::Ready(Arc::new(loaded));
    }

    /// `Loading -> Failed`. Only records the failure if the slot is still
    /// loading `model` (a concurrent transition must not be clobbered).
    pub async fn set_failed(&self, model: String, error: String) {
        let mut guard = self.0.write().await;
        match &*guard {
            SlotSnapshot::Loading {
                model: expected, ..
            } if *expected == model => {
                tracing::warn!("❌ Model load failed ({model}): {error}");
                *guard = SlotSnapshot::Failed {
                    model,
                    error,
                    at: chrono::Utc::now(),
                };
            },
            other => {
                tracing::warn!(
                    "ignored load failure for '{model}' (slot is {}): {error}",
                    other.status_label()
                );
            },
        }
    }

    /// Clear the slot to `Empty`; returns the resident model if there was one
    /// so the caller can drain it. Idempotent: `Empty` and `Failed` clear to
    /// `Empty` and return `None`; `Loading` is left untouched (the admin route
    /// rejects unload during a load before reaching here).
    pub async fn unload(&self) -> Option<Arc<LoadedModel>> {
        let mut guard = self.0.write().await;
        match &*guard {
            SlotSnapshot::Ready(loaded) => {
                let loaded = Arc::clone(loaded);
                *guard = SlotSnapshot::Empty;
                Some(loaded)
            },
            SlotSnapshot::Failed { .. } => {
                *guard = SlotSnapshot::Empty;
                None
            },
            _ => None,
        }
    }
}
