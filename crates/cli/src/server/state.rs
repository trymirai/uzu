use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::session::chat::ChatSession;

/// Shared state for the OpenAI-compatible server.
///
/// A single [`ChatSession`] is loaded at startup and shared across all
/// requests. OpenAI semantics are stateless (each request carries the full
/// conversation), so handlers `reset()` the session before each request. The
/// `Mutex` serializes generation, since a `ChatSession` rejects concurrent runs.
pub struct ServerState {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
}
