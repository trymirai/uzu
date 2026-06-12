use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::session::chat::ChatSession;

pub struct ServerState {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
}
