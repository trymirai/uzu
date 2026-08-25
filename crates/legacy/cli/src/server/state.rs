use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::session::chat::ChatSession;

use crate::common::model_capabilities::ThinkingSupport;

pub struct ServerState {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
    pub thinking_support: ThinkingSupport,
    pub prefix_cache: bool,
}
