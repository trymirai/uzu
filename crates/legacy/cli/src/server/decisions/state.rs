use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::session::decision::DecisionSession;

pub struct DecisionState {
    // Locked for a whole request: the session refuses a second decision while one runs.
    pub session: Arc<Mutex<DecisionSession>>,
}
