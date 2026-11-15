use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use uzu::{engine::Engine, session::decision::DecisionSession};
use uzu_engine::d1_omni::D1OmniPipeline;

#[derive(Clone)]
pub enum DecisionBackend {
    Legacy(Arc<Mutex<DecisionSession>>),
    D1Omni(Arc<Mutex<D1OmniPipeline>>),
}

pub struct DecisionState {
    pub engine: Arc<Engine>,
    // Locked for a whole request: the session refuses a second decision while one runs.
    pub backend: Arc<RwLock<Option<(String, DecisionBackend)>>>,
    pub load_gate: Arc<Mutex<()>>,
}
