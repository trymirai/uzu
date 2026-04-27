use std::{
    collections::HashMap,
    sync::{Arc, atomic::AtomicU64},
};

use tokio::sync::{Mutex, Notify, RwLock};

use crate::common::mock_download_server::{RequestRecord, mock_route::MockRoute};

#[derive(Debug)]
pub(super) struct ServerState {
    pub routes: RwLock<HashMap<String, MockRoute>>,
    pub json_routes: RwLock<HashMap<String, String>>,
    pub records: Mutex<Vec<RequestRecord>>,
    pub bytes_sent_by_path: Mutex<HashMap<String, u64>>,
    pub request_counts_by_route: Mutex<HashMap<(String, String), u64>>,
    pub stall_notifies_by_path: Mutex<HashMap<String, Arc<Notify>>>,
    pub next_order: AtomicU64,
}
