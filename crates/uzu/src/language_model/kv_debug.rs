#[derive(Debug, Clone)]
pub struct KvDebugLayerSnapshot {
    pub layer_index: usize,
    pub positions: Vec<usize>,
    pub sparse_recent_positions: Option<Vec<usize>>,
    pub sparse_pending_positions: Option<Vec<usize>>,
    pub num_groups: usize,
    pub head_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub sparse_recent_values: Option<Vec<f32>>,
    pub sparse_pending_length: usize,
    pub sparse_pending_values: Option<Vec<f32>>,
    pub storage_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct KvDebugSnapshot {
    pub layers: Vec<KvDebugLayerSnapshot>,
}
